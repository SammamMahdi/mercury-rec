"""Item-to-item collaborative filtering on weighted implicit feedback.

Scoring is the standard neighbourhood formulation::

    score(u, i) = sum over j in history(u) of  w(u, j) * sim(i, j)

where ``sim`` is cosine similarity between item columns of the weighted
user-item matrix and ``w(u, j)`` is the user's confidence in item *j*.

Three implementation decisions carry most of the weight:

**Truncated neighbourhoods.** A dense 36,044 x 36,044 similarity matrix is
1.3 billion float32 cells, about 5.2 GB - more than a third of this machine's
RAM, for a matrix that is almost entirely noise. Only the top ``n_neighbors``
similarities per item are kept. Beyond a few hundred neighbours the values are
indistinguishable from zero and contribute nothing but memory traffic.

**Block-wise computation.** Even computing the full matrix transiently would
spike memory, so similarities are produced in column blocks and truncated
before the next block is started. Peak memory is
``block_size x n_items x 4`` bytes rather than ``n_items^2 x 4``.

**Popularity damping.** Raw cosine still favours popular items, because a
blockbuster co-occurs with everything. Dividing by ``popularity^alpha``
(Deshpande & Karypis) tunes that down. ``alpha = 0`` recovers plain cosine,
so the effect is measurable rather than assumed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from scipy import sparse

from mercury_rec.config.loader import load_config
from mercury_rec.config.schemas import EventsConfig
from mercury_rec.core.logging import get_logger
from mercury_rec.models.base import RecommendationContext, Recommender
from mercury_rec.models.popularity import event_weights

logger = get_logger(__name__)


class ItemCFRecommender(Recommender):
    """Item-item CF with cosine similarity over weighted implicit events.

    Args:
        n_users: Catalogue user count.
        n_items: Catalogue item count.
        n_neighbors: Similarities retained per item.
        popularity_damping: Exponent ``alpha`` in the ``popularity^alpha``
            denominator. 0 disables damping.
        min_similarity: Similarities below this are dropped as noise.
        block_size: Items per similarity block; trades speed for peak memory.
    """

    name = "item_cf"

    def __init__(
        self,
        n_users: int,
        n_items: int,
        *,
        n_neighbors: int = 200,
        popularity_damping: float = 0.5,
        min_similarity: float = 1e-4,
        block_size: int = 2_000,
    ) -> None:
        super().__init__(n_users, n_items)
        self.n_neighbors = n_neighbors
        self.popularity_damping = popularity_damping
        self.min_similarity = min_similarity
        self.block_size = block_size

        self._similarity: sparse.csr_matrix | None = None
        self._user_history: sparse.csr_matrix | None = None
        self._weights = event_weights()
        self._min_event_weight = float(load_config("events", EventsConfig).cf_min_event_weight)

    def params(self) -> dict[str, Any]:
        return {
            "n_neighbors": self.n_neighbors,
            "popularity_damping": self.popularity_damping,
            "min_similarity": self.min_similarity,
            "cf_min_event_weight": self._min_event_weight,
        }

    def _build_matrix(self, interactions: pd.DataFrame) -> sparse.csr_matrix:
        """Build the weighted user-item matrix from the training window.

        Low-intent events are dropped: without a floor, co-occurrence is
        dominated by incidental views and the similarities lose their
        discriminative power entirely.
        """
        weights = (
            interactions["event_type"].map(self._weights).fillna(0.0).to_numpy(dtype=np.float32)
        )
        keep = weights >= self._min_event_weight
        if not keep.any():
            raise ValueError(
                f"No interactions meet cf_min_event_weight={self._min_event_weight}. "
                "Lower it in configs/events.yaml."
            )

        users = interactions["user_id"].to_numpy(dtype=np.int32)[keep]
        items = interactions["item_id"].to_numpy(dtype=np.int32)[keep]
        values = weights[keep]

        matrix = sparse.csr_matrix(
            (values, (users, items)), shape=(self.n_users, self.n_items), dtype=np.float32
        )
        # A user who interacted with the same item repeatedly produces
        # duplicate coordinates; sum_duplicates folds them into one entry so
        # repeated engagement raises confidence rather than corrupting shape.
        matrix.sum_duplicates()
        logger.info(
            "model.item_cf.matrix",
            rows_kept=int(keep.sum()),
            rows_dropped=int((~keep).sum()),
            nnz=matrix.nnz,
        )
        return matrix

    def _fit(self, interactions: pd.DataFrame) -> dict[str, Any]:
        matrix = self._build_matrix(interactions)
        self._user_history = matrix

        # L2-normalise item columns so a dot product IS cosine similarity.
        norms = np.sqrt(matrix.multiply(matrix).sum(axis=0)).A1.astype(np.float32)
        norms[norms == 0] = 1.0
        normalised = matrix.multiply(sparse.csr_matrix(1.0 / norms)).tocsc().astype(np.float32)

        popularity = np.asarray((matrix > 0).sum(axis=0)).ravel().astype(np.float32)
        damping = np.power(np.maximum(popularity, 1.0), self.popularity_damping)

        rows: list[np.ndarray] = []
        cols: list[np.ndarray] = []
        vals: list[np.ndarray] = []

        transposed = normalised.T.tocsr()
        n_blocks = 0
        for start in range(0, self.n_items, self.block_size):
            end = min(start + self.block_size, self.n_items)
            # (block x n_items) similarities for this slice of items.
            block = (transposed[start:end] @ normalised).toarray()

            # An item is trivially its own nearest neighbour; keeping it would
            # just re-recommend the user's own history.
            for local, item in enumerate(range(start, end)):
                block[local, item] = 0.0

            block /= damping[np.newaxis, :]
            block[block < self.min_similarity] = 0.0

            keep = min(self.n_neighbors, block.shape[1] - 1)
            if keep > 0:
                top = np.argpartition(-block, keep - 1, axis=1)[:, :keep]
                row_index = np.repeat(np.arange(block.shape[0]), keep)
                flat = top.ravel()
                values = block[row_index, flat]
                nonzero = values > 0
                rows.append(row_index[nonzero] + start)
                cols.append(flat[nonzero])
                vals.append(values[nonzero])
            n_blocks += 1

        self._similarity = sparse.csr_matrix(
            (
                np.concatenate(vals) if vals else np.array([], dtype=np.float32),
                (
                    np.concatenate(rows) if rows else np.array([], dtype=np.int32),
                    np.concatenate(cols) if cols else np.array([], dtype=np.int32),
                ),
            ),
            shape=(self.n_items, self.n_items),
            dtype=np.float32,
        )

        density = self._similarity.nnz / max(self.n_items**2, 1)
        return {
            "similarity_nnz": int(self._similarity.nnz),
            "similarity_density": round(float(density), 8),
            "blocks": n_blocks,
            "mean_neighbors": round(float(self._similarity.nnz / max(self.n_items, 1)), 2),
        }

    def _score(
        self, user_id: int, candidates: np.ndarray, context: RecommendationContext
    ) -> np.ndarray:
        assert self._similarity is not None
        assert self._user_history is not None

        history = self._user_history[user_id]
        if history.nnz == 0:
            # No history means no neighbourhood signal. Returning zeros lets
            # the caller fall through to the cold-start path rather than
            # inventing a ranking from nothing.
            return np.zeros(len(candidates), dtype=np.float32)

        # (1 x n_items) sparse row of accumulated similarity to the history.
        scores = history @ self._similarity
        dense = np.asarray(scores.todense()).ravel()

        # Augment with the session's items so within-session context shifts
        # the ranking, which is what the contextual demo depends on.
        if context.session_items:
            session = np.fromiter(context.session_items, dtype=np.int64)
            session = session[(session >= 0) & (session < self.n_items)]
            if session.size:
                dense = dense + np.asarray(self._similarity[session].sum(axis=0)).ravel()

        selected: np.ndarray = dense[candidates].astype(np.float32)
        return selected

    def similar_items(self, item_id: int, k: int = 10) -> list[tuple[int, float]]:
        """Nearest neighbours of an item, powering ``/items/{id}/similar``."""
        self._require_fitted()
        assert self._similarity is not None
        row = self._similarity[item_id]
        if row.nnz == 0:
            return []
        order = np.argsort(-row.data)[:k]
        return [(int(row.indices[i]), float(row.data[i])) for i in order]


__all__ = ["ItemCFRecommender"]
