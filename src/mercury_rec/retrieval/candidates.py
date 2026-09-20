"""Multi-source candidate generation and fusion.

Retrieval narrows 36,044 items to a few hundred the ranker can afford to score
properly. Several sources are used rather than one because they fail in
different, complementary ways:

- **Two-tower** generalises through content features, so it can surface items
  a user has no collaborative path to -- including brand-new ones.
- **Item-CF** captures co-occurrence the neural model smooths over, and is
  strongest exactly where a user has rich history.
- **Popularity** is the floor. It always returns something sensible, which is
  what makes it the cold-start and fallback path.

The union is what matters: a candidate missed by every source can never be
recommended, no matter how good the ranker is. **Retrieval recall is therefore
the ceiling on the entire system**, which is why it is measured separately
from end-to-end quality rather than inferred from it.

Fusion uses Reciprocal Rank Fusion because the sources produce scores on
incomparable scales -- a cosine similarity in [-1, 1], a summed CF similarity
in the hundreds, a decayed popularity count in the thousands. Normalising them
onto a shared scale means choosing a normalisation per source, and every such
choice is an unjustified hyperparameter. RRF uses only ranks, which are
directly comparable, and is robust to one source producing wild magnitudes.
"""

from __future__ import annotations

import time
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mercury_rec.core.enums import RetrievalSource
from mercury_rec.core.logging import get_logger

logger = get_logger(__name__)

#: RRF damping constant from the original paper. It flattens the contribution
#: of the very top ranks so one confident source cannot dominate the fusion.
DEFAULT_RRF_K = 60


@dataclass(slots=True)
class SourceResult:
    """What one retrieval source returned, and how long it took."""

    source: RetrievalSource
    item_ids: np.ndarray
    scores: np.ndarray
    elapsed_ms: float

    def __len__(self) -> int:
        return len(self.item_ids)


@dataclass(slots=True)
class CandidateSet:
    """The fused candidate pool handed to the ranker."""

    item_ids: np.ndarray
    fused_scores: np.ndarray
    per_source_scores: dict[str, np.ndarray]
    """Per-source score per candidate, NaN where that source did not return
    it. Carried into the ranker as features and into the API response as
    provenance, so a candidate's origin stays visible end to end."""

    source_counts: dict[str, int]
    sources_per_item: np.ndarray
    timings_ms: dict[str, float] = field(default_factory=dict)

    def __len__(self) -> int:
        return len(self.item_ids)

    def summary(self) -> dict[str, Any]:
        return {
            "n_candidates": len(self.item_ids),
            "per_source": self.source_counts,
            "mean_sources_per_item": (
                round(float(self.sources_per_item.mean()), 2) if len(self.item_ids) else 0.0
            ),
            "timings_ms": {k: round(v, 3) for k, v in self.timings_ms.items()},
        }


def reciprocal_rank_fusion(
    results: list[SourceResult],
    *,
    k_rrf: int = DEFAULT_RRF_K,
    max_candidates: int = 500,
    exclude: set[int] | None = None,
) -> CandidateSet:
    """Fuse ranked lists from several sources into one candidate pool.

    Args:
        results: One entry per source.
        k_rrf: RRF damping constant.
        max_candidates: Pool size cap. The ranker's cost is linear in this,
            so it is the main retrieval/ranking budget dial.
        exclude: Items to drop, normally the user's own history.

    Returns:
        The fused pool, ordered best first.
    """
    excluded = exclude or set()
    fused: dict[int, float] = {}
    appearances: dict[int, int] = {}
    per_source: dict[str, dict[int, float]] = {}
    counts: dict[str, int] = {}
    timings: dict[str, float] = {}

    for result in results:
        name = result.source.value
        counts[name] = len(result)
        timings[name] = result.elapsed_ms
        scores: dict[int, float] = {}

        for rank, (item_id, score) in enumerate(
            zip(result.item_ids.tolist(), result.scores.tolist(), strict=True), start=1
        ):
            item = int(item_id)
            if item in excluded or item < 0:
                # FAISS pads short results with -1 when a query returns fewer
                # than k neighbours; those are not items.
                continue
            fused[item] = fused.get(item, 0.0) + 1.0 / (k_rrf + rank)
            appearances[item] = appearances.get(item, 0) + 1
            scores[item] = float(score)

        per_source[name] = scores

    if not fused:
        empty = np.array([], dtype=np.int64)
        return CandidateSet(
            item_ids=empty,
            fused_scores=np.array([], dtype=np.float32),
            per_source_scores={name: np.array([], dtype=np.float32) for name in per_source},
            source_counts=counts,
            sources_per_item=np.array([], dtype=np.int32),
            timings_ms=timings,
        )

    ordered = sorted(fused, key=lambda item: -fused[item])[:max_candidates]
    item_ids = np.array(ordered, dtype=np.int64)

    return CandidateSet(
        item_ids=item_ids,
        fused_scores=np.array([fused[item] for item in ordered], dtype=np.float32),
        per_source_scores={
            name: np.array([scores.get(item, np.nan) for item in ordered], dtype=np.float32)
            for name, scores in per_source.items()
        },
        source_counts=counts,
        sources_per_item=np.array([appearances[item] for item in ordered], dtype=np.int32),
        timings_ms=timings,
    )


def retrieve_from_index(
    index: Any,
    query: np.ndarray,
    *,
    source: RetrievalSource,
    k: int,
) -> SourceResult:
    """Query a vector index, timing it for the per-stage telemetry."""
    started = time.perf_counter()
    scores, ids = index.search(query.reshape(1, -1), k)
    elapsed_ms = (time.perf_counter() - started) * 1000.0
    return SourceResult(
        source=source,
        item_ids=np.asarray(ids[0], dtype=np.int64),
        scores=np.asarray(scores[0], dtype=np.float32),
        elapsed_ms=elapsed_ms,
    )


def retrieve_from_scores(
    scores: np.ndarray,
    *,
    source: RetrievalSource,
    k: int,
    elapsed_ms: float = 0.0,
) -> SourceResult:
    """Take the top-``k`` from a dense score vector over the catalogue.

    Used by sources that score the whole catalogue directly (popularity,
    item-CF) rather than through a vector index.
    """
    top_k = min(k, len(scores))
    if top_k == 0:
        return SourceResult(
            source=source,
            item_ids=np.array([], dtype=np.int64),
            scores=np.array([], dtype=np.float32),
            elapsed_ms=elapsed_ms,
        )

    partitioned = np.argpartition(-scores, top_k - 1)[:top_k]
    ordered = partitioned[np.argsort(-scores[partitioned], kind="stable")]
    return SourceResult(
        source=source,
        item_ids=ordered.astype(np.int64),
        scores=scores[ordered].astype(np.float32),
        elapsed_ms=elapsed_ms,
    )


def retrieval_recall(candidates: np.ndarray, relevant: set[int]) -> float:
    """Share of relevant items present in the candidate pool.

    **The ceiling on the whole system.** An item retrieval never proposes
    cannot be ranked, re-ranked or recommended, so this bounds end-to-end
    recall regardless of how good the ranker is. Reported separately for
    exactly that reason: an end-to-end number alone cannot tell you whether
    to invest in retrieval or in ranking.
    """
    if not relevant:
        return 0.0
    present = len(set(candidates.tolist()) & relevant)
    return present / len(relevant)


__all__ = [
    "DEFAULT_RRF_K",
    "CandidateSet",
    "SourceResult",
    "reciprocal_rank_fusion",
    "retrieval_recall",
    "retrieve_from_index",
    "retrieve_from_scores",
]
