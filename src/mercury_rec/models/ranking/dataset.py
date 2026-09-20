"""Build the learning-to-rank training set from retrieval output.

A ranker is not trained on interactions. It is trained on **the candidate
lists retrieval actually produces**, because that is the only distribution it
will ever see at serving time. Training it on a different distribution -- say,
positives against items sampled uniformly from the catalogue -- teaches it to
separate relevant items from obviously irrelevant ones, which retrieval has
already done. It then contributes nothing on the hard distinction it exists
to make: which of 300 plausible candidates is best.

So each training group is::

    (user, as_of) -> [ candidate_1 ... candidate_N ], labels, features

with candidates drawn from the same retrieval fusion used in production, and
labels marking which of them the user actually went on to interact with.

Group construction
------------------
LightGBM's LambdaRank optimises *within* a group, so the groups must be the
unit the metric cares about: one user's candidate list at one moment. Getting
this wrong -- for example one group per user across all time -- optimises a
ranking no request ever asks for.

Sampling
--------
Groups with no positive at all carry no ranking signal: every pairwise
comparison inside them is between two negatives, and LambdaRank's gradient is
identically zero. They are dropped, which is both correct and a large speed-up.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np
import pandas as pd

from mercury_rec.core.logging import get_logger
from mercury_rec.features.asof import FEATURE_COLUMNS

logger = get_logger(__name__)

#: Ranking features = the as-of feature vector plus retrieval-stage signals.
#: The retrieval scores matter: they carry collaborative information the
#: tabular features do not, and letting the ranker calibrate against them is
#: much of the value of a two-stage system.
RETRIEVAL_FEATURES: tuple[str, ...] = (
    "retrieval_score_two_tower",
    "retrieval_score_item_cf",
    "retrieval_score_popularity",
    "retrieval_rank_fused",
    "retrieval_n_sources",
)

RANKING_FEATURES: tuple[str, ...] = (*FEATURE_COLUMNS, *RETRIEVAL_FEATURES)


@dataclass(slots=True)
class RankingDataset:
    """A LightGBM-ready ranking dataset.

    Attributes:
        features: ``(n_rows, n_features)`` float32.
        labels: ``(n_rows,)`` binary relevance.
        groups: Row count per group, in order. LightGBM consumes this rather
            than group ids, and it must sum to ``len(labels)``.
        user_ids: ``(n_rows,)``, kept for slicing metrics by cohort.
        item_ids: ``(n_rows,)``, kept for diversity and coverage reporting.
        feature_names: Column order, so SHAP and the API agree with training.
    """

    features: np.ndarray
    labels: np.ndarray
    groups: np.ndarray
    user_ids: np.ndarray
    item_ids: np.ndarray
    feature_names: tuple[str, ...] = RANKING_FEATURES

    def __post_init__(self) -> None:
        if self.groups.sum() != len(self.labels):
            raise ValueError(
                f"Group sizes sum to {self.groups.sum()} but there are "
                f"{len(self.labels)} rows. LightGBM would silently mis-align "
                "groups with labels and optimise nonsense."
            )
        if self.features.shape[0] != len(self.labels):
            raise ValueError(
                f"Feature matrix has {self.features.shape[0]} rows against "
                f"{len(self.labels)} labels."
            )

    @property
    def n_groups(self) -> int:
        return len(self.groups)

    @property
    def positive_rate(self) -> float:
        return float(self.labels.mean()) if len(self.labels) else 0.0

    def summary(self) -> dict[str, Any]:
        return {
            "rows": len(self.labels),
            "groups": self.n_groups,
            "positives": int(self.labels.sum()),
            "positive_rate": round(self.positive_rate, 5),
            "mean_group_size": round(float(self.groups.mean()), 2) if self.n_groups else 0.0,
            "n_features": self.features.shape[1],
        }


def build_ranking_dataset(
    candidates: pd.DataFrame,
    *,
    drop_groups_without_positives: bool = True,
) -> RankingDataset:
    """Assemble a ranking dataset from a scored candidate frame.

    Args:
        candidates: One row per (user, candidate item), carrying every column
            in :data:`RANKING_FEATURES`, plus ``user_id``, ``item_id`` and
            ``label``. Must be sorted by user so groups are contiguous --
            LightGBM assumes contiguity and will silently mis-group otherwise.
        drop_groups_without_positives: Remove groups with no relevant item.
            LambdaRank's gradient is identically zero on them, so they cost
            time and contribute nothing.

    Returns:
        The assembled dataset.
    """
    missing = [name for name in RANKING_FEATURES if name not in candidates.columns]
    if missing:
        raise ValueError(f"Candidate frame is missing ranking features: {missing}")
    if "label" not in candidates.columns:
        raise ValueError("Candidate frame must carry a 'label' column.")

    frame = candidates.sort_values(["user_id"], kind="stable").reset_index(drop=True)

    if drop_groups_without_positives:
        positives_per_user = frame.groupby("user_id")["label"].transform("sum")
        before_groups = frame["user_id"].nunique()
        frame = frame[positives_per_user > 0].reset_index(drop=True)
        logger.info(
            "ranking.dataset.filtered",
            groups_before=before_groups,
            groups_after=int(frame["user_id"].nunique()) if len(frame) else 0,
        )

    if frame.empty:
        raise ValueError(
            "No ranking groups survived. Either retrieval never surfaced a "
            "relevant item, or the labels are misaligned."
        )

    group_sizes = frame.groupby("user_id", sort=False).size().to_numpy()

    dataset = RankingDataset(
        features=frame[list(RANKING_FEATURES)].to_numpy(dtype=np.float32),
        labels=frame["label"].to_numpy(dtype=np.int32),
        groups=group_sizes,
        user_ids=frame["user_id"].to_numpy(dtype=np.int64),
        item_ids=frame["item_id"].to_numpy(dtype=np.int64),
    )
    logger.info("ranking.dataset.built", **dataset.summary())
    return dataset


def fuse_candidate_scores(
    per_source: dict[str, dict[int, float]],
    *,
    k_rrf: int = 60,
) -> pd.DataFrame:
    """Merge several retrieval sources into one candidate frame.

    Uses Reciprocal Rank Fusion: each source contributes ``1 / (k + rank)``.

    RRF rather than a weighted score sum because the sources produce
    incomparable scales -- a cosine similarity in [-1, 1], a summed CF
    similarity in the hundreds, and a decayed popularity count in the
    thousands. Normalising those onto a common scale requires choosing a
    normalisation per source, and every such choice is a hyperparameter that
    has to be justified. RRF uses only the ranks, which are directly
    comparable, and is robust to one source producing wild magnitudes.

    ``k_rrf = 60`` is the value from the original RRF paper; it damps the
    influence of the very top ranks so a single confident source cannot
    dominate the fusion outright.

    Args:
        per_source: source name -> {item_id: score}.
        k_rrf: RRF damping constant.

    Returns:
        A frame of ``item_id``, one ``retrieval_score_<source>`` column per
        source, ``retrieval_rank_fused`` and ``retrieval_n_sources``.
    """
    fused: dict[int, float] = {}
    source_scores: dict[str, dict[int, float]] = {}
    appearances: dict[int, int] = {}

    for source, scores in per_source.items():
        source_scores[source] = scores
        ranked = sorted(scores.items(), key=lambda pair: -pair[1])
        for rank, (item_id, _) in enumerate(ranked, start=1):
            fused[item_id] = fused.get(item_id, 0.0) + 1.0 / (k_rrf + rank)
            appearances[item_id] = appearances.get(item_id, 0) + 1

    if not fused:
        return pd.DataFrame(columns=["item_id", *RETRIEVAL_FEATURES])

    items = sorted(fused, key=lambda item: -fused[item])
    frame = pd.DataFrame({"item_id": items})
    for source in ("two_tower", "item_cf", "popularity"):
        scores = source_scores.get(source, {})
        frame[f"retrieval_score_{source}"] = [scores.get(item, np.nan) for item in items]
    frame["retrieval_rank_fused"] = np.arange(1, len(items) + 1, dtype=np.float32)
    frame["retrieval_n_sources"] = [appearances[item] for item in items]
    return frame


__all__ = [
    "RANKING_FEATURES",
    "RETRIEVAL_FEATURES",
    "RankingDataset",
    "build_ranking_dataset",
    "fuse_candidate_scores",
]
