"""Ranking metrics for top-K recommendation.

Every number this project publishes passes through here, so these are
implemented from their definitions and verified against hand-computed cases in
``tests/ml/test_metrics.py`` rather than trusted because they look plausible.
A quietly wrong NDCG is worse than no NDCG: the models get tuned against it,
the comparison table is built on it, and nothing ever looks broken.

Conventions used throughout, stated once:

- **Binary relevance.** An item is relevant if the user interacted with it in
  the evaluation window. Retailrocket has no graded judgments, so graded-gain
  NDCG would be inventing a scale the data does not contain.
- **Ideal DCG uses ``min(K, |relevant|)``.** A user with 2 relevant items
  cannot achieve more than 2 hits in a top-10 list. Normalising against a
  full-length ideal would cap that user's NDCG at 0.2 and make the metric a
  measure of how many items a user happened to interact with.
- **Recall@K is capped at 1.0**: repeated items are de-duplicated before
  scoring, so a candidate-fusion bug cannot push a ratio above its own
  maximum.
- **Users with no relevant items are skipped**, not scored 0 - including them
  would measure the composition of the evaluation set rather than the model.
  Users the model declined to serve ARE scored 0, so poor coverage costs.
- **Rank positions are 1-based** in all discounting, matching the literature.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Final

import numpy as np

from mercury_rec.core.logging import get_logger

logger = get_logger(__name__)

#: K values reported across the project.
DEFAULT_K_VALUES: Final[tuple[int, ...]] = (5, 10, 20)


def _validate_k(k: int) -> None:
    if k < 1:
        raise ValueError(f"K must be >= 1, got {k}")


def _dedupe(recommended: Sequence[int]) -> list[int]:
    """Drop repeated items, preserving order of first appearance.

    A well-behaved recommender never emits the same item twice, but candidate
    fusion merges several sources and a de-duplication bug there must not be
    able to inflate a metric. Without this, ``[A, A, A]`` against relevant
    ``{A}`` scores recall 3.0 - a counter that exceeds its own maximum while
    looking like an unusually good model.

    A repeated slot is treated as a wasted slot rather than removed from the
    denominator: ``[A, A, A]`` scores precision@3 = 1/3, because the model did
    occupy three slots and two of them carried nothing new.
    """
    seen: set[int] = set()
    unique: list[int] = []
    for item in recommended:
        if item not in seen:
            seen.add(item)
            unique.append(item)
    return unique


def precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of the top-K that is relevant.

    Divided by ``k`` rather than by ``len(recommended)``: a system that
    returns 3 items when 10 were requested should not be rewarded for its
    short list. Precision@10 of a 3-item list is at most 0.3.
    """
    _validate_k(k)
    if not relevant:
        return 0.0
    hits = sum(1 for item in _dedupe(recommended[:k]) if item in relevant)
    return hits / k


def recall_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Share of relevant items retrieved within the top-K."""
    _validate_k(k)
    if not relevant:
        return 0.0
    hits = sum(1 for item in _dedupe(recommended[:k]) if item in relevant)
    return hits / len(relevant)


def hit_rate_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """1.0 if any relevant item appears in the top-K, else 0.0."""
    _validate_k(k)
    if not relevant:
        return 0.0
    return 1.0 if any(item in relevant for item in recommended[:k]) else 0.0


def average_precision_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Average precision: mean of precision values at each relevant hit.

    Normalised by ``min(k, |relevant|)`` for the reason given in the module
    docstring - a user with 2 relevant items cannot fill 10 slots, and
    dividing by ``|relevant|`` alone would also misreport when a user has more
    relevant items than K.
    """
    _validate_k(k)
    if not relevant:
        return 0.0

    hits = 0
    precision_sum = 0.0
    for position, item in enumerate(_dedupe(recommended[:k]), start=1):
        if item in relevant:
            hits += 1
            precision_sum += hits / position

    denominator = min(k, len(relevant))
    return precision_sum / denominator if denominator else 0.0


def reciprocal_rank(recommended: Sequence[int], relevant: set[int], k: int | None = None) -> float:
    """Reciprocal of the first relevant item's 1-based rank, else 0."""
    candidates = recommended[:k] if k is not None else recommended
    for position, item in enumerate(_dedupe(candidates), start=1):
        if item in relevant:
            return 1.0 / position
    return 0.0


def dcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """Discounted cumulative gain with binary relevance.

    Gain is 1 for a relevant item; the discount is ``1 / log2(rank + 1)``, so
    rank 1 is undiscounted (log2(2) = 1).
    """
    _validate_k(k)
    total = 0.0
    for position, item in enumerate(_dedupe(recommended[:k]), start=1):
        if item in relevant:
            total += 1.0 / np.log2(position + 1)
    return total


def ndcg_at_k(recommended: Sequence[int], relevant: set[int], k: int) -> float:
    """DCG normalised by the best achievable DCG for this user.

    The ideal list places ``min(k, |relevant|)`` relevant items at the top.
    Using the user's own achievable ideal is what keeps NDCG comparable
    between a user with 2 relevant items and one with 50.
    """
    _validate_k(k)
    if not relevant:
        return 0.0

    ideal_hits = min(k, len(relevant))
    ideal = sum(1.0 / np.log2(position + 1) for position in range(1, ideal_hits + 1))
    if ideal == 0:
        return 0.0
    return float(dcg_at_k(recommended, relevant, k) / ideal)


# ---------------------------------------------------------------------------
# Aggregate evaluation
# ---------------------------------------------------------------------------


@dataclass(slots=True)
class EvaluationResult:
    """Metrics for one model, plus the population they were measured over."""

    model: str
    n_users_evaluated: int
    n_users_skipped: int
    metrics: dict[str, float] = field(default_factory=dict)
    catalog_coverage: float = 0.0
    gini: float = 0.0
    mean_popularity_rank: float = 0.0
    intra_list_diversity: float = 0.0
    n_distinct_items: int = 0

    def as_dict(self) -> dict[str, float | int | str]:
        return {
            "model": self.model,
            "n_users_evaluated": self.n_users_evaluated,
            "n_users_skipped": self.n_users_skipped,
            **{name: round(value, 6) for name, value in self.metrics.items()},
            "catalog_coverage": round(self.catalog_coverage, 6),
            "gini": round(self.gini, 6),
            "mean_popularity_rank": round(self.mean_popularity_rank, 6),
            "intra_list_diversity": round(self.intra_list_diversity, 6),
            "n_distinct_items": self.n_distinct_items,
        }


def gini_coefficient(counts: np.ndarray) -> float:
    """Gini over recommendation frequency: 0 = uniform, 1 = winner-takes-all.

    The headline popularity-bias number. Accuracy metrics are blind to a model
    that recommends the same 50 blockbusters to everyone; this is not.
    """
    if counts.size == 0:
        return 0.0
    values = np.sort(counts.astype(np.float64))
    total = values.sum()
    if total <= 0:
        return 0.0
    n = values.size
    index = np.arange(1, n + 1)
    return float((2.0 * (index * values).sum()) / (n * total) - (n + 1.0) / n)


def evaluate_recommendations(
    recommendations: Mapping[int, Sequence[int]],
    ground_truth: Mapping[int, set[int]],
    *,
    model: str,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    n_catalog_items: int | None = None,
    item_popularity: np.ndarray | None = None,
    item_categories: np.ndarray | None = None,
) -> EvaluationResult:
    """Aggregate ranking and beyond-accuracy metrics across users.

    Args:
        recommendations: user id -> ranked item ids, best first.
        ground_truth: user id -> set of items the user actually interacted
            with in the evaluation window.
        model: Label recorded in the result.
        k_values: K values to report.
        n_catalog_items: Catalogue size, for coverage.
        item_popularity: Per-item interaction counts from TRAINING (never the
            evaluation window - that would be leakage into a reported metric).
        item_categories: Per-item category, for intra-list diversity.

    Returns:
        Metrics plus the evaluated/skipped user counts, so the population is
        always visible alongside the numbers.
    """
    for k in k_values:
        _validate_k(k)

    per_user: dict[str, list[float]] = {}
    recommended_counts: dict[int, int] = {}
    diversity_scores: list[float] = []
    popularity_ranks: list[float] = []

    popularity_order: np.ndarray | None = None
    if item_popularity is not None and item_popularity.size:
        # Rank 0 = most popular; normalised to [0, 1] so it is comparable
        # across catalogues of different sizes.
        order = np.argsort(np.argsort(-item_popularity))
        popularity_order = order / max(len(order) - 1, 1)

    evaluated = 0
    skipped = 0

    for user_id, relevant in ground_truth.items():
        # A user with no relevant items in the window cannot be scored.
        # Including them as zeros would measure the composition of the
        # evaluation set rather than the model.
        if not relevant:
            skipped += 1
            continue

        # A user the model declined to serve IS a failure and is scored 0;
        # excluding them would flatter a model with poor coverage.
        ranked = list(recommendations.get(user_id, ()))
        served = bool(ranked)

        evaluated += 1

        for k in k_values:
            per_user.setdefault(f"precision@{k}", []).append(precision_at_k(ranked, relevant, k))
            per_user.setdefault(f"recall@{k}", []).append(recall_at_k(ranked, relevant, k))
            per_user.setdefault(f"ndcg@{k}", []).append(ndcg_at_k(ranked, relevant, k))
            per_user.setdefault(f"map@{k}", []).append(average_precision_at_k(ranked, relevant, k))
            per_user.setdefault(f"hit_rate@{k}", []).append(hit_rate_at_k(ranked, relevant, k))
        per_user.setdefault("mrr", []).append(reciprocal_rank(ranked, relevant))

        if not served:
            continue

        largest_k = max(k_values)
        top = ranked[:largest_k]
        for item in top:
            recommended_counts[item] = recommended_counts.get(item, 0) + 1

        if popularity_order is not None:
            valid = [int(i) for i in top if 0 <= int(i) < len(popularity_order)]
            if valid:
                popularity_ranks.append(float(np.mean(popularity_order[valid])))

        if item_categories is not None and len(top) > 1:
            valid_items = [int(i) for i in top if 0 <= int(i) < len(item_categories)]
            if len(valid_items) > 1:
                categories = item_categories[valid_items]
                # Share of distinct categories: 1.0 means every slot is a
                # different category.
                diversity_scores.append(len(np.unique(categories)) / len(categories))

    metrics = {name: float(np.mean(values)) for name, values in per_user.items()}

    counts = np.array(list(recommended_counts.values()), dtype=np.float64)
    coverage = len(recommended_counts) / n_catalog_items if n_catalog_items else 0.0

    result = EvaluationResult(
        model=model,
        n_users_evaluated=evaluated,
        n_users_skipped=skipped,
        metrics=metrics,
        catalog_coverage=coverage,
        gini=gini_coefficient(counts),
        mean_popularity_rank=float(np.mean(popularity_ranks)) if popularity_ranks else 0.0,
        intra_list_diversity=float(np.mean(diversity_scores)) if diversity_scores else 0.0,
        n_distinct_items=len(recommended_counts),
    )
    logger.info(
        "evaluation.complete",
        model=model,
        users=evaluated,
        skipped=skipped,
        **{name: round(value, 4) for name, value in metrics.items()},
    )
    return result


__all__ = [
    "DEFAULT_K_VALUES",
    "EvaluationResult",
    "average_precision_at_k",
    "dcg_at_k",
    "evaluate_recommendations",
    "gini_coefficient",
    "hit_rate_at_k",
    "ndcg_at_k",
    "precision_at_k",
    "recall_at_k",
    "reciprocal_rank",
]
