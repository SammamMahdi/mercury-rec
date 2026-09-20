"""Business re-ranking: deterministic policy applied after ML scoring.

The single most important property of this layer is that **the ML relevance
score and the business adjustment are never merged into one number**. Each
recommendation carries both, plus the final score that combines them, and
every adjustment carries the rule that produced it.

That separation is not bookkeeping. Once a sponsorship boost is folded into
the relevance score, three things become impossible: you cannot tell whether a
ranking change came from the model or from policy, you cannot evaluate the
model's quality independently of commercial rules, and you cannot answer a
regulator, a merchant or a colleague asking why a particular item appeared
where it did. Systems that blend the two are why "the algorithm promoted this"
is so often unanswerable.

The stages run in a fixed order, and the order is load-bearing:

1. **Hard filters** -- availability, delivery feasibility. These remove items
   that must not be shown at all, so nothing downstream can resurface them.
2. **Business adjustments** -- additive, recorded per rule.
3. **Diversity** -- greedy MMR-style selection with per-merchant and
   per-category caps, applied last because it selects from what survives.

Filters before boosts, because a boost applied to an unavailable item is
wasted work; diversity last, because it is a property of the final list rather
than of any individual item.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mercury_rec.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class RerankConfig:
    """Business policy. Every value is configuration, never a code constant."""

    max_per_merchant: int = 2
    """Cap on items from one merchant in the final list. Two lets a strong
    merchant appear twice without a single one owning the page."""

    max_per_category: int = 3

    enforce_availability: bool = True
    max_delivery_minutes: int | None = None

    diversity_weight: float = 0.0
    """MMR trade-off. 0 keeps pure relevance order subject to the caps.
    Non-zero costs relevance, so the default is off and the cost is measured
    in the A/B framework rather than assumed to be free."""

    sponsored_boost: float = 0.0
    """Additive boost for sponsored items. Defaults to OFF. When enabled the
    contribution is recorded separately on every affected item, so a sponsored
    placement is always attributable."""

    popularity_penalty: float = 0.0
    """Subtracts ``weight * normalised_popularity_rank``, damping head-item
    concentration. Also off by default, and also measured."""


@dataclass(slots=True)
class ScoredItem:
    """One candidate, with relevance and policy kept strictly apart."""

    item_id: int
    ml_relevance_score: float
    """The model's output. Never modified by this layer."""

    business_adjustment: float = 0.0
    """Sum of policy adjustments. Always reported alongside, never folded in."""

    adjustments: dict[str, float] = field(default_factory=dict)
    """Per-rule contributions, so any adjustment is attributable."""

    merchant_id: int | None = None
    category_id: int | None = None
    vertical: int | None = None
    is_available: bool = True
    delivery_minutes: int | None = None
    is_sponsored: bool = False

    @property
    def final_score(self) -> float:
        return self.ml_relevance_score + self.business_adjustment

    def apply(self, rule: str, delta: float) -> None:
        """Record an adjustment under the rule that produced it."""
        if delta == 0.0:
            return
        self.adjustments[rule] = self.adjustments.get(rule, 0.0) + delta
        self.business_adjustment += delta

    def as_dict(self, rank: int) -> dict[str, Any]:
        return {
            "item_id": self.item_id,
            "rank": rank,
            "ml_relevance_score": round(self.ml_relevance_score, 6),
            "business_adjustment": round(self.business_adjustment, 6),
            "final_score": round(self.final_score, 6),
            "adjustments": {k: round(v, 6) for k, v in self.adjustments.items()},
        }


@dataclass(slots=True)
class RerankResult:
    items: list[ScoredItem]
    filtered_counts: dict[str, int]
    n_input: int

    def summary(self) -> dict[str, Any]:
        return {
            "n_input": self.n_input,
            "n_output": len(self.items),
            "filtered": self.filtered_counts,
            "n_adjusted": sum(1 for item in self.items if item.business_adjustment != 0.0),
        }


def apply_hard_filters(
    items: list[ScoredItem], config: RerankConfig
) -> tuple[list[ScoredItem], dict[str, int]]:
    """Remove items that must not be shown, counting each reason.

    Counts are returned rather than only logged: if filtering silently removes
    most of the candidate pool the recommendations degrade with no visible
    cause, so the numbers surface in the API response and on the monitoring
    dashboard.
    """
    kept: list[ScoredItem] = []
    removed = {"unavailable": 0, "delivery_too_slow": 0}

    for item in items:
        if config.enforce_availability and not item.is_available:
            removed["unavailable"] += 1
            continue
        if (
            config.max_delivery_minutes is not None
            and item.delivery_minutes is not None
            and item.delivery_minutes > config.max_delivery_minutes
        ):
            removed["delivery_too_slow"] += 1
            continue
        kept.append(item)

    return kept, removed


def apply_business_adjustments(
    items: list[ScoredItem],
    config: RerankConfig,
    *,
    popularity_rank: np.ndarray | None = None,
) -> None:
    """Apply additive policy adjustments in place, recorded per rule."""
    for item in items:
        if config.sponsored_boost and item.is_sponsored:
            item.apply("sponsored_boost", config.sponsored_boost)

        if (
            config.popularity_penalty
            and popularity_rank is not None
            and 0 <= item.item_id < len(popularity_rank)
        ):
            # popularity_rank is normalised to [0, 1], 0 = most popular, so
            # the penalty is largest for head items.
            penalty = -config.popularity_penalty * (1.0 - float(popularity_rank[item.item_id]))
            item.apply("popularity_penalty", penalty)


def enforce_diversity(items: list[ScoredItem], config: RerankConfig, k: int) -> list[ScoredItem]:
    """Greedily select ``k`` items subject to merchant and category caps.

    Greedy rather than a global optimum: the exact formulation is a
    constrained subset-selection problem, and greedy selection by score is
    both the standard approach and the only one that fits a few milliseconds
    of request budget. It also has a property users notice -- the highest
    scoring item is always shown first.

    Items skipped by a cap are not discarded; if the caps cannot fill ``k``
    slots, the best skipped items are appended in score order. Returning a
    short list would be a worse failure than briefly exceeding a diversity
    target.
    """
    ordered = sorted(items, key=lambda item: -item.final_score)

    selected: list[ScoredItem] = []
    deferred: list[ScoredItem] = []
    per_merchant: dict[int, int] = {}
    per_category: dict[int, int] = {}

    for item in ordered:
        if len(selected) >= k:
            break

        merchant = item.merchant_id
        category = item.category_id
        if merchant is not None and per_merchant.get(merchant, 0) >= config.max_per_merchant:
            deferred.append(item)
            continue
        if category is not None and per_category.get(category, 0) >= config.max_per_category:
            deferred.append(item)
            continue

        selected.append(item)
        if merchant is not None:
            per_merchant[merchant] = per_merchant.get(merchant, 0) + 1
        if category is not None:
            per_category[category] = per_category.get(category, 0) + 1

    if len(selected) < k and deferred:
        shortfall = k - len(selected)
        logger.debug("rerank.caps_relaxed", shortfall=shortfall)
        selected.extend(deferred[:shortfall])

    return selected


def rerank(
    items: list[ScoredItem],
    *,
    k: int = 10,
    config: RerankConfig | None = None,
    popularity_rank: np.ndarray | None = None,
) -> RerankResult:
    """Run the full re-ranking pipeline.

    Args:
        items: Candidates carrying their ML relevance scores.
        k: Size of the final list.
        config: Business policy. Defaults are conservative: caps on, boosts
            off.
        popularity_rank: Normalised popularity rank per item id, for the
            popularity penalty.

    Returns:
        The final items plus what was filtered and why.
    """
    policy = config or RerankConfig()
    n_input = len(items)

    survivors, removed = apply_hard_filters(items, policy)
    apply_business_adjustments(survivors, policy, popularity_rank=popularity_rank)
    selected = enforce_diversity(survivors, policy, k)

    result = RerankResult(items=selected, filtered_counts=removed, n_input=n_input)
    logger.debug("rerank.complete", **result.summary())
    return result


__all__ = [
    "RerankConfig",
    "RerankResult",
    "ScoredItem",
    "apply_business_adjustments",
    "apply_hard_filters",
    "enforce_diversity",
    "rerank",
]
