"""Tests for the business re-ranking layer.

The property these exist to protect is the separation between the model's
relevance score and commercial policy. Once those are merged you can no longer
answer "why did this item appear here?", which is the question the layer is
built to keep answerable.
"""

from __future__ import annotations

import numpy as np
import pytest

from mercury_rec.reranking.pipeline import (
    RerankConfig,
    ScoredItem,
    apply_business_adjustments,
    apply_hard_filters,
    enforce_diversity,
    rerank,
)


def _item(item_id: int, score: float, **kwargs: object) -> ScoredItem:
    return ScoredItem(item_id=item_id, ml_relevance_score=score, **kwargs)  # type: ignore[arg-type]


class TestScoreSeparation:
    def test_ml_score_is_never_modified(self) -> None:
        """The whole point of the layer.

        Once a boost is folded into the relevance score you cannot tell
        whether a ranking change came from the model or from policy.
        """
        item = _item(1, 0.8, is_sponsored=True)
        original = item.ml_relevance_score

        apply_business_adjustments([item], RerankConfig(sponsored_boost=0.5))

        assert item.ml_relevance_score == original
        assert item.business_adjustment == pytest.approx(0.5)
        assert item.final_score == pytest.approx(1.3)

    def test_every_adjustment_names_its_rule(self) -> None:
        """An unattributable adjustment is indistinguishable from a bug."""
        item = _item(1, 0.5, is_sponsored=True)
        config = RerankConfig(sponsored_boost=0.3, popularity_penalty=0.2)
        popularity = np.array([0.0, 0.0])  # item 1 is maximally popular

        apply_business_adjustments([item], config, popularity_rank=popularity)

        assert "sponsored_boost" in item.adjustments
        assert "popularity_penalty" in item.adjustments
        assert item.business_adjustment == pytest.approx(sum(item.adjustments.values()))

    def test_zero_adjustments_are_not_recorded(self) -> None:
        """Noise in the attribution trail makes real adjustments harder to see."""
        item = _item(1, 0.5, is_sponsored=True)
        apply_business_adjustments([item], RerankConfig(sponsored_boost=0.0))
        assert item.adjustments == {}

    def test_serialised_output_exposes_both_scores(self) -> None:
        item = _item(1, 0.8, is_sponsored=True)
        apply_business_adjustments([item], RerankConfig(sponsored_boost=0.2))
        payload = item.as_dict(rank=1)

        assert payload["ml_relevance_score"] == pytest.approx(0.8)
        assert payload["business_adjustment"] == pytest.approx(0.2)
        assert payload["final_score"] == pytest.approx(1.0)

    def test_defaults_apply_no_commercial_adjustment(self) -> None:
        """Boosts default OFF, so any adjustment is a deliberate choice."""
        items = [_item(i, 1.0 - i * 0.1, is_sponsored=True) for i in range(5)]
        apply_business_adjustments(items, RerankConfig())
        assert all(item.business_adjustment == 0.0 for item in items)


class TestHardFilters:
    def test_unavailable_items_are_removed(self) -> None:
        items = [_item(1, 0.9, is_available=False), _item(2, 0.5, is_available=True)]
        kept, removed = apply_hard_filters(items, RerankConfig())

        assert [item.item_id for item in kept] == [2]
        assert removed["unavailable"] == 1

    def test_availability_enforcement_can_be_disabled(self) -> None:
        items = [_item(1, 0.9, is_available=False)]
        kept, _ = apply_hard_filters(items, RerankConfig(enforce_availability=False))
        assert len(kept) == 1

    def test_delivery_constraint_is_applied(self) -> None:
        items = [_item(1, 0.9, delivery_minutes=90), _item(2, 0.5, delivery_minutes=20)]
        kept, removed = apply_hard_filters(items, RerankConfig(max_delivery_minutes=45))

        assert [item.item_id for item in kept] == [2]
        assert removed["delivery_too_slow"] == 1

    def test_filters_beat_relevance(self) -> None:
        """A hard filter must not be overridable by a high score."""
        items = [_item(1, 99.0, is_available=False)]
        kept, _ = apply_hard_filters(items, RerankConfig())
        assert kept == []


class TestDiversity:
    def test_merchant_cap_is_enforced(self) -> None:
        items = [_item(i, 1.0 - i * 0.01, merchant_id=7) for i in range(10)]
        items.extend(_item(100 + i, 0.5 - i * 0.01, merchant_id=i) for i in range(10))

        selected = enforce_diversity(items, RerankConfig(max_per_merchant=2), k=10)

        from collections import Counter

        counts = Counter(item.merchant_id for item in selected)
        assert counts[7] <= 2

    def test_category_cap_is_enforced(self) -> None:
        items = [_item(i, 1.0 - i * 0.01, category_id=3) for i in range(10)]
        items.extend(_item(100 + i, 0.5 - i * 0.01, category_id=i) for i in range(10))

        selected = enforce_diversity(items, RerankConfig(max_per_category=3), k=10)

        from collections import Counter

        counts = Counter(item.category_id for item in selected)
        assert counts[3] <= 3

    def test_the_best_item_is_always_shown_first(self) -> None:
        """A property users notice immediately if it breaks."""
        items = [_item(i, 1.0 - i * 0.1, merchant_id=0) for i in range(5)]
        selected = enforce_diversity(items, RerankConfig(max_per_merchant=1), k=3)
        assert selected[0].item_id == 0

    def test_caps_relax_rather_than_returning_a_short_list(self) -> None:
        """A short list is a worse failure than briefly exceeding a cap."""
        items = [_item(i, 1.0 - i * 0.01, merchant_id=0, category_id=0) for i in range(20)]
        selected = enforce_diversity(items, RerankConfig(max_per_merchant=2), k=10)
        assert len(selected) == 10

    def test_output_is_ordered_by_final_score(self) -> None:
        items = [_item(i, float(i), merchant_id=i) for i in range(10)]
        selected = enforce_diversity(items, RerankConfig(), k=5)
        scores = [item.final_score for item in selected]
        assert scores == sorted(scores, reverse=True)


class TestFullPipeline:
    def test_stage_order_filters_before_boosting(self) -> None:
        """A boosted but unavailable item must never appear."""
        items = [
            _item(1, 0.9, is_available=False, is_sponsored=True, merchant_id=1),
            _item(2, 0.5, is_available=True, merchant_id=2),
        ]
        result = rerank(items, k=5, config=RerankConfig(sponsored_boost=10.0))

        assert [item.item_id for item in result.items] == [2]
        assert result.filtered_counts["unavailable"] == 1

    def test_result_reports_what_was_filtered(self) -> None:
        """Silent filtering degrades recommendations with no visible cause."""
        items = [_item(i, 0.5, is_available=i % 2 == 0, merchant_id=i) for i in range(10)]
        result = rerank(items, k=10)

        assert result.n_input == 10
        assert result.filtered_counts["unavailable"] == 5
        assert result.summary()["n_output"] == 5

    def test_returns_at_most_k(self) -> None:
        items = [_item(i, float(i), merchant_id=i) for i in range(50)]
        assert len(rerank(items, k=10).items) == 10

    def test_empty_input_is_handled(self) -> None:
        result = rerank([], k=10)
        assert result.items == []
        assert result.n_input == 0

    def test_popularity_penalty_demotes_head_items(self) -> None:
        """Damping popularity bias must actually change the order."""
        # Item 0 is the most popular (rank 0.0), item 1 the least (rank 1.0).
        popularity = np.array([0.0, 1.0])
        items = [_item(0, 1.00, merchant_id=0), _item(1, 0.95, merchant_id=1)]

        result = rerank(
            items,
            k=2,
            config=RerankConfig(popularity_penalty=0.5),
            popularity_rank=popularity,
        )
        assert [item.item_id for item in result.items] == [1, 0], (
            "the popular item should have been demoted below the long-tail item"
        )
        # And the model's own scores are untouched by that demotion.
        assert result.items[1].ml_relevance_score == pytest.approx(1.00)
