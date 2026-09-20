"""Verify ranking metrics against hand-computed values.

Every figure this project publishes flows through these functions. A quietly
wrong NDCG would not raise anything - the models would simply be tuned
against it and the whole comparison table would rest on it.

So the expected values below were worked out by hand from the definitions,
with the arithmetic shown, rather than captured from the implementation. A
test that records whatever the code currently returns proves only that the
code is deterministic.
"""

from __future__ import annotations

import math

import numpy as np
import pytest

from mercury_rec.evaluation.metrics import (
    average_precision_at_k,
    dcg_at_k,
    evaluate_recommendations,
    gini_coefficient,
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
    reciprocal_rank,
)

# Worked example used throughout.
#   recommended: [10, 11, 12, 13, 14]   (ranks 1..5)
#   relevant:    {10, 12}               (hits at ranks 1 and 3)
RECOMMENDED = [10, 11, 12, 13, 14]
RELEVANT = {10, 12}


class TestPrecision:
    def test_hand_computed(self) -> None:
        # 2 hits in 5 slots
        assert precision_at_k(RECOMMENDED, RELEVANT, 5) == pytest.approx(2 / 5)
        # rank 1 is a hit, rank 2 is not
        assert precision_at_k(RECOMMENDED, RELEVANT, 1) == pytest.approx(1.0)
        assert precision_at_k(RECOMMENDED, RELEVANT, 2) == pytest.approx(1 / 2)
        assert precision_at_k(RECOMMENDED, RELEVANT, 3) == pytest.approx(2 / 3)

    def test_divides_by_k_not_list_length(self) -> None:
        """A short list must not be rewarded for returning fewer items.

        Precision@10 over a 2-item list is 2/10, not 2/2 - otherwise a model
        that returns one lucky item scores a perfect 1.0.
        """
        assert precision_at_k([10, 11], {10, 11}, 10) == pytest.approx(0.2)

    def test_no_relevant_items_scores_zero(self) -> None:
        assert precision_at_k(RECOMMENDED, set(), 5) == 0.0


class TestRecall:
    def test_hand_computed(self) -> None:
        assert recall_at_k(RECOMMENDED, RELEVANT, 5) == pytest.approx(1.0)  # 2 of 2
        assert recall_at_k(RECOMMENDED, RELEVANT, 1) == pytest.approx(0.5)  # 1 of 2
        assert recall_at_k(RECOMMENDED, RELEVANT, 3) == pytest.approx(1.0)

    def test_never_exceeds_one(self) -> None:
        assert recall_at_k([10, 10, 10], {10}, 3) <= 1.0


class TestHitRate:
    def test_hand_computed(self) -> None:
        assert hit_rate_at_k(RECOMMENDED, RELEVANT, 1) == 1.0
        assert hit_rate_at_k(RECOMMENDED, {12}, 2) == 0.0  # 12 sits at rank 3
        assert hit_rate_at_k(RECOMMENDED, {12}, 3) == 1.0

    def test_is_binary(self) -> None:
        assert hit_rate_at_k(RECOMMENDED, RELEVANT, 5) in (0.0, 1.0)


class TestDCG:
    def test_hand_computed(self) -> None:
        # Hits at ranks 1 and 3:
        #   1/log2(1+1) = 1/1       = 1.0
        #   1/log2(3+1) = 1/2       = 0.5
        #   total                   = 1.5
        assert dcg_at_k(RECOMMENDED, RELEVANT, 5) == pytest.approx(1.5)

    def test_rank_one_is_undiscounted(self) -> None:
        assert dcg_at_k([10], {10}, 1) == pytest.approx(1.0)

    def test_discount_is_monotonic_in_rank(self) -> None:
        """The same hit must be worth strictly less further down the list."""
        assert dcg_at_k([10, 99], {10}, 2) > dcg_at_k([99, 10], {10}, 2)


class TestNDCG:
    def test_hand_computed(self) -> None:
        # DCG  = 1.5 (above)
        # IDCG = min(5, 2) = 2 hits at ranks 1, 2
        #      = 1/log2(2) + 1/log2(3) = 1.0 + 0.630930 = 1.630930
        # NDCG = 1.5 / 1.630930 = 0.919721
        ideal = 1.0 + 1.0 / math.log2(3)
        assert ndcg_at_k(RECOMMENDED, RELEVANT, 5) == pytest.approx(1.5 / ideal)
        assert ndcg_at_k(RECOMMENDED, RELEVANT, 5) == pytest.approx(0.919721, abs=1e-6)

    def test_perfect_ranking_scores_one(self) -> None:
        """All relevant items at the top must give exactly 1.0."""
        assert ndcg_at_k([10, 12, 11, 13], {10, 12}, 4) == pytest.approx(1.0)

    def test_ideal_uses_achievable_hits_not_k(self) -> None:
        """A user with one relevant item can still score 1.0.

        Normalising against a full-length ideal would cap that user at
        1/IDCG(k), turning NDCG into a measure of how many items the user
        happened to interact with rather than of ranking quality.
        """
        assert ndcg_at_k([10, 11, 12], {10}, 3) == pytest.approx(1.0)

    def test_no_hits_scores_zero(self) -> None:
        assert ndcg_at_k([1, 2, 3], {99}, 3) == 0.0

    def test_bounded_in_unit_interval(self) -> None:
        rng = np.random.default_rng(0)
        for _ in range(50):
            ranked = rng.permutation(20).tolist()
            relevant = set(rng.choice(20, size=rng.integers(1, 8), replace=False).tolist())
            assert 0.0 <= ndcg_at_k(ranked, relevant, 10) <= 1.0


class TestAveragePrecision:
    def test_hand_computed(self) -> None:
        # Hit at rank 1 -> precision 1/1 = 1.000000
        # Hit at rank 3 -> precision 2/3 = 0.666667
        # sum = 1.666667, denominator = min(5, 2) = 2
        # AP  = 0.833333
        assert average_precision_at_k(RECOMMENDED, RELEVANT, 5) == pytest.approx((1.0 + 2 / 3) / 2)
        assert average_precision_at_k(RECOMMENDED, RELEVANT, 5) == pytest.approx(0.833333, abs=1e-6)

    def test_perfect_ranking_scores_one(self) -> None:
        assert average_precision_at_k([10, 12, 99], {10, 12}, 3) == pytest.approx(1.0)

    def test_rewards_earlier_hits(self) -> None:
        early = average_precision_at_k([10, 98, 99], {10}, 3)
        late = average_precision_at_k([98, 99, 10], {10}, 3)
        assert early > late


class TestReciprocalRank:
    def test_hand_computed(self) -> None:
        assert reciprocal_rank([99, 98, 10], {10}) == pytest.approx(1 / 3)
        assert reciprocal_rank([10, 98, 99], {10}) == pytest.approx(1.0)

    def test_uses_only_the_first_hit(self) -> None:
        assert reciprocal_rank([99, 10, 12], {10, 12}) == pytest.approx(1 / 2)

    def test_no_hit_scores_zero(self) -> None:
        assert reciprocal_rank([1, 2, 3], {99}) == 0.0


class TestGini:
    def test_uniform_distribution_is_zero(self) -> None:
        assert gini_coefficient(np.array([5, 5, 5, 5])) == pytest.approx(0.0, abs=1e-9)

    def test_concentration_approaches_one(self) -> None:
        """Recommending one item to everyone is maximum popularity bias."""
        counts = np.zeros(1000)
        counts[0] = 1000
        assert gini_coefficient(counts) > 0.99

    def test_more_concentrated_scores_higher(self) -> None:
        spread = gini_coefficient(np.array([10, 10, 10, 10]))
        skewed = gini_coefficient(np.array([37, 1, 1, 1]))
        assert skewed > spread

    def test_empty_input_is_zero(self) -> None:
        assert gini_coefficient(np.array([])) == 0.0


class TestAggregateEvaluation:
    def test_averages_across_users(self) -> None:
        """Two users, one perfect and one miss, must average to 0.5 hit rate."""
        result = evaluate_recommendations(
            recommendations={1: [10, 11, 12], 2: [20, 21, 22]},
            ground_truth={1: {10}, 2: {99}},
            model="test",
            k_values=(3,),
            n_catalog_items=100,
        )
        assert result.metrics["hit_rate@3"] == pytest.approx(0.5)
        assert result.metrics["precision@3"] == pytest.approx((1 / 3 + 0.0) / 2)
        assert result.n_users_evaluated == 2

    def test_users_without_relevant_items_are_skipped_not_zeroed(self) -> None:
        """Scoring them zero would measure the evaluation set, not the model."""
        result = evaluate_recommendations(
            recommendations={1: [10], 2: [20]},
            ground_truth={1: {10}, 2: set()},
            model="test",
            k_values=(1,),
        )
        assert result.n_users_evaluated == 1
        assert result.n_users_skipped == 1
        assert result.metrics["hit_rate@1"] == pytest.approx(1.0)

    def test_users_the_model_declined_to_serve_score_zero(self) -> None:
        """Poor coverage must cost the model, not be quietly excluded."""
        result = evaluate_recommendations(
            recommendations={1: [10]},  # user 2 gets nothing
            ground_truth={1: {10}, 2: {20}},
            model="test",
            k_values=(1,),
        )
        assert result.n_users_evaluated == 2
        assert result.metrics["hit_rate@1"] == pytest.approx(0.5)

    def test_catalog_coverage(self) -> None:
        result = evaluate_recommendations(
            recommendations={1: [1, 2], 2: [2, 3]},
            ground_truth={1: {1}, 2: {3}},
            model="test",
            k_values=(2,),
            n_catalog_items=10,
        )
        # Distinct items recommended: {1, 2, 3} out of 10
        assert result.catalog_coverage == pytest.approx(0.3)
        assert result.n_distinct_items == 3

    def test_intra_list_diversity(self) -> None:
        """Three items from three categories is maximal diversity."""
        categories = np.array([0, 1, 2, 0, 0])
        result = evaluate_recommendations(
            recommendations={1: [0, 1, 2]},
            ground_truth={1: {0}},
            model="test",
            k_values=(3,),
            item_categories=categories,
        )
        assert result.intra_list_diversity == pytest.approx(1.0)

        same_category = evaluate_recommendations(
            recommendations={1: [0, 3, 4]},  # all category 0
            ground_truth={1: {0}},
            model="test",
            k_values=(3,),
            item_categories=categories,
        )
        assert same_category.intra_list_diversity == pytest.approx(1 / 3)

    def test_result_serialises_with_the_population(self) -> None:
        """Metrics must never travel without the population they describe."""
        result = evaluate_recommendations(
            recommendations={1: [10]},
            ground_truth={1: {10}},
            model="popularity",
            k_values=(1,),
            n_catalog_items=5,
        )
        payload = result.as_dict()
        assert payload["model"] == "popularity"
        assert payload["n_users_evaluated"] == 1
        assert "hit_rate@1" in payload


@pytest.mark.parametrize("k", [0, -1])
def test_invalid_k_is_rejected(k: int) -> None:
    with pytest.raises(ValueError, match="K must be"):
        precision_at_k([1], {1}, k)
