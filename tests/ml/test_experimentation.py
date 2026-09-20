"""Tests for the A/B simulation and the promotion gate.

Two properties matter most and neither is obvious from reading the code:

- The bootstrap must resample **users**, not observations. Resampling rows
  produces intervals several times too narrow, which turns noise into a
  significant result - the most dangerous possible failure for a module whose
  job is to say whether a difference is real.
- The gate must never treat "could not evaluate" as a pass. A gate that waves
  through what it could not check provides the feeling of a check without the
  check.
"""

from __future__ import annotations

import numpy as np
import pytest

from mercury_rec.experimentation.ab_simulation import (
    paired_bootstrap,
    per_user_metrics,
    simulate_ab_test,
)
from mercury_rec.experimentation.validation_gate import (
    GateConfig,
    Verdict,
    evaluate_gate,
)


class TestPairedBootstrap:
    def test_identical_variants_give_an_interval_containing_zero(self) -> None:
        values = np.random.default_rng(0).random(200)
        lower, upper, p_value = paired_bootstrap(values, values.copy())

        assert lower <= 0 <= upper
        assert p_value > 0.05

    def test_a_large_consistent_shift_is_detected(self) -> None:
        rng = np.random.default_rng(1)
        control = rng.random(300)
        treatment = control + 0.2  # unambiguous, uniform improvement

        lower, upper, p_value = paired_bootstrap(control, treatment)
        assert lower > 0, "a consistent +0.2 shift should not include zero"
        assert p_value < 0.05

    def test_direction_is_preserved(self) -> None:
        rng = np.random.default_rng(2)
        control = rng.random(200) + 0.3
        treatment = rng.random(200)  # worse on average

        lower, upper, _ = paired_bootstrap(control, treatment)
        assert upper < 0, "a regression must produce a negative interval"

    def test_p_value_is_never_exactly_zero(self) -> None:
        """Finite resampling cannot justify certainty.

        The +1 correction keeps the reported p-value from claiming more than
        the number of resamples can support.
        """
        control = np.zeros(100)
        treatment = np.ones(100)
        _, _, p_value = paired_bootstrap(control, treatment, n_samples=500)
        assert p_value > 0

    def test_mismatched_shapes_are_rejected(self) -> None:
        """A paired test requires the same users in the same order."""
        with pytest.raises(ValueError, match="matching shapes"):
            paired_bootstrap(np.zeros(10), np.zeros(5))

    def test_empty_input_is_handled(self) -> None:
        lower, upper, p_value = paired_bootstrap(np.array([]), np.array([]))
        assert (lower, upper, p_value) == (0.0, 0.0, 1.0)

    def test_result_is_reproducible(self) -> None:
        """A published interval must be reproducible from the seed."""
        rng = np.random.default_rng(3)
        control, treatment = rng.random(150), rng.random(150)
        assert paired_bootstrap(control, treatment, seed=7) == paired_bootstrap(
            control, treatment, seed=7
        )

    def test_more_users_narrows_the_interval(self) -> None:
        """Basic sanity: precision must improve with sample size."""
        rng = np.random.default_rng(4)
        small_c, small_t = rng.random(30), rng.random(30)
        large_c, large_t = rng.random(3000), rng.random(3000)

        small_lo, small_hi, _ = paired_bootstrap(small_c, small_t)
        large_lo, large_hi, _ = paired_bootstrap(large_c, large_t)
        assert (large_hi - large_lo) < (small_hi - small_lo)


class TestSimulation:
    def test_perfect_treatment_beats_random_control(self) -> None:
        truth = {user: {user * 10} for user in range(50)}
        control = {user: [999, 998, 997] for user in range(50)}  # never relevant
        treatment = {user: [user * 10, 1, 2] for user in range(50)}  # always rank 1

        result = simulate_ab_test(control, treatment, truth, k=3, n_bootstrap=500)
        ndcg = next(c for c in result.comparisons if c.metric == "ndcg@3")

        assert ndcg.treatment > ndcg.control
        assert ndcg.is_significant

    def test_result_is_always_flagged_as_simulated(self) -> None:
        """This module cannot produce a live result, and says so."""
        truth = {1: {10}}
        result = simulate_ab_test({1: [10]}, {1: [10]}, truth, k=1, n_bootstrap=100)

        assert result.is_simulated is True
        assert "No live users" in result.as_dict()["disclaimer"]

    def test_only_shared_users_are_compared(self) -> None:
        """Comparing different populations is not a paired comparison."""
        truth = {1: {10}, 2: {20}, 3: {30}}
        control = {1: [10], 2: [20]}  # user 3 absent
        treatment = {1: [10], 2: [20], 3: [30]}

        result = simulate_ab_test(control, treatment, truth, k=1, n_bootstrap=100)
        assert result.n_users == 2

    def test_no_shared_users_returns_empty(self) -> None:
        result = simulate_ab_test({1: [1]}, {2: [2]}, {3: {3}}, k=1, n_bootstrap=100)
        assert result.n_users == 0

    def test_per_user_metrics_align_with_the_user_order(self) -> None:
        """The bootstrap indexes these positionally, so order is load-bearing."""
        truth = {1: {10}, 2: {20}}
        metrics = per_user_metrics({1: [10], 2: [99]}, truth, [1, 2], k=1)

        assert metrics["hit_rate@1"][0] == 1.0  # user 1 hit
        assert metrics["hit_rate@1"][1] == 0.0  # user 2 missed


class TestValidationGate:
    def _candidate(self, **overrides: float) -> dict[str, float]:
        base = {
            "ndcg@10": 0.0215,
            "catalog_coverage": 0.307,
            "gini": 0.615,
            "p99_latency_ms": 20.0,
        }
        base.update(overrides)
        return base

    def test_clear_improvement_is_promoted(self) -> None:
        result = evaluate_gate(
            self._candidate(),
            self._candidate(**{"ndcg@10": 0.0050, "catalog_coverage": 0.001}),
            candidate_version="bpr",
            incumbent_version="popularity",
        )
        assert result.promoted is True

    def test_quality_regression_is_rejected(self) -> None:
        result = evaluate_gate(
            self._candidate(**{"ndcg@10": 0.010}),
            self._candidate(**{"ndcg@10": 0.020}),
            candidate_version="worse",
        )
        assert result.promoted is False
        assert any("primary_metric" in c.name for c in result.failures)

    def test_a_model_can_fail_on_coverage_while_winning_on_accuracy(self) -> None:
        """The reason the gate is multi-dimensional.

        A model recommending the same 50 items to everyone can post excellent
        accuracy and be entirely unshippable.
        """
        result = evaluate_gate(
            self._candidate(**{"ndcg@10": 0.50, "catalog_coverage": 0.001}),
            self._candidate(),
            candidate_version="narrow",
        )
        assert result.promoted is False
        assert any(c.name == "coverage_floor" for c in result.failures)

    def test_latency_breach_is_rejected(self) -> None:
        result = evaluate_gate(
            self._candidate(p99_latency_ms=5000.0),
            self._candidate(),
            candidate_version="slow",
        )
        assert result.promoted is False
        assert any(c.name == "latency_budget" for c in result.failures)

    def test_extreme_popularity_bias_is_rejected(self) -> None:
        result = evaluate_gate(
            self._candidate(gini=0.99),
            self._candidate(),
            candidate_version="biased",
        )
        assert result.promoted is False

    def test_missing_metric_is_inconclusive_not_a_pass(self) -> None:
        """The most important property of the gate.

        Passing what could not be evaluated produces the feeling of a check
        without the check.
        """
        candidate = self._candidate()
        del candidate["ndcg@10"]

        result = evaluate_gate(candidate, self._candidate(), candidate_version="partial")
        assert result.promoted is False
        assert any(c.verdict is Verdict.INCONCLUSIVE for c in result.checks)

    def test_small_regression_within_tolerance_is_allowed(self) -> None:
        """Zero tolerance rejects equivalent models and trains people to bypass."""
        result = evaluate_gate(
            self._candidate(**{"ndcg@10": 0.0213}),  # -0.9%
            self._candidate(**{"ndcg@10": 0.0215}),
            candidate_version="equivalent",
            config=GateConfig(max_primary_regression=0.02),
        )
        assert result.promoted is True

    def test_first_model_with_no_incumbent_still_faces_absolute_checks(self) -> None:
        result = evaluate_gate(
            self._candidate(catalog_coverage=0.001),
            None,
            candidate_version="first",
        )
        assert result.promoted is False, "the absolute coverage floor must still apply"

    def test_summary_names_the_failing_checks(self) -> None:
        """'Rejected' alone tells an engineer nothing about what to fix."""
        result = evaluate_gate(
            self._candidate(p99_latency_ms=9999.0),
            self._candidate(),
            candidate_version="slow",
        )
        assert "latency_budget" in result.summary()
