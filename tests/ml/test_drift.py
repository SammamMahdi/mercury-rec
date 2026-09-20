"""Drift statistics.

These are easy to implement in a way that looks right and is not. The classic
failures are binning on the combined sample (which hides the very shift being
measured), returning infinity when a bin is empty, and quietly dropping values
outside the reference range — which discards exactly the drift worth catching.

Each of those has a test here.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mercury_rec.monitoring.drift import (
    PSI_MAJOR,
    detect_drift,
    jensen_shannon_divergence,
    ks_statistic,
    population_stability_index,
)


@pytest.fixture
def reference() -> np.ndarray:
    return np.random.default_rng(17).normal(loc=0.0, scale=1.0, size=5_000)


class TestPopulationStabilityIndex:
    def test_identical_samples_score_near_zero(self, reference: np.ndarray) -> None:
        assert population_stability_index(reference, reference.copy()) < 1e-9

    def test_a_shift_registers(self, reference: np.ndarray) -> None:
        shifted = reference + 1.5
        assert population_stability_index(reference, shifted) > PSI_MAJOR

    def test_a_larger_shift_scores_higher(self, reference: np.ndarray) -> None:
        small = population_stability_index(reference, reference + 0.3)
        large = population_stability_index(reference, reference + 1.5)
        assert large > small

    def test_is_finite_when_a_bin_is_empty(self, reference: np.ndarray) -> None:
        """Disjoint supports must not return infinity.

        Infinity is not a measurement, it is a report that a bin was empty,
        and it turns a dashboard cell blank exactly when the drift is worst.
        """
        disjoint = reference + 50.0
        score = population_stability_index(reference, disjoint)
        assert np.isfinite(score)
        assert score > PSI_MAJOR

    def test_values_beyond_the_reference_range_are_not_dropped(self) -> None:
        """Out-of-range values belong in the end bins, not the bin.

        If they were discarded, a current window that moved entirely past the
        reference range would score as perfectly stable — the single worst
        possible failure for a drift detector.
        """
        reference = np.linspace(0.0, 1.0, 1_000)
        far_above = np.linspace(10.0, 11.0, 1_000)
        assert population_stability_index(reference, far_above) > PSI_MAJOR

    def test_binning_uses_the_reference_only(self) -> None:
        """The statistic must not depend on re-binning the combined sample.

        Binning on the union lets a large shift move the bins with it, which
        damps the score precisely when it should be largest. Holding the
        reference fixed while the current window moves further away must
        produce monotonically increasing scores.
        """
        reference = np.random.default_rng(3).normal(size=4_000)
        scores = [
            population_stability_index(reference, reference + offset)
            for offset in (0.5, 1.0, 2.0, 4.0)
        ]
        assert scores == sorted(scores)

    def test_a_constant_feature_does_not_raise(self) -> None:
        """Count features are often a single value in a short window."""
        constant = np.zeros(500)
        assert np.isfinite(population_stability_index(constant, constant))
        assert np.isfinite(population_stability_index(constant, np.ones(500)))


class TestKsStatistic:
    def test_identical_samples_score_zero(self, reference: np.ndarray) -> None:
        assert ks_statistic(reference, reference.copy()) == pytest.approx(0.0, abs=1e-12)

    def test_disjoint_samples_score_one(self) -> None:
        assert ks_statistic(np.zeros(100), np.ones(100)) == pytest.approx(1.0)

    def test_is_bounded(self, reference: np.ndarray) -> None:
        value = ks_statistic(reference, reference + 2.0)
        assert 0.0 <= value <= 1.0

    def test_is_symmetric(self, reference: np.ndarray) -> None:
        current = reference + 0.8
        assert ks_statistic(reference, current) == pytest.approx(ks_statistic(current, reference))

    def test_an_empty_sample_is_not_an_error(self, reference: np.ndarray) -> None:
        assert ks_statistic(reference, np.array([])) == 0.0


class TestJensenShannonDivergence:
    def test_identical_samples_score_zero(self, reference: np.ndarray) -> None:
        assert jensen_shannon_divergence(reference, reference.copy()) == pytest.approx(
            0.0, abs=1e-9
        )

    def test_is_bounded_by_one_bit(self) -> None:
        """Bounded, unlike KL, which is the reason it is here."""
        value = jensen_shannon_divergence(np.zeros(1_000), np.ones(1_000) * 100)
        assert 0.0 <= value <= 1.0 + 1e-9

    def test_grows_with_separation(self, reference: np.ndarray) -> None:
        near = jensen_shannon_divergence(reference, reference + 0.2)
        far = jensen_shannon_divergence(reference, reference + 3.0)
        assert far > near


class TestDetectDrift:
    @staticmethod
    def _frames() -> tuple[pd.DataFrame, pd.DataFrame]:
        rng = np.random.default_rng(29)
        n = 3_000
        reference = pd.DataFrame(
            {
                "stable_feature": rng.normal(size=n),
                "shifted_feature": rng.normal(size=n),
                "label": ["a"] * n,
            }
        )
        current = pd.DataFrame(
            {
                "stable_feature": rng.normal(size=n),
                "shifted_feature": rng.normal(size=n) + 2.0,
                "label": ["b"] * n,
            }
        )
        return reference, current

    def test_sorts_the_worst_offender_first(self) -> None:
        reference, current = self._frames()
        results = detect_drift(reference, current)

        assert results[0].feature == "shifted_feature"
        assert results[0].severity == "major"

    def test_a_stable_feature_is_labelled_stable(self) -> None:
        reference, current = self._frames()
        results = {entry.feature: entry for entry in detect_drift(reference, current)}
        assert results["stable_feature"].severity == "stable"

    def test_non_numeric_columns_are_skipped(self) -> None:
        reference, current = self._frames()
        features = {entry.feature for entry in detect_drift(reference, current)}
        assert "label" not in features

    def test_explicit_feature_selection_is_honoured(self) -> None:
        reference, current = self._frames()
        results = detect_drift(reference, current, features=["stable_feature"])
        assert [entry.feature for entry in results] == ["stable_feature"]

    def test_reports_the_sample_sizes_it_used(self) -> None:
        """A drift score from 12 rows deserves less trust than one from 12,000,
        and the report has to carry enough to tell them apart."""
        reference, current = self._frames()
        entry = detect_drift(reference, current)[0]
        assert entry.reference_n == len(reference)
        assert entry.current_n == len(current)

    def test_non_finite_values_are_excluded(self) -> None:
        """NaN and inf must not silently become bins of their own."""
        reference = pd.DataFrame({"x": np.concatenate([np.zeros(100), [np.nan, np.inf]])})
        current = pd.DataFrame({"x": np.zeros(100)})

        entry = detect_drift(reference, current)[0]
        assert entry.reference_n == 100
        assert np.isfinite(entry.psi)

    def test_a_fully_empty_feature_is_dropped_rather_than_reported(self) -> None:
        reference = pd.DataFrame({"x": [np.nan, np.nan]})
        current = pd.DataFrame({"x": [1.0, 2.0]})
        assert detect_drift(reference, current) == []
