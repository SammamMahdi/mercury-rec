"""Distribution drift between a reference window and a current one.

Three statistics, because no single one answers the question on its own:

* **PSI** (Population Stability Index) is the industry default for tabular
  features and has widely-recognised thresholds, which is most of its value:
  people already know what 0.25 means.
* **KS** (Kolmogorov-Smirnov) is the largest gap between the two cumulative
  distributions. It is sensitive to a shift in the body that PSI's coarse
  binning can average away.
* **JS** (Jensen-Shannon) divergence is bounded, symmetric, and finite even
  when one distribution has mass where the other has none - the case that
  makes KL divergence return infinity and a dashboard show a blank cell.

They disagree sometimes, and the disagreement is informative: PSI high with KS
low usually means the tails moved, not the centre.

**What drift is and is not.** A drift score is not a model quality metric. A
feature can shift substantially with no effect on ranking, and a model can
degrade badly with every input distribution unchanged. These numbers say the
world the model sees has changed shape - that is a prompt to look, not a
verdict.

The binning is the part most implementations get wrong. Bin edges are taken
from the *reference* window only and then applied to both. Re-binning on the
combined data makes the statistic depend on the thing being measured, so a
large shift partly hides itself by moving the bins with it.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np
import pandas as pd

from mercury_rec.core.logging import get_logger

logger = get_logger(__name__)

#: Conventional PSI thresholds. Widely used, not derived from this dataset;
#: they are reported as the industry rule of thumb they are.
PSI_MINOR = 0.10
PSI_MAJOR = 0.25

#: Laplace-style floor applied to empty bins before taking a logarithm. An
#: empty bin in either window makes PSI infinite, which says "a bin is empty"
#: rather than "this feature drifted infinitely far".
_EPSILON = 1e-6

Severity = Literal["stable", "minor", "major"]


@dataclass(frozen=True, slots=True)
class FeatureDrift:
    """Drift statistics for one feature."""

    feature: str
    psi: float
    ks_statistic: float
    js_divergence: float
    reference_mean: float
    current_mean: float
    reference_n: int
    current_n: int
    severity: Severity

    def as_dict(self) -> dict[str, float | str | int]:
        return {
            "feature": self.feature,
            "psi": round(self.psi, 6),
            "ks_statistic": round(self.ks_statistic, 6),
            "js_divergence": round(self.js_divergence, 6),
            "reference_mean": round(self.reference_mean, 6),
            "current_mean": round(self.current_mean, 6),
            "reference_n": self.reference_n,
            "current_n": self.current_n,
            "severity": self.severity,
        }


def _reference_edges(reference: np.ndarray, bins: int) -> np.ndarray:
    """Quantile bin edges taken from the reference window alone.

    Quantiles rather than equal width: as-of features are heavily skewed
    (most users have few events, a handful have thousands), and equal-width
    bins would put 99% of the mass in the first bin, where no realistic shift
    can register.

    Duplicate edges are collapsed, which happens whenever a feature is mostly
    one value - a real and common case for count features.
    """
    quantiles = np.linspace(0.0, 1.0, bins + 1)
    edges = np.unique(np.quantile(reference, quantiles))
    if edges.size < 2:
        # A constant reference feature. One bin spanning everything, so any
        # change registers as a shift in the current window's share.
        span = max(abs(float(edges[0])) * 1e-6, 1e-6) if edges.size else 1e-6
        edges = np.array([edges[0] - span, edges[0] + span]) if edges.size else np.array([0.0, 1.0])

    # Open the outer edges so values beyond the reference range are counted
    # into the end bins rather than dropped. Silently discarding them would
    # hide exactly the drift worth catching.
    edges[0] = -np.inf
    edges[-1] = np.inf
    return edges


def _proportions(values: np.ndarray, edges: np.ndarray) -> np.ndarray:
    counts, _ = np.histogram(values, bins=edges)
    total = counts.sum()
    if total == 0:
        return np.full(len(counts), 1.0 / len(counts))
    proportions: np.ndarray = np.maximum(counts / total, _EPSILON)
    return proportions


def population_stability_index(
    reference: np.ndarray, current: np.ndarray, *, bins: int = 10
) -> float:
    """PSI between two samples, binned on the reference distribution."""
    edges = _reference_edges(reference, bins)
    expected = _proportions(reference, edges)
    actual = _proportions(current, edges)
    return float(np.sum((actual - expected) * np.log(actual / expected)))


def ks_statistic(reference: np.ndarray, current: np.ndarray) -> float:
    """Two-sample Kolmogorov-Smirnov statistic.

    Implemented directly rather than pulled from SciPy: it is a sort and a
    max-difference, the project does not otherwise need SciPy, and a
    dependency carried for one closed-form statistic is a dependency that will
    be carried into every container image.
    """
    if reference.size == 0 or current.size == 0:
        return 0.0

    combined = np.sort(np.concatenate([reference, current]))
    reference_cdf = np.searchsorted(np.sort(reference), combined, side="right") / reference.size
    current_cdf = np.searchsorted(np.sort(current), combined, side="right") / current.size
    return float(np.max(np.abs(reference_cdf - current_cdf)))


def jensen_shannon_divergence(
    reference: np.ndarray, current: np.ndarray, *, bins: int = 10
) -> float:
    """JS divergence in bits, bounded to [0, 1].

    Bounded and symmetric, unlike KL, so it stays readable when one window has
    mass where the other has none.
    """
    edges = _reference_edges(reference, bins)
    p = _proportions(reference, edges)
    q = _proportions(current, edges)
    m = 0.5 * (p + q)

    def kl(a: np.ndarray, b: np.ndarray) -> float:
        return float(np.sum(a * np.log2(a / b)))

    return 0.5 * kl(p, m) + 0.5 * kl(q, m)


def _severity(psi: float) -> Severity:
    if psi >= PSI_MAJOR:
        return "major"
    if psi >= PSI_MINOR:
        return "minor"
    return "stable"


def detect_drift(
    reference: pd.DataFrame,
    current: pd.DataFrame,
    *,
    features: list[str] | None = None,
    bins: int = 10,
) -> list[FeatureDrift]:
    """Compare two feature windows, one row per feature.

    Args:
        reference: The window the model was trained on.
        current: The window being served, or evaluated.
        features: Columns to compare. Defaults to every numeric column present
            in both frames.
        bins: Quantile bins for PSI and JS.

    Returns:
        One :class:`FeatureDrift` per feature, sorted by PSI descending so the
        thing worth looking at is first.
    """
    if features is None:
        shared = [column for column in reference.columns if column in current.columns]
        features = [column for column in shared if pd.api.types.is_numeric_dtype(reference[column])]

    results: list[FeatureDrift] = []
    for feature in features:
        reference_values = reference[feature].to_numpy(dtype=np.float64)
        current_values = current[feature].to_numpy(dtype=np.float64)

        reference_values = reference_values[np.isfinite(reference_values)]
        current_values = current_values[np.isfinite(current_values)]
        if reference_values.size == 0 or current_values.size == 0:
            logger.warning("drift.feature_empty", feature=feature)
            continue

        psi = population_stability_index(reference_values, current_values, bins=bins)
        results.append(
            FeatureDrift(
                feature=feature,
                psi=psi,
                ks_statistic=ks_statistic(reference_values, current_values),
                js_divergence=jensen_shannon_divergence(
                    reference_values, current_values, bins=bins
                ),
                reference_mean=float(reference_values.mean()),
                current_mean=float(current_values.mean()),
                reference_n=int(reference_values.size),
                current_n=int(current_values.size),
                severity=_severity(psi),
            )
        )

    results.sort(key=lambda entry: entry.psi, reverse=True)
    logger.info(
        "drift.computed",
        features=len(results),
        major=sum(1 for entry in results if entry.severity == "major"),
    )
    return results


__all__ = [
    "PSI_MAJOR",
    "PSI_MINOR",
    "FeatureDrift",
    "detect_drift",
    "jensen_shannon_divergence",
    "ks_statistic",
    "population_stability_index",
]
