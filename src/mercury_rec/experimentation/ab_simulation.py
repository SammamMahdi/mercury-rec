"""Offline A/B simulation with honest uncertainty.

**This is offline, counterfactual evaluation on logged data. There are no
live users.** Every result this module produces is labelled ``is_simulated``,
the database column defaults to true, and the frontend renders it behind a
warning badge. Presenting a simulated lift as an online experiment result is
exactly the kind of claim the project's honesty requirements forbid.

What can and cannot be concluded
--------------------------------
Two variants are scored on the same held-out window, so the comparison is
paired: each user is served by both and the difference is measured per user.
That answers "would this ranking have put more of what the user actually
engaged with near the top?" It does **not** answer "would users have engaged
more", because engagement depends on what was shown, and the log only records
outcomes for what the production system happened to show. Nothing here
corrects for that presentation bias, and no off-policy correction is claimed.

Why bootstrap rather than a t-test
----------------------------------
Per-user ranking metrics are badly non-normal: NDCG@10 is bounded in [0, 1]
and, at this dataset's density, is exactly zero for most users. A t-test
assumes approximate normality of the mean, which a spike-at-zero distribution
with a long thin tail violates badly enough to matter at these sample sizes.
The bootstrap makes no distributional assumption; it just resamples users.

Resampling is at the **user** level, not the observation level, because users
are the independent unit. Resampling individual (user, item) rows would treat
one user's ten interactions as ten independent observations and produce
confidence intervals several times too narrow - which reads as a significant
result where there is none.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from mercury_rec.core.logging import get_logger
from mercury_rec.evaluation.metrics import (
    hit_rate_at_k,
    ndcg_at_k,
    precision_at_k,
    recall_at_k,
)

logger = get_logger(__name__)

DEFAULT_BOOTSTRAP_SAMPLES = 2000


@dataclass(slots=True)
class MetricComparison:
    """One metric, compared across variants with an interval on the difference."""

    metric: str
    control: float
    treatment: float
    absolute_lift: float
    relative_lift: float
    lower_ci: float
    upper_ci: float
    p_value: float
    n_users: int

    @property
    def is_significant(self) -> bool:
        """Whether the 95% interval on the paired difference excludes zero.

        Reported alongside the interval rather than instead of it: an interval
        of [-0.001, 0.049] and one of [0.024, 0.026] can share a p-value and
        mean very different things.
        """
        return self.lower_ci > 0 or self.upper_ci < 0

    def as_dict(self) -> dict[str, Any]:
        return {
            "metric": self.metric,
            "control": round(self.control, 6),
            "treatment": round(self.treatment, 6),
            "absolute_lift": round(self.absolute_lift, 6),
            "relative_lift_pct": round(self.relative_lift * 100, 3),
            "ci_95": [round(self.lower_ci, 6), round(self.upper_ci, 6)],
            "p_value": round(self.p_value, 5),
            "is_significant": self.is_significant,
            "n_users": self.n_users,
        }


@dataclass(slots=True)
class SimulationResult:
    """A full offline comparison of two ranking variants."""

    control_name: str
    treatment_name: str
    n_users: int
    comparisons: list[MetricComparison] = field(default_factory=list)
    bootstrap_samples: int = DEFAULT_BOOTSTRAP_SAMPLES
    is_simulated: bool = True
    """Always true here. This module cannot produce a live result, and the
    field exists so a consumer never has to infer it."""

    def as_dict(self) -> dict[str, Any]:
        return {
            "control": self.control_name,
            "treatment": self.treatment_name,
            "n_users": self.n_users,
            "bootstrap_samples": self.bootstrap_samples,
            "is_simulated": self.is_simulated,
            "disclaimer": (
                "Offline counterfactual evaluation on logged interactions. No "
                "live users were served. Measures whether a ranking places "
                "already-observed engagement higher, NOT whether engagement "
                "would have increased - the log only contains outcomes for "
                "what the production system actually showed."
            ),
            "metrics": [comparison.as_dict() for comparison in self.comparisons],
        }


def paired_bootstrap(
    control: np.ndarray,
    treatment: np.ndarray,
    *,
    n_samples: int = DEFAULT_BOOTSTRAP_SAMPLES,
    confidence: float = 0.95,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap a confidence interval and p-value for a paired difference.

    Paired because both variants score the *same* users: differencing first
    removes between-user variance, which dominates the signal here. An
    unpaired comparison would be far noisier for no benefit.

    Args:
        control: Per-user metric under control.
        treatment: Per-user metric under treatment, same users, same order.
        n_samples: Bootstrap resamples.
        confidence: Interval width.
        seed: Seeded, so a reported interval is reproducible.

    Returns:
        ``(lower, upper, p_value)`` for the mean difference.
    """
    if control.shape != treatment.shape:
        raise ValueError(
            f"Paired comparison needs matching shapes, got {control.shape} and "
            f"{treatment.shape}. Both variants must be scored on the same users."
        )
    if control.size == 0:
        return 0.0, 0.0, 1.0

    differences = treatment - control
    n = differences.size
    rng = np.random.default_rng(seed)

    # Resample USERS with replacement. Resampling rows instead would treat one
    # user's interactions as independent and shrink the interval several-fold.
    indices = rng.integers(0, n, size=(n_samples, n))
    means = differences[indices].mean(axis=1)

    alpha = 1.0 - confidence
    lower = float(np.percentile(means, 100 * alpha / 2))
    upper = float(np.percentile(means, 100 * (1 - alpha / 2)))

    # Two-sided bootstrap p-value: the share of resampled means on the far
    # side of zero from the observed effect, doubled. The +1 corrections keep
    # it from ever being exactly 0, which would overstate certainty that no
    # finite resampling can support.
    observed = float(differences.mean())
    tail = float((means <= 0).sum()) if observed >= 0 else float((means >= 0).sum())
    p_value = min(1.0, 2.0 * (tail + 1) / (n_samples + 1))

    return lower, upper, p_value


def per_user_metrics(
    recommendations: Mapping[int, Sequence[int]],
    ground_truth: Mapping[int, set[int]],
    users: Sequence[int],
    *,
    k: int = 10,
) -> dict[str, np.ndarray]:
    """Compute per-user metric vectors, aligned to ``users``.

    Per-user rather than aggregate, because the bootstrap resamples users and
    needs each one's value separately.
    """
    metrics: dict[str, list[float]] = {
        f"ndcg@{k}": [],
        f"recall@{k}": [],
        f"precision@{k}": [],
        f"hit_rate@{k}": [],
    }
    for user in users:
        ranked = list(recommendations.get(user, ()))
        relevant = ground_truth.get(user, set())
        metrics[f"ndcg@{k}"].append(ndcg_at_k(ranked, relevant, k))
        metrics[f"recall@{k}"].append(recall_at_k(ranked, relevant, k))
        metrics[f"precision@{k}"].append(precision_at_k(ranked, relevant, k))
        metrics[f"hit_rate@{k}"].append(hit_rate_at_k(ranked, relevant, k))

    return {name: np.array(values, dtype=np.float64) for name, values in metrics.items()}


def simulate_ab_test(
    control_recommendations: Mapping[int, Sequence[int]],
    treatment_recommendations: Mapping[int, Sequence[int]],
    ground_truth: Mapping[int, set[int]],
    *,
    control_name: str = "control",
    treatment_name: str = "treatment",
    k: int = 10,
    n_bootstrap: int = DEFAULT_BOOTSTRAP_SAMPLES,
    seed: int = 42,
) -> SimulationResult:
    """Compare two ranking variants on the same logged window.

    Only users present in both variants and with a non-empty truth set are
    scored: a user one variant declined to serve cannot be part of a paired
    comparison, and including them would compare different populations.
    """
    shared = sorted(
        set(control_recommendations) & set(treatment_recommendations) & set(ground_truth)
    )
    users = [user for user in shared if ground_truth[user]]

    if not users:
        logger.warning("ab.no_shared_users")
        return SimulationResult(control_name=control_name, treatment_name=treatment_name, n_users=0)

    control_metrics = per_user_metrics(control_recommendations, ground_truth, users, k=k)
    treatment_metrics = per_user_metrics(treatment_recommendations, ground_truth, users, k=k)

    comparisons: list[MetricComparison] = []
    for name, control_values in control_metrics.items():
        treatment_values = treatment_metrics[name]
        lower, upper, p_value = paired_bootstrap(
            control_values, treatment_values, n_samples=n_bootstrap, seed=seed
        )

        control_mean = float(control_values.mean())
        treatment_mean = float(treatment_values.mean())
        absolute = treatment_mean - control_mean
        relative = absolute / control_mean if control_mean > 0 else 0.0

        comparisons.append(
            MetricComparison(
                metric=name,
                control=control_mean,
                treatment=treatment_mean,
                absolute_lift=absolute,
                relative_lift=relative,
                lower_ci=lower,
                upper_ci=upper,
                p_value=p_value,
                n_users=len(users),
            )
        )

    result = SimulationResult(
        control_name=control_name,
        treatment_name=treatment_name,
        n_users=len(users),
        comparisons=comparisons,
        bootstrap_samples=n_bootstrap,
    )
    logger.info(
        "ab.simulation_complete",
        control=control_name,
        treatment=treatment_name,
        users=len(users),
        significant=[c.metric for c in comparisons if c.is_significant],
    )
    return result


__all__ = [
    "DEFAULT_BOOTSTRAP_SAMPLES",
    "MetricComparison",
    "SimulationResult",
    "paired_bootstrap",
    "per_user_metrics",
    "simulate_ab_test",
]
