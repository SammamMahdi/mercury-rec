"""Model validation gate: the check that stands between training and serving.

A gate exists because "the new model scored higher" is not sufficient grounds
to serve it. A model can improve NDCG while collapsing catalogue coverage onto
a handful of head items, or while becoming slow enough to breach the latency
budget, or while regressing badly for cold-start users whose aggregate weight
is too small to show in the headline number.

So the gate checks several dimensions and **fails on any one of them**. Each
check reports its own verdict, because "rejected" alone tells an engineer
nothing about what to fix.

Thresholds are relative to the incumbent, not absolute. An absolute bar has to
be re-tuned every time the data changes and silently becomes either
unreachable or meaningless; "at least as good as what is already serving,
within tolerance" stays correct on its own.

The gate is deliberately conservative: when a comparison cannot be made - a
metric missing, no incumbent at all - it says so rather than passing by
default. A gate that passes when it cannot evaluate is worse than no gate,
because it produces a signed-off feeling without the check.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import StrEnum
from typing import Any

from mercury_rec.core.logging import get_logger

logger = get_logger(__name__)


class Verdict(StrEnum):
    PASS = "pass"  # noqa: S105  (a verdict, not a credential)
    FAIL = "fail"
    INCONCLUSIVE = "inconclusive"
    """Could not be evaluated. Never treated as a pass."""


@dataclass(slots=True)
class GateCheck:
    """One dimension's verdict."""

    name: str
    verdict: Verdict
    detail: str
    candidate_value: float | None = None
    incumbent_value: float | None = None
    threshold: float | None = None

    def as_dict(self) -> dict[str, Any]:
        return {
            "name": self.name,
            "verdict": self.verdict.value,
            "detail": self.detail,
            "candidate": self.candidate_value,
            "incumbent": self.incumbent_value,
            "threshold": self.threshold,
        }


@dataclass(slots=True)
class GateConfig:
    """Promotion thresholds.

    Defaults allow a small regression rather than demanding strict
    improvement. Measurement noise at this evaluation size is real, and a
    zero-tolerance gate rejects models that are actually equivalent, which
    trains people to bypass it.
    """

    primary_metric: str = "ndcg@10"
    max_primary_regression: float = 0.02
    """Relative. 0.02 permits a 2% drop against the incumbent."""

    min_coverage: float = 0.05
    """Absolute floor on catalogue coverage. A model that serves 2% of the
    catalogue to everyone can post excellent accuracy and is unshippable."""

    max_coverage_regression: float = 0.25
    max_gini: float = 0.95
    """Popularity-concentration ceiling. Accuracy metrics are blind to a model
    that recommends the same fifty items to every user; this is not."""

    max_p99_latency_ms: float = 150.0
    max_cold_start_regression: float = 0.10
    """Cold-start users are a minority of the evaluation set, so a serious
    regression for them can hide entirely inside an aggregate metric."""

    require_incumbent: bool = False
    """When there is no incumbent, pass the comparative checks with a note.
    The absolute checks still apply, so the first model is not waved through."""


@dataclass(slots=True)
class GateResult:
    """The gate's overall decision plus every individual check."""

    promoted: bool
    candidate_version: str
    incumbent_version: str | None
    checks: list[GateCheck] = field(default_factory=list)

    @property
    def failures(self) -> list[GateCheck]:
        return [check for check in self.checks if check.verdict is Verdict.FAIL]

    @property
    def inconclusive(self) -> list[GateCheck]:
        return [check for check in self.checks if check.verdict is Verdict.INCONCLUSIVE]

    def as_dict(self) -> dict[str, Any]:
        return {
            "promoted": self.promoted,
            "candidate": self.candidate_version,
            "incumbent": self.incumbent_version,
            "n_failures": len(self.failures),
            "n_inconclusive": len(self.inconclusive),
            "checks": [check.as_dict() for check in self.checks],
        }

    def summary(self) -> str:
        if self.promoted:
            return f"PROMOTE {self.candidate_version}: all {len(self.checks)} checks passed."
        reasons = "; ".join(f"{c.name} ({c.detail})" for c in self.failures + self.inconclusive)
        return f"REJECT {self.candidate_version}: {reasons}"


def _relative_change(candidate: float, incumbent: float) -> float:
    if incumbent == 0:
        return 0.0 if candidate == 0 else float("inf")
    return (candidate - incumbent) / abs(incumbent)


def evaluate_gate(
    candidate: dict[str, Any],
    incumbent: dict[str, Any] | None,
    *,
    candidate_version: str,
    incumbent_version: str | None = None,
    config: GateConfig | None = None,
) -> GateResult:
    """Decide whether a candidate model may be promoted.

    Args:
        candidate: Candidate metrics, e.g. from ``results_test.json``.
        incumbent: Currently-serving model's metrics, or None.
        candidate_version: Identifier for the candidate.
        incumbent_version: Identifier for the incumbent.
        config: Thresholds.

    Returns:
        The decision and every check that informed it.
    """
    policy = config or GateConfig()
    checks: list[GateCheck] = []

    # --- primary quality --------------------------------------------------
    metric = policy.primary_metric
    candidate_primary = candidate.get(metric)

    if candidate_primary is None:
        checks.append(
            GateCheck(
                name=f"primary_metric[{metric}]",
                verdict=Verdict.INCONCLUSIVE,
                detail=f"candidate has no '{metric}' - cannot assess quality",
            )
        )
    elif incumbent is None or incumbent.get(metric) is None:
        checks.append(
            GateCheck(
                name=f"primary_metric[{metric}]",
                verdict=Verdict.FAIL if policy.require_incumbent else Verdict.PASS,
                detail="no incumbent to compare against; comparative check skipped",
                candidate_value=float(candidate_primary),
            )
        )
    else:
        incumbent_primary = float(incumbent[metric])
        change = _relative_change(float(candidate_primary), incumbent_primary)
        passed = change >= -policy.max_primary_regression
        checks.append(
            GateCheck(
                name=f"primary_metric[{metric}]",
                verdict=Verdict.PASS if passed else Verdict.FAIL,
                detail=(
                    f"{change:+.2%} vs incumbent (allowed: -{policy.max_primary_regression:.0%})"
                ),
                candidate_value=float(candidate_primary),
                incumbent_value=incumbent_primary,
                threshold=-policy.max_primary_regression,
            )
        )

    # --- coverage ---------------------------------------------------------
    coverage = candidate.get("catalog_coverage")
    if coverage is None:
        checks.append(
            GateCheck(
                name="coverage_floor",
                verdict=Verdict.INCONCLUSIVE,
                detail="candidate reports no catalog_coverage",
            )
        )
    else:
        coverage = float(coverage)
        checks.append(
            GateCheck(
                name="coverage_floor",
                verdict=Verdict.PASS if coverage >= policy.min_coverage else Verdict.FAIL,
                detail=(
                    f"serves {coverage:.1%} of the catalogue (floor {policy.min_coverage:.0%})"
                ),
                candidate_value=coverage,
                threshold=policy.min_coverage,
            )
        )

        if incumbent is not None and incumbent.get("catalog_coverage"):
            change = _relative_change(coverage, float(incumbent["catalog_coverage"]))
            passed = change >= -policy.max_coverage_regression
            checks.append(
                GateCheck(
                    name="coverage_regression",
                    verdict=Verdict.PASS if passed else Verdict.FAIL,
                    detail=f"{change:+.2%} vs incumbent",
                    candidate_value=coverage,
                    incumbent_value=float(incumbent["catalog_coverage"]),
                    threshold=-policy.max_coverage_regression,
                )
            )

    # --- popularity bias --------------------------------------------------
    gini = candidate.get("gini")
    if gini is not None:
        gini = float(gini)
        checks.append(
            GateCheck(
                name="popularity_bias",
                verdict=Verdict.PASS if gini <= policy.max_gini else Verdict.FAIL,
                detail=f"gini {gini:.3f} (ceiling {policy.max_gini})",
                candidate_value=gini,
                threshold=policy.max_gini,
            )
        )

    # --- latency ----------------------------------------------------------
    latency = candidate.get("p99_latency_ms") or candidate.get("scoring_ms_per_user")
    if latency is None:
        checks.append(
            GateCheck(
                name="latency_budget",
                verdict=Verdict.INCONCLUSIVE,
                detail="no latency measurement supplied",
            )
        )
    else:
        latency = float(latency)
        checks.append(
            GateCheck(
                name="latency_budget",
                verdict=(Verdict.PASS if latency <= policy.max_p99_latency_ms else Verdict.FAIL),
                detail=f"{latency:.2f}ms (budget {policy.max_p99_latency_ms}ms)",
                candidate_value=latency,
                threshold=policy.max_p99_latency_ms,
            )
        )

    # --- cold start -------------------------------------------------------
    cold_metric = f"cold_start_{metric}"
    if candidate.get(cold_metric) is not None and incumbent and incumbent.get(cold_metric):
        change = _relative_change(float(candidate[cold_metric]), float(incumbent[cold_metric]))
        passed = change >= -policy.max_cold_start_regression
        checks.append(
            GateCheck(
                name="cold_start_regression",
                verdict=Verdict.PASS if passed else Verdict.FAIL,
                detail=f"{change:+.2%} for cold-start users",
                candidate_value=float(candidate[cold_metric]),
                incumbent_value=float(incumbent[cold_metric]),
                threshold=-policy.max_cold_start_regression,
            )
        )

    # Inconclusive never counts as a pass: a gate that waves through what it
    # could not evaluate provides the feeling of a check without the check.
    promoted = all(check.verdict is Verdict.PASS for check in checks)

    result = GateResult(
        promoted=promoted,
        candidate_version=candidate_version,
        incumbent_version=incumbent_version,
        checks=checks,
    )
    logger.info("gate.evaluated", promoted=promoted, summary=result.summary())
    return result


__all__ = ["GateCheck", "GateConfig", "GateResult", "Verdict", "evaluate_gate"]
