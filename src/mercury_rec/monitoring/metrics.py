"""Prometheus instrumentation.

Metric design follows two rules that matter more than the metric list itself.

**Latency is a histogram, never a gauge or an average.** A mean latency of
40ms is compatible with every request taking 40ms and with 95% taking 5ms
while 5% take 700ms, and those are completely different systems. Only a
histogram answers "what does the slow tail look like", which is the question
that actually matters. The buckets below are chosen around this service's
observed range rather than left at the library defaults, which are tuned for
whole-request web latency and would put almost everything here in one bucket.

**Labels are bounded.** Every distinct label combination is a separate time
series. A ``user_id`` label would create one series per user and take the
monitoring stack down long before it helped, so labels are restricted to
values from small closed sets: stage name, endpoint, model version.

The counters here are also what the frontend's dashboard reads. Its cache
hit-rate and latency percentiles are computed from these real observations,
which is why that page is allowed to display them at all.
"""

from __future__ import annotations

from typing import Final

from prometheus_client import Counter, Gauge, Histogram

#: Per-stage buckets, in seconds. The stages here run in single-digit
#: milliseconds to low tens, so the resolution is concentrated there; the
#: library default (starting at 5ms, ending at 10s) would collapse nearly
#: every observation into the first bucket.
_STAGE_BUCKETS: Final = (
    0.0005,
    0.001,
    0.0025,
    0.005,
    0.01,
    0.025,
    0.05,
    0.1,
    0.25,
    0.5,
    1.0,
    2.5,
)

REQUESTS: Final = Counter(
    "mercury_recommendation_requests_total",
    "Recommendation requests served.",
    ["endpoint", "model_version", "cold_start"],
)

STAGE_LATENCY: Final = Histogram(
    "mercury_stage_duration_seconds",
    "Time spent in each pipeline stage.",
    ["stage"],
    buckets=_STAGE_BUCKETS,
)

CACHE_REQUESTS: Final = Counter(
    "mercury_cache_requests_total",
    "Cache reads by result. The dashboard's hit rate is derived from these.",
    ["result"],
)

CANDIDATES: Final = Histogram(
    "mercury_candidates_generated",
    "Candidates reaching the ranking stage.",
    buckets=(0, 10, 50, 100, 200, 300, 400, 600, 800, 1000),
)

RECOMMENDATIONS_RETURNED: Final = Histogram(
    "mercury_recommendations_returned",
    "Items returned per request. Below k means filters removed candidates.",
    buckets=(0, 1, 5, 10, 20, 50, 100),
)

MODEL_INFO: Final = Gauge(
    "mercury_model_info",
    "Currently-served model version. Value is always 1; the label carries it.",
    ["model_version", "dataset_hash"],
)

ERRORS: Final = Counter(
    "mercury_errors_total",
    "Errors by kind.",
    ["kind"],
)

DRIFT_SCORE: Final = Gauge(
    "mercury_drift_score",
    "Population Stability Index per monitored feature.",
    ["feature"],
)


def observe_recommendation(
    *,
    endpoint: str,
    model_version: str,
    cold_start: bool,
    n_candidates: int,
    n_returned: int,
) -> None:
    """Record one served recommendation request."""
    REQUESTS.labels(
        endpoint=endpoint, model_version=model_version, cold_start=str(cold_start).lower()
    ).inc()
    CANDIDATES.observe(n_candidates)
    RECOMMENDATIONS_RETURNED.observe(n_returned)


def observe_stage_latency(timings_ms: dict[str, float]) -> None:
    """Record per-stage durations.

    Accepts milliseconds because that is what the engine measures, and
    converts to seconds because Prometheus convention is base units. Mixing
    the two is a common and very confusing dashboard bug.
    """
    for name, milliseconds in timings_ms.items():
        stage = name.removesuffix("_ms")
        STAGE_LATENCY.labels(stage=stage).observe(milliseconds / 1000.0)


def observe_cache(*, hit: bool, stale: bool = False) -> None:
    result = "stale" if stale else ("hit" if hit else "miss")
    CACHE_REQUESTS.labels(result=result).inc()


def record_error(kind: str) -> None:
    ERRORS.labels(kind=kind).inc()


def set_model_info(model_version: str, dataset_hash: str | None) -> None:
    """Publish the serving model version as a labelled gauge.

    The info-gauge pattern: the value is meaningless, the labels are the
    payload. It makes "which model produced this?" answerable directly from
    the metrics, without correlating against a deploy log.
    """
    MODEL_INFO.labels(model_version=model_version, dataset_hash=dataset_hash or "unknown").set(1)


def set_drift_score(feature: str, psi: float) -> None:
    DRIFT_SCORE.labels(feature=feature).set(psi)


__all__ = [
    "CACHE_REQUESTS",
    "CANDIDATES",
    "DRIFT_SCORE",
    "ERRORS",
    "MODEL_INFO",
    "RECOMMENDATIONS_RETURNED",
    "REQUESTS",
    "STAGE_LATENCY",
    "observe_cache",
    "observe_recommendation",
    "observe_stage_latency",
    "record_error",
    "set_drift_score",
    "set_model_info",
]
