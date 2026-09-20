"""Read-only endpoints backing the frontend dashboards.

Every figure served here is either a live count from the running system or a
value read out of a committed evaluation artifact. Nothing is hard-coded, and
nothing is estimated.

Where a number is not available -- no evaluation has been run, no requests
have been served yet -- these endpoints say so explicitly rather than
returning zero. A dashboard showing "P95: 0.0ms" looks like a measurement and
is in fact the absence of one; "no requests recorded yet" is the honest
rendering, and the frontend is built to display it.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Annotated, Any

from fastapi import APIRouter, HTTPException, Query, Request, status

from mercury_rec.api.deps import CacheDep, EngineDep
from mercury_rec.core.logging import get_logger

logger = get_logger(__name__)

router = APIRouter()


def _evaluation_path(request: Request, name: str) -> Path:
    preset = getattr(request.app.state, "preset", "full")
    return Path("artifacts") / "evaluation" / preset / name


@router.get("/dataset", summary="Dataset counts and provenance")
def dataset_insights(engine: EngineDep) -> dict[str, Any]:
    """Live counts from the loaded dataset, with provenance attached.

    Provenance travels with the numbers deliberately: a consumer must be able
    to tell that behaviour is real Retailrocket while merchants, regions and
    prices are synthesised, without reading the docs.
    """
    bundle = engine.bundle
    metadata_path = Path("data") / "processed" / "full" / "METADATA.json"
    metadata: dict[str, Any] = {}
    if metadata_path.is_file():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))

    return {
        "n_users": bundle.n_users,
        "n_items": bundle.n_items,
        "model_version": bundle.model_version,
        "dataset_hash": bundle.dataset_hash,
        "ingest": metadata.get("ingest", {}),
        "split": metadata.get("split", {}),
        "source": metadata.get("source", {}),
        "columns": metadata.get("columns", {}),
    }


@router.get("/evaluation", summary="Offline model comparison")
def evaluation_insights(request: Request) -> dict[str, Any]:
    """Return the committed model comparison.

    Read from ``artifacts/evaluation/<preset>/results_test.json``, which is
    written by ``mercury train baselines``. If it is absent this returns 404
    with the command that produces it, rather than inventing plausible
    numbers to fill the dashboard.
    """
    path = _evaluation_path(request, "results_test.json")
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No evaluation results available. Run `mercury train baselines` to produce them."
            ),
        )
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@router.get("/ranking", summary="Multi-stage pipeline results")
def ranking_insights(request: Request) -> dict[str, Any]:
    """Return the staged pipeline comparison and the retrieval ceiling."""
    path = _evaluation_path(request, "ranking_results.json")
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No ranking results available. Run `mercury train ranker`.",
        )
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@router.get("/experiment", summary="Offline A/B simulation and gate verdict")
def experiment_insights(request: Request) -> dict[str, Any]:
    """Return the offline A/B comparison and the promotion gate result.

    The payload carries is_simulated and a disclaimer describing exactly what
    the comparison establishes, so the frontend can render it behind a warning
    badge rather than as an online experiment result.
    """
    path = _evaluation_path(request, "experiment_results.json")
    if not path.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail="No experiment results available. Run `mercury train experiment`.",
        )
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@router.get("/metrics/summary", summary="Live latency and cache statistics")
def metrics_summary(cache: CacheDep) -> dict[str, Any]:
    """Summarise this process's own observed latency and cache behaviour.

    Percentiles are computed from the Prometheus histogram buckets this
    process has actually recorded. **When no requests have been served the
    response says so** instead of reporting zeros that would read as
    measurements.
    """
    from prometheus_client import REGISTRY

    buckets: dict[str, list[tuple[float, float]]] = {}
    counts: dict[str, float] = {}
    sums: dict[str, float] = {}

    for metric in REGISTRY.collect():
        if metric.name == "mercury_stage_duration_seconds":
            for sample in metric.samples:
                stage = sample.labels.get("stage", "unknown")
                if sample.name.endswith("_bucket"):
                    upper = float(sample.labels.get("le", "inf"))
                    buckets.setdefault(stage, []).append((upper, sample.value))
                elif sample.name.endswith("_count"):
                    counts[stage] = sample.value
                elif sample.name.endswith("_sum"):
                    sums[stage] = sample.value

    total_observations = sum(counts.values())
    if total_observations == 0:
        return {
            "has_data": False,
            "detail": (
                "No requests recorded in this process yet. Latency percentiles "
                "become available once traffic has been served."
            ),
            "cache": cache.stats.as_dict(),
        }

    def _percentile(stage: str, quantile: float) -> float | None:
        """Estimate a percentile from cumulative histogram buckets.

        Linearly interpolates within the bucket the quantile falls into,
        which is what Prometheus's own ``histogram_quantile`` does. Returning
        the bucket's upper bound instead is simpler and consistently
        overstates: with boundaries at 50ms and 100ms, a set of 60ms requests
        would be reported as a 100ms median. On a coarse bucket ladder that is
        a near-doubling, presented as a measurement.

        It remains an estimate. The histogram keeps counts, not samples, so
        the true value is unrecoverable - the response says so.
        """
        stage_buckets = sorted(buckets.get(stage, []))
        total = counts.get(stage, 0.0)
        if total == 0 or not stage_buckets:
            return None

        target = total * quantile
        previous_upper = 0.0
        previous_cumulative = 0.0

        for upper, cumulative in stage_buckets:
            if cumulative < target:
                previous_upper, previous_cumulative = upper, cumulative
                continue

            if upper == float("inf"):
                # Everything above the last finite boundary. There is no upper
                # edge to interpolate towards, so report that boundary and let
                # it read as "at least this".
                return round(previous_upper * 1000.0, 3)

            in_bucket = cumulative - previous_cumulative
            if in_bucket <= 0:
                return round(upper * 1000.0, 3)

            fraction = (target - previous_cumulative) / in_bucket
            estimate = previous_upper + fraction * (upper - previous_upper)
            return round(estimate * 1000.0, 3)  # seconds -> ms

        return None

    stages = {
        stage: {
            "count": int(counts.get(stage, 0)),
            "mean_ms": round((sums.get(stage, 0.0) / counts[stage]) * 1000.0, 3)
            if counts.get(stage)
            else None,
            "p50_ms": _percentile(stage, 0.50),
            "p95_ms": _percentile(stage, 0.95),
            "p99_ms": _percentile(stage, 0.99),
        }
        for stage in counts
    }

    return {
        "has_data": True,
        "note": (
            "Percentiles are estimated from this process's own histogram by "
            "interpolating within the bucket each quantile falls into. A "
            "histogram stores counts rather than samples, so these are "
            "approximations, and they describe only the requests this process "
            "has served since it started."
        ),
        "stages": stages,
        "cache": cache.stats.as_dict(),
    }


@router.get("/drift", summary="Feature drift between two windows")
def drift_insights(request: Request) -> dict[str, Any]:
    """Serve the committed drift report.

    Read from an artifact rather than computed per request: comparing two
    quarter-million-row windows takes seconds and allocates hundreds of
    megabytes, which is not something an interactive dashboard should trigger
    on every page load.
    """
    path = _evaluation_path(request, "drift.json")
    monitoring_path = (
        Path("artifacts") / "monitoring" / getattr(request.app.state, "preset", "full")
    ) / "drift.json"
    chosen = monitoring_path if monitoring_path.is_file() else path

    if not chosen.is_file():
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=(
                "No drift report has been produced. Run `mercury monitor drift` "
                "to compare the training window against the held-out one."
            ),
        )
    return json.loads(chosen.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


@router.get("/sample-users", summary="Real user ids to explore with")
def sample_users(
    engine: EngineDep,
    n: Annotated[int, Query(ge=1, le=50)] = 8,
) -> dict[str, Any]:
    """Return a spread of real user ids, with how much history each one has.

    Without this the explorer asks for a user id the visitor has no way to
    know, and the only reachable path through the product is the cold-start
    one. The ids are drawn across the history distribution rather than from
    its head, because a demo that only ever shows the most active user in the
    dataset is showing the easiest case and calling it typical.

    Sorted deterministically, so a reload does not reshuffle the list.
    """
    bundle = engine.bundle
    if not bundle.user_history:
        return {"users": [], "detail": "No training history is loaded."}

    # Internal index -> external id, which is what the API accepts.
    external = {internal: source for source, internal in bundle.user_index.items()}

    ranked = sorted(
        ((internal, len(items)) for internal, items in bundle.user_history.items()),
        key=lambda pair: (-pair[1], pair[0]),
    )
    if not ranked:
        return {"users": [], "detail": "No training history is loaded."}

    # Evenly spaced positions in the history distribution: the busiest user,
    # the quietest, and a ladder between them.
    last = len(ranked) - 1
    step = max(1, len(ranked) // n)
    two_tower = bundle.two_tower

    users: list[dict[str, Any]] = []
    for rank in range(0, len(ranked), step):
        if len(users) == n:
            break
        internal, count = ranked[rank]
        users.append(
            {
                "user_id": external.get(internal, str(internal)),
                "history_items": count,
                "percentile": round(100.0 * (1.0 - rank / max(last, 1)), 1),
                "has_two_tower_embedding": (
                    two_tower is not None and two_tower.user_vector(internal) is not None
                ),
            }
        )

    return {"users": users, "n_users_with_history": len(ranked)}


@router.get("/config", summary="Effective serving configuration")
def config_insights(engine: EngineDep) -> dict[str, Any]:
    """Expose the knobs that shape a recommendation.

    Deliberately excludes every connection string and credential: this
    endpoint is reachable by the browser.
    """
    rerank = engine.rerank_config
    return {
        "per_source_k": engine.per_source_k,
        "max_candidates": engine.max_candidates,
        "rerank": {
            "max_per_merchant": rerank.max_per_merchant,
            "max_per_category": rerank.max_per_category,
            "enforce_availability": rerank.enforce_availability,
            "sponsored_boost": rerank.sponsored_boost,
            "popularity_penalty": rerank.popularity_penalty,
        },
        "cache": {
            "enabled": engine.cache.enabled,
            "ttl_seconds": engine.cache.ttl_seconds,
            "soft_ttl_seconds": engine.cache.soft_ttl_seconds,
        },
    }


__all__ = ["router"]
