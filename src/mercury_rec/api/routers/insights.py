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
from typing import Any

from fastapi import APIRouter, HTTPException, Request, status

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
        """Interpolate a percentile from cumulative histogram buckets."""
        stage_buckets = sorted(buckets.get(stage, []))
        total = counts.get(stage, 0.0)
        if total == 0 or not stage_buckets:
            return None
        target = total * quantile
        for upper, cumulative in stage_buckets:
            if cumulative >= target:
                return round(upper * 1000.0, 3)  # seconds -> ms
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
            "Percentiles are upper bucket bounds from this process's own "
            "histogram, so they are conservative rather than exact."
        ),
        "stages": stages,
        "cache": cache.stats.as_dict(),
    }


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
