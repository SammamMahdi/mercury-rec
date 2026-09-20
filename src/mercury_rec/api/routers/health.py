"""Liveness, readiness and Prometheus metrics endpoints.

``/health`` and ``/ready`` answer different questions and are deliberately not
the same endpoint:

- **/health** -- is the process alive? Used by the orchestrator to decide
  whether to restart. It must not depend on Redis or on model artifacts,
  because restarting the process cannot fix either, and a health check that
  fails on a dependency outage turns a degraded service into a crash loop.
- **/ready** -- can this instance serve traffic? Used by the load balancer to
  decide whether to route. It reports each dependency separately, because
  during an incident "not ready" is useless while "ranker failed to load" is
  actionable.
"""

from __future__ import annotations

import time

from fastapi import APIRouter, Request, Response, status

from mercury_rec import __version__
from mercury_rec.api.deps import CacheDep
from mercury_rec.api.schemas.recommendations import (
    HealthResponse,
    ReadinessCheck,
    ReadinessResponse,
)

router = APIRouter()


@router.get("/health", response_model=HealthResponse, summary="Liveness probe")
def health(request: Request) -> HealthResponse:
    """Report that the process is alive.

    Deliberately checks nothing external.
    """
    started_at = getattr(request.app.state, "started_at", time.time())
    return HealthResponse(
        status="ok",
        version=__version__,
        uptime_seconds=round(time.time() - started_at, 1),
    )


@router.get("/ready", response_model=ReadinessResponse, summary="Readiness probe")
def ready(request: Request, response: Response, cache: CacheDep) -> ReadinessResponse:
    """Report whether this instance can serve, dependency by dependency.

    Returns 503 when not ready so a load balancer removes the instance
    without needing to parse the body.
    """
    checks: list[ReadinessCheck] = []

    engine = getattr(request.app.state, "engine", None)
    startup_error = getattr(request.app.state, "startup_error", None)
    checks.append(
        ReadinessCheck(
            name="engine",
            ready=engine is not None,
            detail=None if engine is not None else (startup_error or "not loaded"),
        )
    )

    if engine is not None:
        bundle = engine.bundle
        checks.append(
            ReadinessCheck(
                name="ranker",
                ready=bundle.ranker is not None,
                detail=None
                if bundle.ranker is not None
                # Not fatal: retrieval-only is a valid deployment, and saying
                # so is more useful than a bare false.
                else "absent - serving retrieval-only",
            )
        )
        checks.append(
            ReadinessCheck(
                name="feature_state",
                ready=True,
                detail=f"{bundle.n_users} users, {bundle.n_items} items",
            )
        )

    cache_ready, cache_detail = cache.ping()
    checks.append(
        ReadinessCheck(
            name="cache",
            ready=cache_ready,
            # A cache outage degrades latency, not correctness, so it does not
            # make the instance unready.
            detail=cache_detail or "connected",
        )
    )

    # Readiness hinges on the engine alone. Requiring every check would take
    # the service out of rotation for a Redis blip it can serve straight
    # through.
    is_ready = engine is not None
    if not is_ready:
        response.status_code = status.HTTP_503_SERVICE_UNAVAILABLE

    return ReadinessResponse(ready=is_ready, checks=checks)


@router.get("/metrics", summary="Prometheus metrics", response_class=Response)
def metrics() -> Response:
    """Expose metrics in the Prometheus text exposition format."""
    from prometheus_client import CONTENT_TYPE_LATEST, generate_latest

    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)


__all__ = ["router"]
