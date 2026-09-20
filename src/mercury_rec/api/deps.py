"""Dependency wiring for the API.

Heavy objects -- the artifact bundle, the feature state, the Redis client --
are built once during application startup and shared. Building them per
request would load hundreds of megabytes of model state on every call.

They live on ``app.state`` rather than in module globals so tests can
construct an app with fakes injected, and so two apps can coexist in one
process without fighting over shared state.
"""

from __future__ import annotations

from typing import Annotated

from fastapi import Depends, HTTPException, Request, status

from mercury_rec.cache.redis_cache import RecommendationCache
from mercury_rec.config.settings import Settings, get_settings
from mercury_rec.recommender.engine import RecommendationEngine


def get_app_settings() -> Settings:
    return get_settings()


def get_engine(request: Request) -> RecommendationEngine:
    """Return the shared engine, or 503 if startup never completed.

    503 rather than 500: the service is temporarily unable to serve, which is
    a different thing from a bug, and load balancers treat the two
    differently.
    """
    engine: RecommendationEngine | None = getattr(request.app.state, "engine", None)
    if engine is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=(
                "Recommendation engine is not loaded. The service is starting, "
                "or required artifacts are missing."
            ),
        )
    return engine


def get_cache(request: Request) -> RecommendationCache:
    cache: RecommendationCache | None = getattr(request.app.state, "cache", None)
    return cache or RecommendationCache()


def get_request_id(request: Request) -> str:
    """The correlation id assigned by the middleware."""
    return str(getattr(request.state, "request_id", "unknown"))


EngineDep = Annotated[RecommendationEngine, Depends(get_engine)]
CacheDep = Annotated[RecommendationCache, Depends(get_cache)]
SettingsDep = Annotated[Settings, Depends(get_app_settings)]
RequestIdDep = Annotated[str, Depends(get_request_id)]


__all__ = [
    "CacheDep",
    "EngineDep",
    "RequestIdDep",
    "SettingsDep",
    "get_app_settings",
    "get_cache",
    "get_engine",
    "get_request_id",
]
