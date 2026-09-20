"""FastAPI application: lifespan, middleware, error handling, routers.

Startup loads the artifact bundle once. If that fails the app still starts and
serves ``/health`` while ``/ready`` reports false, because a process that
exits on a missing artifact is much harder to diagnose in a container than one
that stays up and explains itself.

Error responses carry a ``request_id`` and nothing else from the server's
internals -- no stack trace, no SQL, no file path. That detail goes to the
structured log, keyed by the same id, which is where it can be read by someone
entitled to see it.
"""

from __future__ import annotations

import contextlib
import time
import uuid
from collections.abc import AsyncIterator, Awaitable, Callable
from contextlib import asynccontextmanager

from fastapi import FastAPI, Request, status
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse

from mercury_rec import __version__
from mercury_rec.api.routers import embeddings, health, insights, recommendations
from mercury_rec.api.schemas.recommendations import ErrorResponse
from mercury_rec.cache.redis_cache import RecommendationCache, build_client
from mercury_rec.config.settings import Settings, get_settings
from mercury_rec.core.logging import (
    bind_request_context,
    clear_request_context,
    configure_logging,
    get_logger,
)

logger = get_logger(__name__)

#: Header carrying the correlation id, echoed on every response.
REQUEST_ID_HEADER = "X-Request-ID"


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncIterator[None]:
    """Load shared state on startup, release it on shutdown."""
    settings = get_settings()
    configure_logging(
        level="DEBUG" if settings.debug else "INFO",
        json_output=settings.environment != "local",
    )
    app.state.started_at = time.time()
    app.state.settings = settings

    client = build_client(str(settings.redis_dsn) if settings.redis_dsn else None)
    app.state.cache = RecommendationCache(
        client=client,
        ttl_seconds=settings.cache_ttl_seconds,
        soft_ttl_seconds=settings.cache_soft_ttl_seconds,
        jitter_ratio=settings.cache_jitter_ratio,
    )

    app.state.engine = None
    app.state.startup_error = None

    if not settings.load_artifacts_on_startup:
        # A caller is providing the bundle. Say so rather than leaving
        # /ready to report an ambiguous absence.
        app.state.startup_error = "Artifact loading is disabled by configuration."
        logger.info("api.artifact_loading_disabled")
        yield
        if client is not None:
            with contextlib.suppress(Exception):
                client.close()
        return

    try:
        from mercury_rec.recommender.engine import RecommendationEngine
        from mercury_rec.services.artifacts import load_bundle

        bundle = load_bundle()
        app.state.engine = RecommendationEngine(bundle, cache=app.state.cache)
        logger.info("api.ready", model_version=bundle.model_version)
    except Exception as exc:  # noqa: BLE001
        # Stay up and report the reason. A container that crash-loops on a
        # missing artifact tells an operator far less than one that answers
        # /ready with the actual cause.
        app.state.startup_error = str(exc)
        logger.error("api.startup_failed", error=str(exc)[:400])

    yield

    if client is not None:
        # A failure closing the client must not mask the real shutdown reason.
        with contextlib.suppress(Exception):
            client.close()
    logger.info("api.shutdown")


def create_app(settings: Settings | None = None) -> FastAPI:
    """Build the application. A factory so tests can inject configuration."""
    config = settings or get_settings()

    app = FastAPI(
        title="MercuryRec",
        version=__version__,
        description=(
            "Multi-stage recommendation and personalization API.\n\n"
            "**Data provenance:** behavioural events come from the Retailrocket "
            "dataset (CC BY-NC-SA 4.0). Merchants, regions, prices and verticals "
            "are synthesised - see `docs/data-card.md`. Nothing here is real "
            "production traffic."
        ),
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_url="/openapi.json",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=config.cors_origins,
        allow_credentials=False,
        allow_methods=["GET", "POST"],
        allow_headers=["Content-Type", REQUEST_ID_HEADER],
    )

    @app.middleware("http")
    async def request_context(
        request: Request, call_next: Callable[[Request], Awaitable[JSONResponse]]
    ) -> JSONResponse:
        """Assign a correlation id, bind it to the logs, time the request.

        An inbound ``X-Request-ID`` is honoured so a trace can be followed
        across services rather than restarting at this hop.
        """
        request_id = request.headers.get(REQUEST_ID_HEADER) or str(uuid.uuid4())
        request.state.request_id = request_id
        bind_request_context(request_id=request_id, path=request.url.path)

        started = time.perf_counter()
        try:
            response = await call_next(request)
        finally:
            elapsed_ms = (time.perf_counter() - started) * 1000.0
            logger.info(
                "api.request",
                method=request.method,
                path=request.url.path,
                duration_ms=round(elapsed_ms, 2),
            )
            clear_request_context()

        response.headers[REQUEST_ID_HEADER] = request_id
        return response

    @app.middleware("http")
    async def limit_body_size(
        request: Request, call_next: Callable[[Request], Awaitable[JSONResponse]]
    ) -> JSONResponse:
        """Reject oversized bodies before they are read into memory."""
        declared = request.headers.get("content-length")
        if declared is not None and declared.isdigit() and int(declared) > config.max_request_bytes:
            return JSONResponse(
                status_code=status.HTTP_413_CONTENT_TOO_LARGE,
                content=ErrorResponse(
                    error="request_too_large",
                    detail=f"Body exceeds {config.max_request_bytes} bytes.",
                    request_id=getattr(request.state, "request_id", None),
                ).model_dump(),
            )
        return await call_next(request)

    @app.exception_handler(Exception)
    async def unhandled_exception(request: Request, exc: Exception) -> JSONResponse:
        """Log the detail, return only a correlation id.

        Leaking a traceback to a caller is an information-disclosure problem
        and tells them nothing useful anyway; the id lets an operator find the
        full context in the logs.
        """
        request_id = getattr(request.state, "request_id", None)
        logger.exception("api.unhandled_error", path=request.url.path, request_id=request_id)
        return JSONResponse(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            content=ErrorResponse(
                error="internal_error",
                detail="An unexpected error occurred. Quote the request id when reporting it.",
                request_id=request_id,
            ).model_dump(),
        )

    app.include_router(health.router, tags=["health"])
    app.include_router(recommendations.router, prefix="/api/v1", tags=["recommendations"])
    app.include_router(insights.router, prefix="/api/v1/insights", tags=["insights"])
    app.include_router(embeddings.router, prefix="/api/v1/embeddings", tags=["embeddings"])

    return app


app = create_app()

__all__ = ["REQUEST_ID_HEADER", "app", "create_app", "lifespan"]
