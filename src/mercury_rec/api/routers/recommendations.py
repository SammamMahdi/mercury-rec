"""Recommendation, event-ingestion and model-status endpoints."""

from __future__ import annotations

import time
from datetime import datetime
from typing import Annotated

from fastapi import APIRouter, HTTPException, Path, Query, status

from mercury_rec.api.deps import EngineDep, RequestIdDep, SettingsDep
from mercury_rec.api.schemas.recommendations import (
    EventIngest,
    EventIngestResponse,
    ModelsStatusResponse,
    ModelStatus,
    PipelineTraceResponse,
    Recommendation,
    RecommendationExplanation,
    RecommendationResponse,
    SessionRecommendRequest,
    SimilarItemsResponse,
    StageLatency,
)
from mercury_rec.core.logging import get_logger
from mercury_rec.features.store import FEATURE_SCHEMA_VERSION
from mercury_rec.monitoring.metrics import (
    observe_cache,
    observe_recommendation,
    observe_stage_latency,
)
from mercury_rec.recommender.engine import RecommendationResult, RequestContext

logger = get_logger(__name__)

router = APIRouter()


def _to_response(result: RecommendationResult) -> RecommendationResponse:
    """Map the engine's result onto the wire contract."""
    recommendations = [
        Recommendation(
            item_id=item.item_id,
            rank=rank,
            ml_relevance_score=item.ml_relevance_score,
            business_adjustment=item.business_adjustment,
            final_score=item.final_score,
            adjustments=item.adjustments,
            category_id=item.category_id,
            merchant_id=item.merchant_id,
            vertical=item.vertical,
        )
        for rank, item in enumerate(result.items, start=1)
    ]

    explanations = [
        RecommendationExplanation(
            item_id=recommendations[position].item_id, contributions=contributions
        )
        for position, contributions in enumerate(result.explanations)
        if position < len(recommendations)
    ]

    return RecommendationResponse(
        user_id=result.user_id,
        request_id=result.request_id,
        model_version=result.model_version,
        generated_at=result.generated_at,
        recommendations=recommendations,
        explanations=explanations,
        latency=StageLatency(**result.timings.as_dict()),
        cache_hit=result.cache_hit,
        n_candidates=result.n_candidates,
        candidate_sources=result.candidate_sources,
        filtered=result.filtered,
        is_cold_start=result.is_cold_start,
        data_provenance=result.data_provenance,
    )


def _record(result: RecommendationResult, endpoint: str) -> None:
    """Push the request's telemetry into Prometheus."""
    observe_recommendation(
        endpoint=endpoint,
        model_version=result.model_version,
        cold_start=result.is_cold_start,
        n_candidates=result.n_candidates,
        n_returned=len(result.items),
    )
    observe_cache(hit=result.cache_hit)
    observe_stage_latency(result.timings.as_dict())


@router.get(
    "/recommendations/{user_id}",
    response_model=RecommendationResponse,
    summary="Recommendations for a user",
)
def recommend(
    engine: EngineDep,
    settings: SettingsDep,
    request_id: RequestIdDep,
    user_id: Annotated[str, Path(min_length=1, max_length=64)],
    k: Annotated[int, Query(ge=1, le=100)] = 10,
    hour: Annotated[int | None, Query(ge=0, le=23)] = None,
    weekday: Annotated[int | None, Query(ge=0, le=6)] = None,
    region_id: Annotated[int | None, Query(ge=0)] = None,
    vertical: Annotated[int | None, Query(ge=0, le=5)] = None,
    no_cache: Annotated[bool, Query(description="Bypass the cache.")] = False,
    as_of: Annotated[
        datetime | None,
        Query(description="Evaluate as of this instant (demo environments only)."),
    ] = None,
) -> RecommendationResponse:
    """Return ranked, re-ranked recommendations for one user.

    Context parameters are optional; supplying them changes the result, which
    is what makes contextual personalisation demonstrable rather than claimed.
    """
    as_of_ns: int | None = None
    if as_of is not None:
        if not settings.allow_as_of_override:
            # Lets a caller ask what the system would have said at an
            # arbitrary past time, so it stays off outside demo environments.
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail="as_of override is disabled in this environment.",
            )
        as_of_ns = int(as_of.timestamp() * 1_000_000_000)

    result = engine.recommend(
        user_id,
        k=k,
        context=RequestContext(
            hour=hour,
            weekday=weekday,
            region_id=region_id,
            vertical=vertical,
            as_of_ns=as_of_ns,
        ),
        use_cache=not no_cache,
    )
    _record(result, endpoint="recommendations")
    return _to_response(result)


@router.get(
    "/recommendations/{user_id}/trace",
    response_model=PipelineTraceResponse,
    summary="Stage-by-stage membership for one request",
)
def recommend_trace(
    engine: EngineDep,
    request_id: RequestIdDep,
    user_id: Annotated[str, Path(min_length=1, max_length=64)],
    k: Annotated[int, Query(ge=1, le=100)] = 10,
    hour: Annotated[int | None, Query(ge=0, le=23)] = None,
    weekday: Annotated[int | None, Query(ge=0, le=6)] = None,
    region_id: Annotated[int | None, Query(ge=0)] = None,
    vertical: Annotated[int | None, Query(ge=0, le=5)] = None,
) -> PipelineTraceResponse:
    """Return which items each stage held, for the pipeline and galaxy views.

    Counts alone say 600 candidates became 10 recommendations. Only the
    membership says *which* 600, which is the difference between a diagram of
    a funnel and a picture of one that ran.
    """
    result = engine.recommend(
        user_id,
        k=k,
        context=RequestContext(hour=hour, weekday=weekday, region_id=region_id, vertical=vertical),
        trace=True,
    )
    _record(result, endpoint="trace")

    trace = result.trace
    if trace is None:
        # The cold-start path returns before retrieval runs, so there are no
        # stages to report. An empty trace with the reason stated beats a 404
        # for what is a normal, successful request.
        return PipelineTraceResponse(
            user_id=result.user_id,
            request_id=result.request_id,
            model_version=result.model_version,
            generated_at=result.generated_at,
            final_ids=[item.item_id for item in result.items],
            latency=StageLatency(**result.timings.as_dict()),
            is_cold_start=result.is_cold_start,
            note=(
                "Served by the cold-start path, which answers from contextual "
                "popularity without running retrieval, so there are no stages "
                "to trace."
            ),
        )

    return PipelineTraceResponse(
        user_id=result.user_id,
        request_id=result.request_id,
        model_version=result.model_version,
        generated_at=result.generated_at,
        candidate_ids=trace.candidate_ids,
        candidate_sources=trace.candidate_sources,
        ranked_ids=trace.ranked_ids,
        ranked_scores=trace.ranked_scores,
        final_ids=trace.final_ids,
        latency=StageLatency(**result.timings.as_dict()),
        n_candidates=result.n_candidates,
        filtered=result.filtered,
        is_cold_start=result.is_cold_start,
        note="Uncached: a trace forces a full pipeline run.",
    )


@router.post(
    "/session/recommend",
    response_model=RecommendationResponse,
    summary="Session-context recommendations",
)
def session_recommend(
    engine: EngineDep, request_id: RequestIdDep, payload: SessionRecommendRequest
) -> RecommendationResponse:
    """Recommend from in-session behaviour.

    ``user_id`` may be omitted: an anonymous session is served through the
    cold-start path rather than rejected, because an anonymous visitor is the
    most common request a live system receives.
    """
    result = engine.recommend(
        payload.user_id or "anonymous",
        k=payload.k,
        context=RequestContext(
            hour=payload.context.hour,
            weekday=payload.context.weekday,
            region_id=payload.context.region_id,
            vertical=payload.context.vertical,
            device=payload.context.device,
            session_items=tuple(payload.session_items or payload.context.session_items),
        ),
        # Session context is per-request and effectively unique, so caching it
        # would fill Redis with entries that are never read again.
        use_cache=False,
    )
    _record(result, endpoint="session")
    return _to_response(result)


@router.post("/events", response_model=EventIngestResponse, summary="Ingest behavioural events")
def ingest_events(engine: EngineDep, events: list[EventIngest]) -> EventIngestResponse:
    """Accept events, advance the online feature state, invalidate caches.

    Events are folded in through the same ``apply_event`` the offline scan
    uses, so online state evolves exactly as training would have evolved it.
    Unknown users are rejected rather than silently dropped, and the count is
    returned: a client sending ids the service cannot resolve should find out.
    """
    if len(events) > 1000:
        raise HTTPException(
            status_code=status.HTTP_413_CONTENT_TOO_LARGE,
            detail="At most 1000 events per request.",
        )

    accepted = 0
    rejected = 0
    reasons: dict[str, int] = {}
    invalidated = 0

    for event in events:
        timestamp_ns = (
            int(event.timestamp.timestamp() * 1_000_000_000)
            if event.timestamp is not None
            else time.time_ns()
        )
        keys_removed = engine.observe_event(
            event.user_id,
            event.item_id,
            event.event_type,
            timestamp_ns=timestamp_ns,
            price=event.price or 0.0,
        )
        if engine.bundle.external_user(event.user_id) is None:
            rejected += 1
            reasons["unknown_user"] = reasons.get("unknown_user", 0) + 1
            continue
        accepted += 1
        invalidated += keys_removed

    logger.info("api.events_ingested", accepted=accepted, rejected=rejected)
    return EventIngestResponse(
        accepted=accepted,
        rejected=rejected,
        rejection_reasons=reasons,
        invalidated_cache_keys=invalidated,
    )


@router.get(
    "/items/{item_id}/similar",
    response_model=SimilarItemsResponse,
    summary="Items similar to a given item",
)
def similar_items(
    engine: EngineDep,
    item_id: Annotated[int, Path(ge=0)],
    k: Annotated[int, Query(ge=1, le=50)] = 10,
) -> SimilarItemsResponse:
    """Nearest neighbours by item-item collaborative similarity."""
    started = time.perf_counter()
    bundle = engine.bundle

    if bundle.item_cf is None or not bundle.item_cf.is_fitted:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Item similarity is unavailable: the item-CF model is not loaded.",
        )
    if item_id >= bundle.n_items:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Unknown item {item_id}.",
        )

    neighbours = bundle.item_cf.similar_items(item_id, k=k)
    return SimilarItemsResponse(
        item_id=item_id,
        model_version=bundle.model_version,
        similar=[
            Recommendation(
                item_id=neighbour,
                rank=rank,
                ml_relevance_score=score,
                final_score=score,
                sources=["item_cf"],
            )
            for rank, (neighbour, score) in enumerate(neighbours, start=1)
        ],
        latency_ms=round((time.perf_counter() - started) * 1000.0, 3),
    )


@router.get("/models/status", response_model=ModelsStatusResponse, summary="Loaded model versions")
def models_status(engine: EngineDep) -> ModelsStatusResponse:
    """Report which models are loaded and which dataset produced them."""
    bundle = engine.bundle
    retrieval: list[ModelStatus] = [
        ModelStatus(
            name=model.name,
            version=bundle.model_version,
            dataset_hash=bundle.dataset_hash,
            is_loaded=True,
            params={
                k: v
                for k, v in model.params().items()
                if isinstance(v, str | int | float | bool | type(None))
            },
        )
        for model in (
            bundle.popularity,
            bundle.contextual_popularity,
            bundle.item_cf,
            bundle.matrix_factorization,
            bundle.two_tower,
        )
        if model is not None and model.is_fitted
    ]

    ranking = (
        ModelStatus(
            name="lambdarank",
            version=bundle.model_version,
            dataset_hash=bundle.dataset_hash,
            is_loaded=True,
        )
        if bundle.ranker is not None and bundle.ranker.is_fitted
        else None
    )

    return ModelsStatusResponse(
        retrieval=retrieval,
        ranking=ranking,
        feature_schema_version=FEATURE_SCHEMA_VERSION,
        dataset_hash=bundle.dataset_hash,
    )


__all__ = ["router"]
