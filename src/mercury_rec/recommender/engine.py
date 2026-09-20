"""The staged recommendation engine.

One class owns the request path, and it is the only place the stage order
lives::

    cache -> context -> features -> retrieval -> fusion -> ranking
          -> business rerank -> cache write -> response

Every stage is timed individually. The timings go into the response body, into
Prometheus histograms, and into the request log, because "the request took
180ms" is not actionable while "ranking took 174ms of it" is.

Cold start is a first-class path, not an error case. A user with no usable
history cannot be served by collaborative retrieval at all: item-CF returns
zeros and the two-tower has no meaningful id embedding. Rather than returning
an empty list or a degenerate ranking, the engine routes to contextual
popularity and marks the response ``is_cold_start`` so the caller knows which
path served it.

The engine holds no framework types. It knows nothing about FastAPI, HTTP or
Redis clients, which is what lets the same code run inside the API process, a
batch job and the tests.
"""

from __future__ import annotations

import time
import uuid
from collections.abc import Iterator
from contextlib import contextmanager
from dataclasses import dataclass, field
from datetime import UTC, datetime
from typing import Any

import numpy as np
import pandas as pd

from mercury_rec.cache import keys
from mercury_rec.cache.redis_cache import RecommendationCache
from mercury_rec.core.enums import DataSource, EventType, RetrievalSource
from mercury_rec.core.logging import get_logger
from mercury_rec.features.asof import EventRow
from mercury_rec.features.store import AsOfFeatureStore
from mercury_rec.models.item_cf import ItemCFRecommender
from mercury_rec.models.mf import BPRRecommender
from mercury_rec.models.popularity import (
    ContextualPopularityRecommender,
    PopularityRecommender,
    hour_bucket,
)
from mercury_rec.models.ranking.dataset import RANKING_FEATURES
from mercury_rec.models.ranking.ranker import LambdaRanker
from mercury_rec.models.two_tower.recommender import TwoTowerRecommender
from mercury_rec.reranking.pipeline import RerankConfig, ScoredItem, rerank
from mercury_rec.retrieval.candidates import reciprocal_rank_fusion, retrieve_from_scores

logger = get_logger(__name__)


@dataclass(slots=True)
class StageTimings:
    """Accumulated per-stage durations, in milliseconds."""

    cache_lookup_ms: float = 0.0
    feature_lookup_ms: float = 0.0
    candidate_generation_ms: float = 0.0
    ranking_ms: float = 0.0
    reranking_ms: float = 0.0
    total_ms: float = 0.0

    def as_dict(self) -> dict[str, float]:
        return {
            "cache_lookup_ms": round(self.cache_lookup_ms, 3),
            "feature_lookup_ms": round(self.feature_lookup_ms, 3),
            "candidate_generation_ms": round(self.candidate_generation_ms, 3),
            "ranking_ms": round(self.ranking_ms, 3),
            "reranking_ms": round(self.reranking_ms, 3),
            "total_ms": round(self.total_ms, 3),
        }


@dataclass(slots=True)
class RequestContext:
    """Context for one recommendation request."""

    hour: int | None = None
    weekday: int | None = None
    region_id: int | None = None
    vertical: int | None = None
    device: int | None = None
    session_items: tuple[int, ...] = ()
    as_of_ns: int | None = None
    """Evaluate as of this instant instead of now. Powers the frontend's
    journey replay; gated off outside local and demo environments because it
    lets a caller ask what the system would have said at an arbitrary past
    time."""

    def fingerprint(self) -> str:
        return keys.context_fingerprint(
            hour_bucket=hour_bucket(self.hour) if self.hour is not None else None,
            weekday=self.weekday,
            region_id=self.region_id,
            vertical=self.vertical,
            device=self.device,
            session_signature=(
                ",".join(str(i) for i in sorted(self.session_items)[:10])
                if self.session_items
                else None
            ),
        )


@dataclass(slots=True)
class ArtifactBundle:
    """Everything the engine needs, loaded once at startup.

    Held together so a model promotion swaps one object atomically. Swapping
    models individually risks serving a ranker against embeddings from a
    different training run, which produces plausible nonsense rather than an
    error.
    """

    model_version: str
    dataset_hash: str | None
    n_users: int
    n_items: int

    feature_store: AsOfFeatureStore
    popularity: PopularityRecommender
    contextual_popularity: ContextualPopularityRecommender | None = None
    item_cf: ItemCFRecommender | None = None
    matrix_factorization: BPRRecommender | None = None
    two_tower: TwoTowerRecommender | None = None
    ranker: LambdaRanker | None = None

    items: pd.DataFrame | None = None
    user_index: dict[str, int] = field(default_factory=dict)
    item_index: dict[int, int] = field(default_factory=dict)
    user_history: dict[int, set[int]] = field(default_factory=dict)
    popularity_rank: np.ndarray | None = None

    def external_user(self, user_id: str) -> int | None:
        """Map an external user id to its dense internal index."""
        return self.user_index.get(user_id)


@dataclass(slots=True)
class RecommendationResult:
    """The engine's output, before HTTP serialisation."""

    user_id: str
    request_id: str
    model_version: str
    generated_at: datetime
    items: list[ScoredItem]
    timings: StageTimings
    cache_hit: bool = False
    is_cold_start: bool = False
    n_candidates: int = 0
    candidate_sources: dict[str, int] = field(default_factory=dict)
    filtered: dict[str, int] = field(default_factory=dict)
    explanations: list[dict[str, float]] = field(default_factory=list)
    data_provenance: DataSource = DataSource.AUGMENTED


class RecommendationEngine:
    """Orchestrates the staged recommendation pipeline."""

    def __init__(
        self,
        bundle: ArtifactBundle,
        *,
        cache: RecommendationCache | None = None,
        rerank_config: RerankConfig | None = None,
        per_source_k: int = 200,
        max_candidates: int = 600,
        explain_top_n: int = 3,
    ) -> None:
        self.bundle = bundle
        self.cache = cache or RecommendationCache()
        self.rerank_config = rerank_config or RerankConfig()
        self.per_source_k = per_source_k
        self.max_candidates = max_candidates
        self.explain_top_n = explain_top_n

    @contextmanager
    def _timed(self, timings: StageTimings, field_name: str) -> Iterator[None]:
        started = time.perf_counter()
        try:
            yield
        finally:
            elapsed = (time.perf_counter() - started) * 1000.0
            setattr(timings, field_name, getattr(timings, field_name) + elapsed)

    # --- retrieval --------------------------------------------------------

    def _retrieve(self, internal_user: int, context: RequestContext, exclude: set[int]) -> Any:
        """Query every available source and fuse the results."""
        bundle = self.bundle
        results = []
        catalogue = np.arange(bundle.n_items)

        source_model = bundle.contextual_popularity or bundle.popularity
        started = time.perf_counter()
        from mercury_rec.models.base import RecommendationContext as ModelContext

        model_context = ModelContext(
            hour=context.hour,
            weekday=context.weekday,
            region_id=context.region_id,
            vertical=context.vertical,
            session_items=context.session_items,
        )
        popularity_scores = source_model._score(internal_user, catalogue, model_context)
        results.append(
            retrieve_from_scores(
                popularity_scores,
                source=RetrievalSource.POPULARITY,
                k=self.per_source_k,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        )

        if bundle.item_cf is not None:
            started = time.perf_counter()
            results.append(
                retrieve_from_scores(
                    bundle.item_cf._score(internal_user, catalogue, model_context),
                    source=RetrievalSource.ITEM_CF,
                    k=self.per_source_k,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
            )

        if bundle.matrix_factorization is not None:
            started = time.perf_counter()
            results.append(
                retrieve_from_scores(
                    bundle.matrix_factorization._score(internal_user, catalogue, model_context),
                    source=RetrievalSource.MATRIX_FACTORIZATION,
                    k=self.per_source_k,
                    elapsed_ms=(time.perf_counter() - started) * 1000.0,
                )
            )

        if bundle.two_tower is not None:
            started = time.perf_counter()
            try:
                scores = bundle.two_tower._score(internal_user, catalogue, model_context)
                results.append(
                    retrieve_from_scores(
                        scores,
                        source=RetrievalSource.TWO_TOWER,
                        k=self.per_source_k,
                        elapsed_ms=(time.perf_counter() - started) * 1000.0,
                    )
                )
            except RuntimeError:
                # No cached user embedding for this user. The other sources
                # still serve the request; dropping one source degrades
                # quality rather than failing the call.
                logger.debug("engine.two_tower_unavailable", user=internal_user)

        return reciprocal_rank_fusion(results, max_candidates=self.max_candidates, exclude=exclude)

    # --- ranking ----------------------------------------------------------

    def _rank(
        self, internal_user: int, candidates: Any, context: RequestContext
    ) -> tuple[np.ndarray, list[dict[str, float]]]:
        """Score candidates with the ranker, falling back to fusion order."""
        bundle = self.bundle
        if bundle.ranker is None or not bundle.ranker.is_fitted:
            # Without a ranker the fused RRF score IS the ranking. Returning
            # it rather than failing keeps retrieval-only a valid deployment.
            return candidates.fused_scores, []

        now_ns = context.as_of_ns or time.time_ns()
        feature_matrix = bundle.feature_store.compute_batch(
            internal_user, candidates.item_ids, now_ns
        )

        n_rows = len(candidates.item_ids)
        extra = np.full((n_rows, len(RANKING_FEATURES) - feature_matrix.shape[1]), np.nan)
        for position, source in enumerate(
            ("two_tower", "item_cf", "popularity", "matrix_factorization")
        ):
            scores = candidates.per_source_scores.get(source)
            if scores is not None:
                extra[:, position] = scores
        extra[:, -2] = np.arange(1, n_rows + 1)
        extra[:, -1] = candidates.sources_per_item

        full = np.hstack([feature_matrix, extra]).astype(np.float32)
        scores = bundle.ranker.score(full)

        explanations: list[dict[str, float]] = []
        if self.explain_top_n > 0:
            top = np.argsort(-scores)[: self.explain_top_n]
            try:
                explanations = bundle.ranker.explain(full[top], top_n=5)
            except Exception as exc:  # noqa: BLE001
                # Explanations are a nice-to-have; never fail a recommendation
                # because SHAP could not run.
                logger.warning("engine.explain_failed", error=str(exc)[:120])

        return scores, explanations

    # --- public API -------------------------------------------------------

    def recommend(
        self,
        user_id: str,
        *,
        k: int = 10,
        context: RequestContext | None = None,
        use_cache: bool = True,
    ) -> RecommendationResult:
        """Produce recommendations for one user."""
        started = time.perf_counter()
        timings = StageTimings()
        request_id = str(uuid.uuid4())
        ctx = context or RequestContext()
        bundle = self.bundle

        cache_key = keys.recommendations(bundle.model_version, user_id, ctx.fingerprint(), k)

        if use_cache:
            with self._timed(timings, "cache_lookup_ms"):
                cached = self.cache.get(cache_key)
            if cached is not None:
                timings.total_ms = (time.perf_counter() - started) * 1000.0
                return _from_cached(cached.value, request_id, timings, bundle)

        internal_user = bundle.external_user(user_id)
        is_cold_start = internal_user is None or not bundle.user_history.get(internal_user)

        if internal_user is None:
            # Unknown user: serve contextual popularity. Returning an empty
            # list would be a worse answer than a good generic one.
            items = self._cold_start(ctx, k)
            timings.total_ms = (time.perf_counter() - started) * 1000.0
            return RecommendationResult(
                user_id=user_id,
                request_id=request_id,
                model_version=bundle.model_version,
                generated_at=datetime.now(UTC),
                items=items,
                timings=timings,
                is_cold_start=True,
            )

        exclude = bundle.user_history.get(internal_user, set())

        with self._timed(timings, "candidate_generation_ms"):
            candidates = self._retrieve(internal_user, ctx, exclude)

        if len(candidates) == 0:
            items = self._cold_start(ctx, k)
            timings.total_ms = (time.perf_counter() - started) * 1000.0
            return RecommendationResult(
                user_id=user_id,
                request_id=request_id,
                model_version=bundle.model_version,
                generated_at=datetime.now(UTC),
                items=items,
                timings=timings,
                is_cold_start=True,
            )

        with self._timed(timings, "ranking_ms"):
            scores, explanations = self._rank(internal_user, candidates, ctx)

        with self._timed(timings, "reranking_ms"):
            scored = self._build_scored_items(candidates, scores)
            rerank_result = rerank(
                scored,
                k=k,
                config=self.rerank_config,
                popularity_rank=bundle.popularity_rank,
            )

        timings.total_ms = (time.perf_counter() - started) * 1000.0

        result = RecommendationResult(
            user_id=user_id,
            request_id=request_id,
            model_version=bundle.model_version,
            generated_at=datetime.now(UTC),
            items=rerank_result.items,
            timings=timings,
            is_cold_start=is_cold_start,
            n_candidates=len(candidates),
            candidate_sources=candidates.source_counts,
            filtered=rerank_result.filtered_counts,
            explanations=explanations,
        )

        if use_cache:
            self.cache.set(cache_key, _to_cacheable(result), user_id=user_id)

        return result

    def _build_scored_items(self, candidates: Any, scores: np.ndarray) -> list[ScoredItem]:
        """Attach catalogue attributes needed by the business rules."""
        bundle = self.bundle
        items = bundle.items
        scored: list[ScoredItem] = []

        for position, item_id in enumerate(candidates.item_ids.tolist()):
            attributes: dict[str, Any] = {}
            if items is not None and item_id < len(items):
                row = items.iloc[item_id]
                attributes = {
                    "merchant_id": int(row["merchant_id"]),
                    "category_id": int(row["category_id"]),
                    "vertical": int(row["vertical"]),
                    "is_available": bool(row.get("is_available", True)),
                }
            scored.append(
                ScoredItem(
                    item_id=int(item_id),
                    ml_relevance_score=float(scores[position]),
                    **attributes,
                )
            )
        return scored

    def _cold_start(self, context: RequestContext, k: int) -> list[ScoredItem]:
        """Contextual popularity for users with no usable history.

        Deliberately a real path rather than an error: a new user is the most
        common case a live system faces, and "no recommendations" is never the
        right answer to it.
        """
        bundle = self.bundle
        model = bundle.contextual_popularity or bundle.popularity
        from mercury_rec.models.base import RecommendationContext as ModelContext

        top = model.recommend(
            0,
            k=k,
            context=ModelContext(
                hour=context.hour,
                weekday=context.weekday,
                region_id=context.region_id,
                vertical=context.vertical,
            ),
        )
        scores = model.item_scores
        return [
            ScoredItem(item_id=item_id, ml_relevance_score=float(scores[item_id]))
            for item_id in top
        ]

    def observe_event(
        self,
        user_id: str,
        item_id: int,
        event_type: int,
        *,
        timestamp_ns: int | None = None,
        price: float = 0.0,
    ) -> int:
        """Fold an incoming event into the online feature state.

        Uses the same ``apply_event`` as the offline scan, so online state
        evolves exactly as training would have evolved it. Then invalidates
        that user's cached recommendations, because they were computed before
        this event existed.
        """
        internal_user = self.bundle.external_user(user_id)
        if internal_user is None:
            return 0

        self.bundle.feature_store.observe(
            EventRow(
                user=internal_user,
                item=item_id,
                now=timestamp_ns or time.time_ns(),
                event_type=event_type,
                price=price,
            )
        )
        if event_type != int(EventType.IMPRESSION):
            self.bundle.user_history.setdefault(internal_user, set()).add(item_id)

        return self.cache.invalidate_user(user_id)


def _to_cacheable(result: RecommendationResult) -> dict[str, Any]:
    return {
        "model_version": result.model_version,
        "generated_at": result.generated_at.isoformat(),
        "is_cold_start": result.is_cold_start,
        "n_candidates": result.n_candidates,
        "candidate_sources": result.candidate_sources,
        "filtered": result.filtered,
        "items": [item.as_dict(rank) for rank, item in enumerate(result.items, start=1)],
        "explanations": result.explanations,
    }


def _from_cached(
    payload: dict[str, Any],
    request_id: str,
    timings: StageTimings,
    bundle: ArtifactBundle,
) -> RecommendationResult:
    items = [
        ScoredItem(
            item_id=int(entry["item_id"]),
            ml_relevance_score=float(entry["ml_relevance_score"]),
            business_adjustment=float(entry.get("business_adjustment", 0.0)),
            adjustments=dict(entry.get("adjustments", {})),
        )
        for entry in payload.get("items", [])
    ]
    return RecommendationResult(
        user_id=payload.get("user_id", ""),
        request_id=request_id,
        model_version=payload.get("model_version", bundle.model_version),
        generated_at=datetime.fromisoformat(
            payload.get("generated_at", datetime.now(UTC).isoformat())
        ),
        items=items,
        timings=timings,
        cache_hit=True,
        is_cold_start=bool(payload.get("is_cold_start", False)),
        n_candidates=int(payload.get("n_candidates", 0)),
        candidate_sources=dict(payload.get("candidate_sources", {})),
        filtered=dict(payload.get("filtered", {})),
        explanations=list(payload.get("explanations", [])),
    )


__all__ = [
    "ArtifactBundle",
    "RecommendationEngine",
    "RecommendationResult",
    "RequestContext",
    "StageTimings",
]
