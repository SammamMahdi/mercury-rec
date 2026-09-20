"""Request and response models for the recommendation API.

Pydantic here rather than Pandera: these are per-object contracts at a request
boundary where the input is untrusted, which is exactly what Pydantic is for.
The columnar frames inside the pipeline use Pandera instead.

Three properties of these models are deliberate:

1. **Relevance and business policy stay separate.** Every recommendation
   carries ``ml_relevance_score``, ``business_adjustment`` and ``final_score``
   as distinct fields. A client can always tell which part of a ranking came
   from the model and which from commercial policy.
2. **Per-stage latency is part of the response**, not just a server-side
   metric. A caller investigating a slow request should not need access to
   the monitoring stack to see which stage cost the time.
3. **Provenance is explicit.** Each item reports which retrieval sources
   proposed it, and the response reports the model version that produced it.
"""

from __future__ import annotations

from datetime import datetime
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field

from mercury_rec.core.enums import DataSource


class ResponseModel(BaseModel):
    """Base for every model this API returns.

    ``json_schema_serialization_defaults_required`` makes a field that has a
    default REQUIRED in the published schema. For a response that is simply
    the truth: FastAPI serialises every field, so a client always receives
    them. A schema that calls them optional forces every consumer to handle an
    ``undefined`` that cannot arrive, and the frontend's generated types then
    disagree with the hand-written ones for no real reason.

    Request models deliberately do NOT inherit this. There a default means
    what it says - the caller may omit the field.

    Frozen because a response is a snapshot. Mutating one after it is built
    means something was decided too late.
    """

    model_config = ConfigDict(frozen=True, json_schema_serialization_defaults_required=True)


class StageLatency(ResponseModel):
    """Milliseconds spent in each stage of the request.

    Returned on every response so a slow request is diagnosable from the
    response alone. The fields mirror the pipeline stages exactly, which is
    what lets the frontend's Pipeline Inspector show real numbers rather than
    a drawing.
    """

    cache_lookup_ms: float = 0.0
    feature_lookup_ms: float = 0.0
    candidate_generation_ms: float = 0.0
    ranking_ms: float = 0.0
    reranking_ms: float = 0.0
    total_ms: float = 0.0


class Recommendation(ResponseModel):
    """One recommended item."""

    item_id: int
    rank: Annotated[int, Field(ge=1, description="1-based position in the list")]

    ml_relevance_score: float = Field(
        description=(
            "The ranking model's output. An ordinal utility from LambdaRank, "
            "NOT a calibrated probability - do not read it as one."
        )
    )
    business_adjustment: float = Field(
        default=0.0,
        description=(
            "Sum of deterministic policy adjustments. Kept separate from the "
            "model score so any ranking change is attributable."
        ),
    )
    final_score: float = Field(description="ml_relevance_score + business_adjustment")

    adjustments: dict[str, float] = Field(
        default_factory=dict,
        description="Per-rule breakdown of business_adjustment.",
    )
    sources: list[str] = Field(
        default_factory=list,
        description="Retrieval sources that proposed this item.",
    )

    category_id: int | None = None
    merchant_id: int | None = None
    vertical: int | None = None
    price: float | None = None


class RecommendationExplanation(ResponseModel):
    """Why one item was ranked where it was.

    Values are exact TreeSHAP contributions from the ranking model, signed:
    negative means the feature pushed this item *down*. They are the model's
    real attributions for this specific score, not a narrative written after
    the fact.
    """

    item_id: int
    contributions: dict[str, float]


class RequestContext(BaseModel):
    """Context a caller may supply to shape the recommendation."""

    hour: Annotated[int | None, Field(default=None, ge=0, le=23)] = None
    weekday: Annotated[int | None, Field(default=None, ge=0, le=6)] = None
    region_id: Annotated[int | None, Field(default=None, ge=0)] = None
    vertical: Annotated[int | None, Field(default=None, ge=0, le=5)] = None
    session_items: list[int] = Field(
        default_factory=list, max_length=100, description="Items seen this session."
    )
    device: Annotated[int | None, Field(default=None, ge=0, le=3)] = None


class RecommendationResponse(ResponseModel):
    """The response returned by the recommendation endpoints."""

    user_id: str
    request_id: str
    model_version: str
    generated_at: datetime

    recommendations: list[Recommendation]
    explanations: list[RecommendationExplanation] = Field(default_factory=list)

    latency: StageLatency
    cache_hit: bool = False

    n_candidates: int = Field(default=0, description="Candidates that reached the ranking stage.")
    candidate_sources: dict[str, int] = Field(
        default_factory=dict, description="Candidate count contributed per source."
    )
    filtered: dict[str, int] = Field(
        default_factory=dict,
        description=(
            "Items removed by hard filters, per reason. Surfaced because "
            "silent filtering degrades results with no visible cause."
        ),
    )

    is_cold_start: bool = Field(
        default=False,
        description="True when the user had no usable history and the cold-start path served this.",
    )
    data_provenance: DataSource = Field(
        default=DataSource.AUGMENTED,
        description=(
            "Provenance of the underlying dataset. Behaviour is real "
            "Retailrocket; merchants, regions, prices and verticals are "
            "synthesised. See docs/data-card.md."
        ),
    )


class SessionRecommendRequest(BaseModel):
    """Body for session-based recommendation."""

    user_id: str | None = Field(
        default=None, description="Omit for an anonymous session; cold-start serves it."
    )
    session_items: list[int] = Field(default_factory=list, max_length=100)
    context: RequestContext = RequestContext()
    k: Annotated[int, Field(default=10, ge=1, le=100)] = 10


class EventIngest(BaseModel):
    """One behavioural event submitted to the API.

    Accepted events advance the online feature state through the same
    ``apply_event`` the offline scan uses, so serving stays consistent with
    training rather than drifting between them.
    """

    user_id: str
    item_id: int = Field(ge=0)
    event_type: Annotated[int, Field(ge=0, le=10, description="EventType value.")]
    timestamp: datetime | None = Field(default=None, description="Defaults to server receive time.")
    session_id: str | None = None
    price: float | None = Field(default=None, ge=0)


class EventIngestResponse(ResponseModel):
    accepted: int
    rejected: int = 0
    rejection_reasons: dict[str, int] = Field(default_factory=dict)
    invalidated_cache_keys: int = 0


class SimilarItemsResponse(ResponseModel):
    item_id: int
    model_version: str
    similar: list[Recommendation]
    latency_ms: float


class ModelStatus(ResponseModel):
    """One loaded model's identity and provenance."""

    model_config = ConfigDict(
        frozen=True,
        json_schema_serialization_defaults_required=True,
        # 'model_version' would otherwise collide with Pydantic's own
        # protected 'model_' prefix.
        protected_namespaces=(),
    )

    name: str
    version: str
    trained_at: datetime | None = None
    dataset_hash: str | None = None
    is_loaded: bool = True
    params: dict[str, float | int | str | bool | None] = Field(default_factory=dict)


class ModelsStatusResponse(ResponseModel):
    retrieval: list[ModelStatus]
    ranking: ModelStatus | None = None
    feature_schema_version: int
    dataset_hash: str | None = None


class HealthResponse(ResponseModel):
    status: str
    version: str
    uptime_seconds: float


class ReadinessCheck(ResponseModel):
    name: str
    ready: bool
    detail: str | None = None


class ReadinessResponse(ResponseModel):
    """Readiness, with per-dependency detail.

    A bare boolean is not actionable during an incident: what matters is
    *which* dependency is down, so each is reported separately.
    """

    ready: bool
    checks: list[ReadinessCheck]


class ErrorResponse(ResponseModel):
    """A structured error.

    Deliberately carries no internal detail - no stack trace, no SQL, no file
    path. ``request_id`` is the handle for correlating with the server logs,
    which is where that detail belongs.
    """

    error: str
    detail: str
    request_id: str | None = None


class PipelineTraceResponse(ResponseModel):
    """Which items each stage of the pipeline held, for one request.

    A diagnostic, served from its own endpoint rather than bolted onto every
    recommendation response: it is two orders of magnitude larger than the
    result it explains, and the serving path has no use for it.

    Requesting a trace forces a full pipeline run and bypasses the cache,
    because a cached response has no stage membership to report and
    reconstructing one from the final list would describe a funnel that never
    ran.
    """

    user_id: str
    request_id: str
    model_version: str
    generated_at: datetime

    candidate_ids: list[int] = Field(
        default_factory=list, description="Every item that survived retrieval and fusion."
    )
    candidate_sources: dict[str, list[int]] = Field(
        default_factory=dict,
        description=(
            "Item ids proposed by each source. An item appears under every "
            "source that returned it, so the overlap between sources is visible."
        ),
    )
    ranked_ids: list[int] = Field(
        default_factory=list, description="Candidates in ranker order, best first."
    )
    ranked_scores: list[float] = Field(
        default_factory=list,
        description="Ranker output per entry in ranked_ids. An ordinal utility, not a probability.",
    )
    final_ids: list[int] = Field(
        default_factory=list, description="What the business rerank actually returned."
    )

    latency: StageLatency
    n_candidates: int = 0
    filtered: dict[str, int] = Field(default_factory=dict)
    is_cold_start: bool = False
    note: str = ""


__all__ = [
    "ErrorResponse",
    "EventIngest",
    "EventIngestResponse",
    "HealthResponse",
    "ModelStatus",
    "ModelsStatusResponse",
    "PipelineTraceResponse",
    "ReadinessCheck",
    "ReadinessResponse",
    "Recommendation",
    "RecommendationExplanation",
    "RecommendationResponse",
    "RequestContext",
    "ResponseModel",
    "SessionRecommendRequest",
    "SimilarItemsResponse",
    "StageLatency",
]
