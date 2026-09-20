"""SQLAlchemy 2.0 ORM models.

Two roles are deliberately separated here.

**Dimension tables** (users, items, merchants) are the catalogue the API reads
to enrich a recommendation. They are small, change slowly, and benefit from
foreign keys and constraints.

**Log tables** (recommendation_requests, recommendation_results,
interactions) are append-only and high volume. They carry no foreign keys:
the write path is bulk ``COPY`` of hundreds of thousands of rows, and FK
validation on that is the dominant cost for an integrity guarantee the loader
already provides. This is a considered trade rather than an omission, and it
is recorded in ``docs/data-model.md``.

Two constraints are worth pointing at because they turn application
assumptions into database invariants:

- A partial unique index guarantees **at most one production model per type**.
  Expressed in application code that is a hope; expressed here it is enforced
  even against a manual UPDATE.
- ``experiments.is_simulated`` is ``NOT NULL DEFAULT true``. The project's
  honesty requirement about offline experimentation is encoded in the schema,
  so a row claiming a live experiment has to say so explicitly.
"""

from __future__ import annotations

from datetime import datetime
from typing import Any

from sqlalchemy import (
    BigInteger,
    Boolean,
    CheckConstraint,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    SmallInteger,
    String,
    Text,
    UniqueConstraint,
    func,
)
from sqlalchemy.dialects.postgresql import ARRAY, JSONB
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column, relationship


class Base(DeclarativeBase):
    """Declarative base with a shared type map."""

    type_annotation_map = {  # noqa: RUF012
        dict[str, Any]: JSONB,
        datetime: DateTime(timezone=True),
    }


class Merchant(Base):
    """A merchant. Synthesised - Retailrocket has no merchant concept."""

    __tablename__ = "merchants"

    merchant_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    vertical: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    region_id: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    rating: Mapped[float] = mapped_column(Float, nullable=False)
    delivery_radius_km: Mapped[float] = mapped_column(Float, nullable=False)
    avg_delivery_minutes: Mapped[int] = mapped_column(SmallInteger, nullable=False)

    items: Mapped[list[Item]] = relationship(back_populates="merchant")

    __table_args__ = (
        Index("ix_merchants_region_vertical", "region_id", "vertical"),
        CheckConstraint("rating >= 0 AND rating <= 5", name="ck_merchant_rating_range"),
    )


class Item(Base):
    """An item. Ids and categories are real; the rest is synthesised."""

    __tablename__ = "items"

    item_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_item_id: Mapped[int] = mapped_column(BigInteger, nullable=False, unique=True)
    """The original Retailrocket id, kept so results stay traceable to source."""

    merchant_id: Mapped[int] = mapped_column(ForeignKey("merchants.merchant_id"), nullable=False)
    category_id: Mapped[int | None] = mapped_column(Integer)
    vertical: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    price: Mapped[float] = mapped_column(Float, nullable=False)
    base_rating: Mapped[float | None] = mapped_column(Float)
    available_from: Mapped[datetime] = mapped_column(nullable=False)
    available_until: Mapped[datetime | None] = mapped_column()
    is_cold: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    merchant: Mapped[Merchant] = relationship(back_populates="items")

    __table_args__ = (
        Index("ix_items_category", "category_id"),
        Index("ix_items_vertical_category", "vertical", "category_id"),
        # Partial index over currently-listed items only. The availability
        # filter runs on every request, and most of the catalogue is delisted,
        # so indexing the whole table would be several times larger for no gain.
        Index(
            "ix_items_currently_listed",
            "item_id",
            postgresql_where=(available_until.is_(None)),
        ),
        CheckConstraint("price > 0", name="ck_item_price_positive"),
    )


class User(Base):
    """A user. Ids are real Retailrocket visitors; attributes are synthesised."""

    __tablename__ = "users"

    user_id: Mapped[int] = mapped_column(Integer, primary_key=True)
    source_user_id: Mapped[int] = mapped_column(BigInteger, nullable=False, unique=True)
    region_id: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    signup_at: Mapped[datetime] = mapped_column(nullable=False)
    device_pref: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    is_cold: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (Index("ix_users_region", "region_id"),)


class Interaction(Base):
    """One behavioural event. Append-only, high volume.

    No foreign keys by design: the loader writes these with a binary ``COPY``
    of hundreds of thousands of rows, where per-row FK validation dominates
    the cost. Referential integrity is enforced at load time instead.
    """

    __tablename__ = "interactions"

    event_id: Mapped[int] = mapped_column(BigInteger, primary_key=True)
    ts: Mapped[datetime] = mapped_column(nullable=False)
    user_id: Mapped[int] = mapped_column(Integer, nullable=False)
    item_id: Mapped[int] = mapped_column(Integer, nullable=False)
    session_id: Mapped[int | None] = mapped_column(BigInteger)
    event_type: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    merchant_id: Mapped[int | None] = mapped_column(Integer)
    vertical: Mapped[int | None] = mapped_column(SmallInteger)
    region_id: Mapped[int | None] = mapped_column(SmallInteger)
    device: Mapped[int | None] = mapped_column(SmallInteger)
    price_at_event: Mapped[float | None] = mapped_column(Float)
    is_repeat: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)

    __table_args__ = (
        # Both leading-column orders are needed: user history is read by
        # (user, time desc) and item statistics by (item, time desc).
        Index("ix_interactions_user_ts", "user_id", "ts"),
        Index("ix_interactions_item_ts", "item_id", "ts"),
        Index("ix_interactions_ts", "ts"),
    )


class RecommendationRequest(Base):
    """One served request, logged for monitoring and offline replay.

    This is what makes simulated A/B testing possible after the fact: the
    request, its context, the model that served it and the per-stage latency
    are all recorded, so a later analysis can reconstruct exactly what
    happened without re-running the models.
    """

    __tablename__ = "recommendation_requests"

    request_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    ts: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    user_id: Mapped[int | None] = mapped_column(Integer)
    external_user_id: Mapped[str | None] = mapped_column(String(64))
    k: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    feature_schema_version: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    context: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    cache_hit: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    is_cold_start: Mapped[bool] = mapped_column(Boolean, nullable=False, default=False)
    n_candidates: Mapped[int] = mapped_column(Integer, nullable=False, default=0)

    cache_ms: Mapped[float | None] = mapped_column(Float)
    feature_ms: Mapped[float | None] = mapped_column(Float)
    retrieval_ms: Mapped[float | None] = mapped_column(Float)
    ranking_ms: Mapped[float | None] = mapped_column(Float)
    rerank_ms: Mapped[float | None] = mapped_column(Float)
    total_ms: Mapped[float | None] = mapped_column(Float)

    results: Mapped[list[RecommendationResult]] = relationship(back_populates="request")

    __table_args__ = (
        Index("ix_rec_requests_user_ts", "user_id", "ts"),
        Index("ix_rec_requests_model_ts", "model_version", "ts"),
        # GIN over the context so "which requests had vertical=2 at 8am" is
        # answerable without a sequential scan.
        Index("ix_rec_requests_context", "context", postgresql_using="gin"),
    )


class RecommendationResult(Base):
    """One item within one served request.

    ``ml_score`` and ``business_delta`` are stored separately, exactly as the
    API returns them. Storing only the final score would make it impossible to
    answer afterwards whether a placement came from the model or from policy.
    """

    __tablename__ = "recommendation_results"

    request_id: Mapped[str] = mapped_column(
        ForeignKey("recommendation_requests.request_id", ondelete="CASCADE"), primary_key=True
    )
    rank: Mapped[int] = mapped_column(SmallInteger, primary_key=True)
    item_id: Mapped[int] = mapped_column(Integer, nullable=False)
    ml_score: Mapped[float] = mapped_column(Float, nullable=False)
    business_delta: Mapped[float] = mapped_column(Float, nullable=False, default=0.0)
    final_score: Mapped[float] = mapped_column(Float, nullable=False)
    sources: Mapped[list[str] | None] = mapped_column(ARRAY(Text))

    request: Mapped[RecommendationRequest] = relationship(back_populates="results")

    __table_args__ = (
        # Powers catalogue coverage and popularity-concentration queries.
        Index("ix_rec_results_item", "item_id"),
    )


class ModelMetadata(Base):
    """A registered model version and its lifecycle state."""

    __tablename__ = "model_metadata"

    model_version: Mapped[str] = mapped_column(String(128), primary_key=True)
    model_type: Mapped[str] = mapped_column(String(64), nullable=False)
    trained_at: Mapped[datetime] = mapped_column(nullable=False)
    dataset_hash: Mapped[str | None] = mapped_column(String(64))
    feature_schema_version: Mapped[int] = mapped_column(SmallInteger, nullable=False)
    params: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    metrics: Mapped[dict[str, Any] | None] = mapped_column(JSONB)
    artifact_uri: Mapped[str | None] = mapped_column(Text)
    mlflow_run_id: Mapped[str | None] = mapped_column(String(64))
    status: Mapped[str] = mapped_column(String(16), nullable=False, default="candidate")
    promoted_at: Mapped[datetime | None] = mapped_column()

    __table_args__ = (
        CheckConstraint(
            "status IN ('candidate', 'staging', 'production', 'archived')",
            name="ck_model_status",
        ),
        # The invariant that matters: at most ONE production model per type.
        # In application code this is a convention that a manual UPDATE can
        # break; here the database refuses.
        Index(
            "uq_one_production_model_per_type",
            "model_type",
            unique=True,
            postgresql_where=(status == "production"),
        ),
        Index("ix_model_metadata_type_trained", "model_type", "trained_at"),
    )


class Experiment(Base):
    """A control/treatment comparison."""

    __tablename__ = "experiments"

    experiment_id: Mapped[str] = mapped_column(String(36), primary_key=True)
    name: Mapped[str] = mapped_column(String(128), nullable=False, unique=True)
    created_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())
    control_model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    treatment_model_version: Mapped[str] = mapped_column(String(128), nullable=False)
    traffic_split: Mapped[float] = mapped_column(Float, nullable=False, default=0.5)

    is_simulated: Mapped[bool] = mapped_column(Boolean, nullable=False, default=True)
    """Defaults to true, NOT NULL. This project has no live user traffic, and
    encoding that in the schema means a row asserting a live experiment must
    do so explicitly rather than by omission."""

    status: Mapped[str] = mapped_column(String(16), nullable=False, default="draft")
    config: Mapped[dict[str, Any] | None] = mapped_column(JSONB)

    results: Mapped[list[ExperimentResult]] = relationship(back_populates="experiment")

    __table_args__ = (
        CheckConstraint(
            "traffic_split > 0 AND traffic_split < 1", name="ck_experiment_split_range"
        ),
    )


class ExperimentResult(Base):
    """One metric for one variant, with its confidence interval.

    The interval is stored alongside the point estimate rather than derived
    later, because a lift quoted without one invites reading noise as signal.
    """

    __tablename__ = "experiment_results"

    experiment_id: Mapped[str] = mapped_column(
        ForeignKey("experiments.experiment_id", ondelete="CASCADE"), primary_key=True
    )
    metric: Mapped[str] = mapped_column(String(64), primary_key=True)
    variant: Mapped[str] = mapped_column(String(32), primary_key=True)

    value: Mapped[float] = mapped_column(Float, nullable=False)
    lower_ci: Mapped[float | None] = mapped_column(Float)
    upper_ci: Mapped[float | None] = mapped_column(Float)
    n: Mapped[int] = mapped_column(BigInteger, nullable=False)
    p_value: Mapped[float | None] = mapped_column(Float)
    computed_at: Mapped[datetime] = mapped_column(nullable=False, server_default=func.now())

    experiment: Mapped[Experiment] = relationship(back_populates="results")

    __table_args__ = (
        UniqueConstraint("experiment_id", "metric", "variant", name="uq_experiment_metric_variant"),
    )


__all__ = [
    "Base",
    "Experiment",
    "ExperimentResult",
    "Interaction",
    "Item",
    "Merchant",
    "ModelMetadata",
    "RecommendationRequest",
    "RecommendationResult",
    "User",
]
