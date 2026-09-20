"""Pydantic models for the YAML files under ``configs/``.

Validating configuration into typed models means a mistyped key fails at load
time, naming the field, instead of silently falling back to a default and
quietly changing an experiment's result months later.

These live in the ``config`` layer (the lowest) so any layer may depend on
them without creating an upward import.
"""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Frozen(BaseModel):
    """Base: immutable, and rejects unknown keys.

    ``extra="forbid"`` is the point of this whole module. Silently ignoring an
    unrecognised key is how a typo like ``lerning_rate`` costs a day.
    """

    model_config = ConfigDict(frozen=True, extra="forbid")


# ---------------------------------------------------------------------------
# events.yaml
# ---------------------------------------------------------------------------


class EventWeights(_Frozen):
    """Per-event-type implicit-feedback confidence weights."""

    impression: float = Field(ge=0)
    skip: float = Field(ge=0)
    click: float = Field(ge=0)
    view: float = Field(ge=0)
    search: float = Field(ge=0)
    wishlist: float = Field(ge=0)
    add_to_cart: float = Field(ge=0)
    remove_from_cart: float = Field(ge=0)
    rating: float = Field(ge=0)
    purchase: float = Field(ge=0)
    repeat_purchase: float = Field(ge=0)

    @model_validator(mode="after")
    def _intent_ordering_holds(self) -> EventWeights:
        """Stronger intent must carry more weight.

        Without this, an edit that made a click outrank a purchase would train
        successfully and silently degrade every implicit-feedback model.
        """
        if not self.impression < self.click < self.add_to_cart < self.purchase:
            raise ValueError(
                "Event weights must increase with intent: "
                "impression < click < add_to_cart < purchase. Got "
                f"{self.impression}, {self.click}, {self.add_to_cart}, {self.purchase}."
            )
        if self.purchase > self.repeat_purchase:
            raise ValueError("repeat_purchase must weigh at least as much as purchase.")
        return self


class EventsConfig(_Frozen):
    weights: EventWeights
    alpha: float = Field(gt=0, description="Implicit confidence scaling: c = 1 + alpha * w")
    recency_half_life_days: float = Field(gt=0)
    cf_min_event_weight: float = Field(ge=0)


# ---------------------------------------------------------------------------
# data.yaml
# ---------------------------------------------------------------------------


class PresetConfig(_Frozen):
    max_users: int | None = None
    max_items: int | None = None
    min_user_interactions: int = Field(ge=1)
    min_item_interactions: int = Field(ge=1)


class KCoreConfig(_Frozen):
    enabled: bool = True
    max_iterations: int = Field(ge=1, le=200)


class SessionConfig(_Frozen):
    inactivity_gap_minutes: int = Field(gt=0)
    max_events_per_session: int = Field(gt=0)


class SplitConfig(_Frozen):
    strategy: str
    train_fraction: float = Field(gt=0, lt=1)
    validation_fraction: float = Field(gt=0, lt=1)
    require_train_history: bool = True

    @model_validator(mode="after")
    def _leaves_room_for_test(self) -> SplitConfig:
        if self.train_fraction + self.validation_fraction >= 1.0:
            raise ValueError(
                "train_fraction + validation_fraction must be < 1 so a test "
                f"window remains; got {self.train_fraction} + {self.validation_fraction}."
            )
        return self


class VerticalWeights(_Frozen):
    food_delivery: float = Field(gt=0)
    restaurant: float = Field(gt=0)
    grocery: float = Field(gt=0)
    pharmacy: float = Field(gt=0)
    commerce: float = Field(gt=0)
    lifestyle: float = Field(gt=0)


class VerticalsConfig(_Frozen):
    weights: VerticalWeights


class RegionsConfig(_Frozen):
    count: int = Field(gt=0, le=1000)
    population_zipf_s: float = Field(gt=0)


class RangeFloat(_Frozen):
    min: float
    max: float

    @model_validator(mode="after")
    def _ordered(self) -> RangeFloat:
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must not exceed max ({self.max}).")
        return self


class RangeInt(_Frozen):
    min: int
    max: int

    @model_validator(mode="after")
    def _ordered(self) -> RangeInt:
        if self.min > self.max:
            raise ValueError(f"min ({self.min}) must not exceed max ({self.max}).")
        return self


class MerchantsConfig(_Frozen):
    count_per_vertical: int = Field(gt=0)
    rating_beta_a: float = Field(gt=0)
    rating_beta_b: float = Field(gt=0)
    delivery_radius_km: RangeFloat
    delivery_minutes: RangeInt


class LogNormalParams(_Frozen):
    mu: float
    sigma: float = Field(gt=0)


class PricesConfig(_Frozen):
    lognormal: dict[str, LogNormalParams]
    min_price: float = Field(gt=0)


class ColdStartConfig(_Frozen):
    item_first_seen_within_days: int = Field(ge=0)
    user_first_seen_within_days: int = Field(ge=0)


class AugmentationSection(_Frozen):
    verticals: VerticalsConfig
    regions: RegionsConfig
    merchants: MerchantsConfig
    prices: PricesConfig
    cold_start: ColdStartConfig


class OutputConfig(_Frozen):
    partition_by: str
    compression: str
    row_group_size: int = Field(gt=0)


class DataConfig(_Frozen):
    """Top-level model for ``configs/data.yaml``."""

    seed: int
    presets: dict[str, PresetConfig]
    k_core: KCoreConfig
    sessions: SessionConfig
    split: SplitConfig
    augmentation: AugmentationSection
    output: OutputConfig

    def preset(self, name: str) -> PresetConfig:
        """Return a named preset, listing the valid names if it is absent."""
        try:
            return self.presets[name]
        except KeyError as exc:
            raise KeyError(
                f"Unknown data preset {name!r}. Available: {sorted(self.presets)}."
            ) from exc


__all__ = [
    "AugmentationSection",
    "ColdStartConfig",
    "DataConfig",
    "EventWeights",
    "EventsConfig",
    "KCoreConfig",
    "LogNormalParams",
    "MerchantsConfig",
    "OutputConfig",
    "PresetConfig",
    "PricesConfig",
    "RangeFloat",
    "RangeInt",
    "RegionsConfig",
    "SessionConfig",
    "SplitConfig",
    "VerticalWeights",
    "VerticalsConfig",
]
