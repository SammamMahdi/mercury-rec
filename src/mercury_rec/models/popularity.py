"""Popularity baselines - the bar every other model has to clear.

Popularity is the most important baseline in recommender systems and the one
most often reported dishonestly. It is a genuinely strong method on sparse
data, so a paper or portfolio that shows a neural model beating a *weak*
popularity implementation has demonstrated very little.

The implementations here are therefore built to be as strong as the signal
allows, not as weak as convenient:

- **Weighted by intent.** A purchase counts far more than a view, using the
  configured weights from ``configs/events.yaml``.
- **Time-decayed.** Recent behaviour is weighted more heavily via an
  exponential half-life, because in on-demand commerce a preference from four
  months ago is much weaker evidence than one from last week.
- **Contextual.** Per-region, per-vertical and per-hour variants let
  popularity itself be personalised, which closes most of the easy gap a
  naive comparison would show.

Every count is computed from the **training window only**. Popularity derived
from the evaluation window would be leakage of the most direct possible kind:
it would encode which items were about to be popular.
"""

from __future__ import annotations

from typing import Any, Final, cast

import numpy as np
import pandas as pd

from mercury_rec.config.loader import load_config
from mercury_rec.config.schemas import EventsConfig
from mercury_rec.core.enums import EventType
from mercury_rec.core.logging import get_logger
from mercury_rec.models.base import RecommendationContext, Recommender

logger = get_logger(__name__)

_NS_PER_DAY: Final = 86_400_000_000_000

#: Time-of-day buckets. Six three-to-five hour bands rather than 24 hourly
#: ones: hourly buckets fragment an already sparse dataset into slices too
#: thin to estimate, and consumption patterns genuinely cluster into
#: meal-shaped bands rather than changing every hour.
HOUR_BUCKETS: Final[tuple[tuple[int, int, str], ...]] = (
    (0, 6, "night"),
    (6, 10, "breakfast"),
    (10, 14, "midday"),
    (14, 17, "afternoon"),
    (17, 21, "dinner"),
    (21, 24, "late"),
)


def hour_bucket(hour: int) -> int:
    """Map an hour of day to its bucket index."""
    for index, (start, end, _) in enumerate(HOUR_BUCKETS):
        if start <= hour < end:
            return index
    return 0


def event_weights() -> dict[int, float]:
    """Load per-event-type weights, keyed by :class:`EventType` value."""
    config = load_config("events", EventsConfig)
    raw = config.weights.model_dump()
    return {int(EventType[name.upper()]): float(weight) for name, weight in raw.items()}


class PopularityRecommender(Recommender):
    """Global popularity, weighted by intent and decayed by recency.

    Args:
        n_users: Catalogue user count.
        n_items: Catalogue item count.
        half_life_days: Exponential decay half-life. ``None`` disables decay,
            which is the classic naive baseline and is kept available so the
            decay's contribution can be measured rather than assumed.
    """

    name = "popularity"

    def __init__(self, n_users: int, n_items: int, *, half_life_days: float | None = 30.0) -> None:
        super().__init__(n_users, n_items)
        self.half_life_days = half_life_days
        self._scores = np.zeros(n_items, dtype=np.float64)
        self._weights = event_weights()

    def params(self) -> dict[str, Any]:
        return {"half_life_days": self.half_life_days}

    def _decayed_weights(self, interactions: pd.DataFrame) -> np.ndarray:
        """Per-row weight: intent weight, optionally decayed toward the end."""
        weights = (
            interactions["event_type"].map(self._weights).fillna(1.0).to_numpy(dtype=np.float64)
        )
        if self.half_life_days is None:
            return weights

        timestamps = interactions["ts"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
        # Decay is measured back from the END of the training window, which is
        # "now" from the model's point of view at serving time.
        age_days = (timestamps.max() - timestamps) / _NS_PER_DAY
        decayed: np.ndarray = weights * np.power(0.5, age_days / self.half_life_days)
        return decayed

    def _fit(self, interactions: pd.DataFrame) -> dict[str, Any]:
        weights = self._decayed_weights(interactions)
        items = interactions["item_id"].to_numpy(dtype=np.int64)
        self._scores = np.bincount(items, weights=weights, minlength=self.n_items)

        nonzero = int((self._scores > 0).sum())
        return {
            "items_with_signal": nonzero,
            "coverage": round(nonzero / self.n_items, 4),
        }

    def _score(
        self, user_id: int, candidates: np.ndarray, context: RecommendationContext
    ) -> np.ndarray:
        scores: np.ndarray = self._scores[candidates]
        return scores

    @property
    def item_scores(self) -> np.ndarray:
        """Raw popularity scores, reused as a prior by other components."""
        self._require_fitted()
        return self._scores


class ContextualPopularityRecommender(PopularityRecommender):
    """Popularity conditioned on region, vertical and time-of-day.

    This is what makes the baseline genuinely hard to beat, and it is the
    honest comparison: a neural model should be measured against contextual
    popularity, not against a global top-50 list.

    Falls back through progressively broader contexts when a slice is too
    thin to trust - a (region, vertical, hour-bucket) cell with three
    observations is noise, not a preference. The fallback chain is:

        (region, vertical, bucket) -> (region, vertical) -> (vertical) -> global

    ``min_observations`` sets the threshold for trusting a slice.
    """

    name = "popularity_contextual"

    def __init__(
        self,
        n_users: int,
        n_items: int,
        *,
        half_life_days: float | None = 30.0,
        min_observations: int = 30,
    ) -> None:
        super().__init__(n_users, n_items, half_life_days=half_life_days)
        self.min_observations = min_observations
        self._by_region_vertical_bucket: dict[tuple[int, int, int], np.ndarray] = {}
        self._by_region_vertical: dict[tuple[int, int], np.ndarray] = {}
        self._by_vertical: dict[int, np.ndarray] = {}

    def params(self) -> dict[str, Any]:
        return {
            "half_life_days": self.half_life_days,
            "min_observations": self.min_observations,
        }

    def _fit(self, interactions: pd.DataFrame) -> dict[str, Any]:
        extra = super()._fit(interactions)

        weights = self._decayed_weights(interactions)
        frame = pd.DataFrame(
            {
                "item_id": interactions["item_id"].to_numpy(dtype=np.int64),
                "weight": weights,
                "region_id": interactions["region_id"].fillna(-1).to_numpy(dtype=np.int64),
                "vertical": interactions["vertical"].fillna(-1).to_numpy(dtype=np.int64),
                "bucket": [hour_bucket(h) for h in interactions["ts"].dt.hour.to_numpy()],
            }
        )

        def _accumulate(group: pd.DataFrame) -> np.ndarray:
            return np.bincount(
                group["item_id"].to_numpy(),
                weights=group["weight"].to_numpy(),
                minlength=self.n_items,
            )

        # A multi-column groupby key arrives as a tuple of Hashable, so the
        # int-ness of each part is cast back explicitly rather than assumed.
        for key3, group in frame.groupby(["region_id", "vertical", "bucket"], sort=False):
            if len(group) >= self.min_observations:
                triple = tuple(int(part) for part in cast("tuple[int, ...]", key3))
                self._by_region_vertical_bucket[(triple[0], triple[1], triple[2])] = _accumulate(
                    group
                )
        for key2, group in frame.groupby(["region_id", "vertical"], sort=False):
            if len(group) >= self.min_observations:
                pair = tuple(int(part) for part in cast("tuple[int, ...]", key2))
                self._by_region_vertical[(pair[0], pair[1])] = _accumulate(group)
        for vertical_key, group in frame.groupby("vertical", sort=False):
            if len(group) >= self.min_observations:
                self._by_vertical[int(cast("int", vertical_key))] = _accumulate(group)

        extra.update(
            {
                "context_cells_rvb": len(self._by_region_vertical_bucket),
                "context_cells_rv": len(self._by_region_vertical),
                "context_cells_v": len(self._by_vertical),
            }
        )
        return extra

    def _score(
        self, user_id: int, candidates: np.ndarray, context: RecommendationContext
    ) -> np.ndarray:
        region = context.region_id if context.region_id is not None else -1
        vertical = context.vertical if context.vertical is not None else -1

        if context.hour is not None:
            bucket = hour_bucket(context.hour)
            cell = self._by_region_vertical_bucket.get((region, vertical, bucket))
            if cell is not None:
                by_bucket: np.ndarray = cell[candidates]
                return by_bucket

        cell = self._by_region_vertical.get((region, vertical))
        if cell is not None:
            by_region_vertical: np.ndarray = cell[candidates]
            return by_region_vertical

        cell = self._by_vertical.get(vertical)
        if cell is not None:
            by_vertical: np.ndarray = cell[candidates]
            return by_vertical

        fallback: np.ndarray = self._scores[candidates]
        return fallback


class TrendingRecommender(PopularityRecommender):
    """Recent-window popularity: what is rising, not what is established.

    A short trailing window rather than a decayed lifetime count. The two
    answer different questions - decayed popularity still favours long-running
    hits, while a hard window surfaces genuinely new items and is the more
    useful cold-start prior for a catalogue that turns over.
    """

    name = "popularity_trending"

    def __init__(self, n_users: int, n_items: int, *, window_days: float = 7.0) -> None:
        super().__init__(n_users, n_items, half_life_days=None)
        self.window_days = window_days

    def params(self) -> dict[str, Any]:
        return {"window_days": self.window_days}

    def _fit(self, interactions: pd.DataFrame) -> dict[str, Any]:
        # pd.to_timedelta with an explicit unit: pandas deprecates the generic
        # NumPy timedelta path that Timedelta(days=<float>) can take.
        cutoff = interactions["ts"].max() - pd.to_timedelta(self.window_days, unit="D")
        recent = interactions[interactions["ts"] >= cutoff]
        if recent.empty:
            logger.warning("model.trending.empty_window", window_days=self.window_days)
            recent = interactions

        weights = recent["event_type"].map(self._weights).fillna(1.0).to_numpy(dtype=np.float64)
        self._scores = np.bincount(
            recent["item_id"].to_numpy(dtype=np.int64), weights=weights, minlength=self.n_items
        )
        return {
            "window_events": len(recent),
            "items_with_signal": int((self._scores > 0).sum()),
        }


__all__ = [
    "HOUR_BUCKETS",
    "ContextualPopularityRecommender",
    "PopularityRecommender",
    "TrendingRecommender",
    "event_weights",
    "hour_bucket",
]
