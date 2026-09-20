"""Dataframe schemas for every stage of the data pipeline.

Pandera is used rather than Pydantic here because these are *columnar*
contracts over millions of rows: Pandera validates vectorised, while Pydantic
would instantiate one object per row. Pydantic still owns the API request and
response models, where per-object validation is exactly what is wanted.

Two distinct jobs are done here:

1. **Ingest schemas** (``RawEvents``, ``RawItemProperties``, ``RawCategoryTree``)
   describe the Retailrocket CSVs *as published*. They are the tripwire: this
   project never assumes the upstream column layout, so if Kaggle republishes
   the dataset with different columns, the pipeline fails immediately naming
   the offending column instead of silently producing an empty join and a
   meaningless model.

2. **Processed schemas** describe our own canonical frames after cleaning,
   augmentation and sessionisation. They encode the invariants the models rely
   on: no null user ids, chronological ordering, event types drawn from the
   closed vocabulary in :mod:`mercury_rec.core.enums`.

Processed schemas use ``strict="filter"`` so an accidental extra column cannot
ride along into a feature matrix unnoticed.
"""

from __future__ import annotations

from typing import TYPE_CHECKING

import pandera.pandas as pa
from pandera.typing import Series

from mercury_rec.core.enums import Device, EventType, Vertical

if TYPE_CHECKING:
    import pandas as pd

# ---------------------------------------------------------------------------
# Raw Retailrocket CSVs, exactly as published on Kaggle.
# ---------------------------------------------------------------------------


class RawEvents(pa.DataFrameModel):
    """``events.csv`` - the real behavioural log.

    ``transactionid`` is populated only on ``transaction`` rows and is null
    everywhere else, hence nullable. ``timestamp`` is epoch **milliseconds**,
    not seconds; treating it as seconds silently places the entire dataset in
    1970 and every temporal feature collapses to a constant.
    """

    timestamp: Series[int] = pa.Field(gt=0, description="Epoch milliseconds (UTC)")
    visitorid: Series[int] = pa.Field(ge=0)
    event: Series[str] = pa.Field(isin=["view", "addtocart", "transaction"])
    itemid: Series[int] = pa.Field(ge=0)
    transactionid: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = True


class RawItemProperties(pa.DataFrameModel):
    """``item_properties_part{1,2}.csv`` - time-versioned item attributes.

    This is a change log, not a snapshot: one row per (item, property,
    change-time). Reading it as a snapshot multiply-counts items. Values are
    hashed except ``categoryid`` and ``available``, and numeric values carry an
    ``n`` prefix (e.g. ``n277.200``) - which is why ``value`` is a string
    rather than a float.
    """

    timestamp: Series[int] = pa.Field(gt=0)
    itemid: Series[int] = pa.Field(ge=0)
    property: Series[str] = pa.Field(nullable=False)
    value: Series[str] = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = True


class RawCategoryTree(pa.DataFrameModel):
    """``category_tree.csv`` - the category hierarchy.

    ``parentid`` is null for root categories, so the structure is a forest
    rather than a single tree. Code walking upward must terminate on null, not
    on a sentinel id.
    """

    categoryid: Series[int] = pa.Field(ge=0)
    parentid: Series[float] = pa.Field(nullable=True)

    class Config:
        strict = True
        coerce = True


# ---------------------------------------------------------------------------
# Canonical processed frames.
# ---------------------------------------------------------------------------

_EVENT_VALUES = [int(e) for e in EventType]
_VERTICAL_VALUES = [int(v) for v in Vertical]
_DEVICE_VALUES = [int(d) for d in Device]


class Interactions(pa.DataFrameModel):
    """The canonical behavioural frame every model consumes.

    Dtypes are deliberately narrow. Across tens of millions of rows the gap
    between pandas' int64/float64 defaults and these types is roughly 3x peak
    memory, which on a 16 GB machine decides whether the pipeline runs at all.

    ``ts`` must be timezone-aware UTC. Naive timestamps are rejected because
    time-of-day features underpin the contextual-personalisation claim, and a
    silent timezone shift would move every event by hours while everything
    still appeared to work.
    """

    event_id: Series[int] = pa.Field(ge=0, unique=True)
    ts: Series[pa.DateTime] = pa.Field(nullable=False)
    user_id: Series[int] = pa.Field(ge=0, nullable=False)
    item_id: Series[int] = pa.Field(ge=0, nullable=False)
    session_id: Series[int] = pa.Field(ge=0, nullable=False)
    event_type: Series[int] = pa.Field(isin=_EVENT_VALUES)
    merchant_id: Series[int] = pa.Field(ge=0, nullable=True)
    vertical: Series[int] = pa.Field(isin=_VERTICAL_VALUES, nullable=True)
    region_id: Series[int] = pa.Field(ge=0, nullable=True)
    device: Series[int] = pa.Field(isin=_DEVICE_VALUES, nullable=True)
    price_at_event: Series[float] = pa.Field(ge=0, nullable=True)
    is_repeat: Series[bool] = pa.Field(nullable=False)

    class Config:
        strict = "filter"
        coerce = True

    @pa.dataframe_check(name="chronologically_sorted")
    def _sorted_by_time(cls, df: pd.DataFrame) -> bool:  # noqa: N805
        """The as-of feature engine assumes chronological order.

        The forward-only scan in ``features/asof.py`` is correct only on a
        time-sorted frame. Asserting it here turns a silent wrong-numbers bug
        into a loud, early failure.
        """
        return bool(df["ts"].is_monotonic_increasing)


class Items(pa.DataFrameModel):
    """Item catalogue: real Retailrocket ids and categories, augmented attributes."""

    item_id: Series[int] = pa.Field(ge=0, unique=True)
    category_id: Series[int] = pa.Field(ge=0, nullable=True)
    merchant_id: Series[int] = pa.Field(ge=0, nullable=False)
    vertical: Series[int] = pa.Field(isin=_VERTICAL_VALUES)
    price: Series[float] = pa.Field(gt=0, nullable=False)
    base_rating: Series[float] = pa.Field(ge=0, le=5, nullable=True)
    available_from: Series[pa.DateTime] = pa.Field(nullable=False)
    available_until: Series[pa.DateTime] = pa.Field(nullable=True)
    is_cold: Series[bool] = pa.Field(nullable=False)

    class Config:
        strict = "filter"
        coerce = True


class Users(pa.DataFrameModel):
    """User dimension. Ids are Retailrocket ``visitorid`` values, densely re-indexed."""

    user_id: Series[int] = pa.Field(ge=0, unique=True)
    region_id: Series[int] = pa.Field(ge=0, nullable=False)
    signup_at: Series[pa.DateTime] = pa.Field(nullable=False)
    device_pref: Series[int] = pa.Field(isin=_DEVICE_VALUES)
    is_cold: Series[bool] = pa.Field(nullable=False)

    class Config:
        strict = "filter"
        coerce = True


class Merchants(pa.DataFrameModel):
    """Merchant dimension - entirely synthesised; Retailrocket has no merchants."""

    merchant_id: Series[int] = pa.Field(ge=0, unique=True)
    vertical: Series[int] = pa.Field(isin=_VERTICAL_VALUES)
    region_id: Series[int] = pa.Field(ge=0)
    rating: Series[float] = pa.Field(ge=0, le=5)
    delivery_radius_km: Series[float] = pa.Field(gt=0)
    avg_delivery_minutes: Series[int] = pa.Field(gt=0)

    class Config:
        strict = "filter"
        coerce = True


class Sessions(pa.DataFrameModel):
    """Sessions derived from the real event stream by inactivity gap."""

    session_id: Series[int] = pa.Field(ge=0, unique=True)
    user_id: Series[int] = pa.Field(ge=0)
    started_at: Series[pa.DateTime] = pa.Field(nullable=False)
    ended_at: Series[pa.DateTime] = pa.Field(nullable=False)
    n_events: Series[int] = pa.Field(gt=0)
    device: Series[int] = pa.Field(isin=_DEVICE_VALUES)

    class Config:
        strict = "filter"
        coerce = True

    @pa.dataframe_check(name="session_ends_after_start")
    def _ordered(cls, df: pd.DataFrame) -> bool:  # noqa: N805
        return bool((df["ended_at"] >= df["started_at"]).all())


__all__ = [
    "Interactions",
    "Items",
    "Merchants",
    "RawCategoryTree",
    "RawEvents",
    "RawItemProperties",
    "Sessions",
    "Users",
]
