"""As-of feature computation: the engine the no-leakage guarantee rests on.

The problem
-----------
Every training example is a *(user, item, timestamp)* triple, and the features
attached to it must reflect only what was knowable **strictly before** that
timestamp. Getting this wrong is the classic way a recommender evaluation
becomes meaningless: the model trains on aggregates that already incorporate
the outcome it is asked to predict, every offline metric improves, and the
system fails in production.

Why the obvious implementations are wrong
-----------------------------------------
*Filtering per row* -- ``events[events.ts < row.ts]`` for each row -- is
correct but quadratic. At 545k rows that is ~3 x 10^11 comparisons.

*Grouping over the whole frame* -- ``df.groupby("user_id").agg(...)`` -- is
fast and **silently wrong**: the aggregate for a January event includes
September behaviour. It produces excellent offline numbers and a broken model,
which is the worst failure mode available because nothing looks wrong.

The approach used here
----------------------
A single forward-only scan over chronologically sorted events, maintaining
running aggregate state. For each event, in this exact order:

1. :func:`emit_features` -- read features from the current state.
2. :func:`apply_event` -- fold this event into the state.

Leakage is therefore impossible *by construction* rather than by convention:
at emit time the state has only ever observed strictly-earlier events. There
is no filtering step that could be forgotten, and no window that could be
mis-specified. The scan is O(n) with O(users + items + pairs) memory.

Ties matter. Events sharing an identical timestamp are emitted against the
state as it stood before *any* of them was applied, so two simultaneous events
cannot see each other. That is the conservative reading of "strictly before".

Online/offline consistency
--------------------------
:func:`emit_features` is the **only** implementation of "what are this
(user, item) pair's features right now". Offline training calls it once per
historical event; online serving calls it per candidate at request time
against state hydrated from Redis. There are not two implementations that are
meant to agree -- there is one, called from two places. That makes
training/serving skew a structural impossibility rather than a convention
someone has to remember, and it is verified by an explicit parity test.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Final

import numpy as np
import pandas as pd

from mercury_rec.core.enums import EventType
from mercury_rec.core.logging import get_logger

logger = get_logger(__name__)

_NS_PER_DAY: Final = 86_400_000_000_000

#: Returned when a quantity is undefined because nothing has been observed
#: yet -- a user's first-ever event has no "days since last interaction".
#: NaN rather than 0: zero would be a factual claim the data does not support,
#: and tree models will happily split on it. LightGBM handles NaN natively.
UNDEFINED: Final = np.float32(np.nan)

#: Feature columns produced, in emit order. The ranking model, the online
#: feature service and the API response all read this, so the contract lives
#: in exactly one place.
FEATURE_COLUMNS: Final[tuple[str, ...]] = (
    "user_event_count",
    "user_purchase_count",
    "user_conversion_rate",
    "user_distinct_items",
    "user_days_since_last",
    "user_tenure_days",
    "user_avg_interaction_price",
    "user_avg_order_value",
    "item_event_count",
    "item_purchase_count",
    "item_conversion_rate",
    "item_distinct_users",
    "item_days_since_last",
    "item_age_days",
    "ui_prior_events",
    "ui_prior_purchases",
    "ui_days_since_last",
    "user_category_affinity",
    "user_merchant_affinity",
    "price_ratio_to_user_avg",
)

N_FEATURES: Final = len(FEATURE_COLUMNS)

_CONVERSION_EVENTS: Final = frozenset({int(EventType.PURCHASE), int(EventType.REPEAT_PURCHASE)})

#: Composite-key strides. Vertical is a small closed enum; merchant ids run to
#: a few thousand. Generous strides keep the packed keys collision-free.
_VERTICAL_STRIDE: Final = 1_000
_MERCHANT_STRIDE: Final = 1_000_000


@dataclass(frozen=True, slots=True)
class EventRow:
    """One scoring context: a (user, item) pair evaluated at an instant.

    Used for both a historical event during training and a candidate item at
    request time -- which is exactly why one emit path can serve both.
    """

    user: int
    item: int
    now: int
    """Timestamp in epoch nanoseconds."""
    event_type: int = int(EventType.VIEW)
    price: float = 0.0
    vertical: int = -1
    merchant: int = -1


@dataclass(slots=True)
class AsOfState:
    """Running aggregate state at a point in time.

    Dense numpy arrays for per-user and per-item counters: ids are densified
    to 0..n-1 during ingest, so an array index *is* the id and lookup is O(1)
    with no hashing. Dicts for the sparse cross-entity counters, where a dense
    ``n_users x n_items`` matrix would be 2.4 billion cells holding 778k
    non-zeros.
    """

    n_users: int
    n_items: int

    user_events: np.ndarray = field(init=False)
    user_purchases: np.ndarray = field(init=False)
    user_order_sum: np.ndarray = field(init=False)
    user_order_count: np.ndarray = field(init=False)
    user_price_sum: np.ndarray = field(init=False)
    user_price_count: np.ndarray = field(init=False)
    user_last_ts: np.ndarray = field(init=False)
    user_first_ts: np.ndarray = field(init=False)
    user_distinct_items: np.ndarray = field(init=False)

    item_events: np.ndarray = field(init=False)
    item_purchases: np.ndarray = field(init=False)
    item_last_ts: np.ndarray = field(init=False)
    item_first_ts: np.ndarray = field(init=False)
    item_distinct_users: np.ndarray = field(init=False)

    #: Keyed by ``user_id * n_items + item_id``. A single int64 key hashes
    #: markedly faster than a tuple, and dense ids make it exact.
    ui_events: dict[int, int] = field(default_factory=dict)
    ui_purchases: dict[int, int] = field(default_factory=dict)
    ui_last_ts: dict[int, int] = field(default_factory=dict)

    user_category: dict[int, int] = field(default_factory=dict)
    user_merchant: dict[int, int] = field(default_factory=dict)

    def __post_init__(self) -> None:
        n_u, n_i = self.n_users, self.n_items
        self.user_events = np.zeros(n_u, dtype=np.int32)
        self.user_purchases = np.zeros(n_u, dtype=np.int32)
        self.user_order_sum = np.zeros(n_u, dtype=np.float64)
        self.user_order_count = np.zeros(n_u, dtype=np.int32)
        self.user_price_sum = np.zeros(n_u, dtype=np.float64)
        self.user_price_count = np.zeros(n_u, dtype=np.int32)
        self.user_last_ts = np.zeros(n_u, dtype=np.int64)
        self.user_first_ts = np.zeros(n_u, dtype=np.int64)
        self.user_distinct_items = np.zeros(n_u, dtype=np.int32)

        self.item_events = np.zeros(n_i, dtype=np.int32)
        self.item_purchases = np.zeros(n_i, dtype=np.int32)
        self.item_last_ts = np.zeros(n_i, dtype=np.int64)
        self.item_first_ts = np.zeros(n_i, dtype=np.int64)
        self.item_distinct_users = np.zeros(n_i, dtype=np.int32)

    def pair_key(self, user_id: int, item_id: int) -> int:
        return user_id * self.n_items + item_id


def _ratio(numerator: float, denominator: float) -> float:
    """Ratio that is NaN rather than 0 when the denominator is empty."""
    if denominator <= 0:
        return float(UNDEFINED)
    return numerator / denominator


def _days_between(later_ns: int, earlier_ns: int) -> float:
    if earlier_ns <= 0:
        return float(UNDEFINED)
    return (later_ns - earlier_ns) / _NS_PER_DAY


def emit_features(state: AsOfState, row: EventRow, out: np.ndarray) -> None:
    """Write this pair's features, as of ``row.now``, into ``out``.

    **The single definition of a feature vector in this system.** Offline
    training calls it once per historical event; online serving calls it per
    candidate at request time. Changing a feature here changes both paths at
    once, which is what makes training/serving skew structurally impossible
    rather than a convention.

    Args:
        state: Aggregates reflecting only events strictly before ``row.now``.
        row: The (user, item, instant) being scored.
        out: Length-:data:`N_FEATURES` float32 buffer, written in place.
            Caller-supplied so the batch path writes directly into a
            preallocated matrix instead of allocating per row.
    """
    user, item, now = row.user, row.item, row.now
    pair = user * state.n_items + item

    user_events = int(state.user_events[user])
    item_events = int(state.item_events[item])
    order_count = int(state.user_order_count[user])
    price_count = int(state.user_price_count[user])

    # Price sensitivity uses ALL interactions -- browsing reveals preference
    # too, and is dense. Average order value uses purchases only and is
    # necessarily sparse at a ~1.3% conversion rate. Keeping them apart means
    # the dense signal stays usable without pretending the sparse one is.
    avg_price = float(state.user_price_sum[user] / price_count) if price_count else 0.0
    avg_order = float(state.user_order_sum[user] / order_count) if order_count else 0.0

    category_key = user * _VERTICAL_STRIDE + row.vertical if row.vertical >= 0 else -1
    merchant_key = user * _MERCHANT_STRIDE + row.merchant if row.merchant >= 0 else -1

    out[0] = user_events
    out[1] = state.user_purchases[user]
    out[2] = _ratio(float(state.user_purchases[user]), float(user_events))
    out[3] = state.user_distinct_items[user]
    out[4] = _days_between(now, int(state.user_last_ts[user]))
    out[5] = _days_between(now, int(state.user_first_ts[user]))
    out[6] = avg_price if price_count else UNDEFINED
    out[7] = avg_order if order_count else UNDEFINED
    out[8] = item_events
    out[9] = state.item_purchases[item]
    out[10] = _ratio(float(state.item_purchases[item]), float(item_events))
    out[11] = state.item_distinct_users[item]
    out[12] = _days_between(now, int(state.item_last_ts[item]))
    out[13] = _days_between(now, int(state.item_first_ts[item]))
    out[14] = state.ui_events.get(pair, 0)
    out[15] = state.ui_purchases.get(pair, 0)
    out[16] = _days_between(now, state.ui_last_ts.get(pair, 0))
    out[17] = (
        _ratio(float(state.user_category.get(category_key, 0)), float(user_events))
        if category_key >= 0
        else UNDEFINED
    )
    out[18] = (
        _ratio(float(state.user_merchant.get(merchant_key, 0)), float(user_events))
        if merchant_key >= 0
        else UNDEFINED
    )
    out[19] = _ratio(row.price, avg_price) if avg_price > 0 else UNDEFINED


def apply_event(state: AsOfState, row: EventRow) -> None:
    """Fold one observed event into the state. Call only after emitting.

    Also the single definition of a state update, for the same reason: the
    online path folds in events arriving at ``POST /events`` through this, so
    online state evolves exactly as the offline scan would have evolved it.
    """
    user, item, now = row.user, row.item, row.now
    pair = user * state.n_items + item

    if state.user_events[user] == 0:
        state.user_first_ts[user] = now
    if state.item_events[item] == 0:
        state.item_first_ts[item] = now
    if pair not in state.ui_events:
        state.user_distinct_items[user] += 1
        state.item_distinct_users[item] += 1

    state.user_events[user] += 1
    state.item_events[item] += 1
    state.user_last_ts[user] = now
    state.item_last_ts[item] = now
    state.ui_events[pair] = state.ui_events.get(pair, 0) + 1
    state.ui_last_ts[pair] = now

    if row.price > 0:
        state.user_price_sum[user] += row.price
        state.user_price_count[user] += 1

    if row.event_type in _CONVERSION_EVENTS:
        state.user_purchases[user] += 1
        state.item_purchases[item] += 1
        state.ui_purchases[pair] = state.ui_purchases.get(pair, 0) + 1
        if row.price > 0:
            state.user_order_sum[user] += row.price
            state.user_order_count[user] += 1

    if row.vertical >= 0:
        key = user * _VERTICAL_STRIDE + row.vertical
        state.user_category[key] = state.user_category.get(key, 0) + 1
    if row.merchant >= 0:
        key = user * _MERCHANT_STRIDE + row.merchant
        state.user_merchant[key] = state.user_merchant.get(key, 0) + 1


def rows_from_frame(events: pd.DataFrame) -> list[EventRow]:
    """Materialise a frame as :class:`EventRow` records.

    Converting to numpy up front and building records once is markedly faster
    than pandas row access inside a 545k-iteration loop.
    """
    n = len(events)
    users = events["user_id"].to_numpy(dtype=np.int64)
    items = events["item_id"].to_numpy(dtype=np.int64)
    stamps = events["ts"].to_numpy(dtype="datetime64[ns]").astype(np.int64)
    types = events["event_type"].to_numpy(dtype=np.int64)

    def _column(name: str, default: float) -> np.ndarray:
        if name not in events.columns:
            return np.full(n, default, dtype=np.float64)
        return events[name].fillna(default).to_numpy(dtype=np.float64)

    prices = _column("price_at_event", 0.0)
    verticals = _column("vertical", -1.0)
    merchants = _column("merchant_id", -1.0)

    return [
        EventRow(
            user=int(users[i]),
            item=int(items[i]),
            now=int(stamps[i]),
            event_type=int(types[i]),
            price=float(prices[i]),
            vertical=int(verticals[i]),
            merchant=int(merchants[i]),
        )
        for i in range(n)
    ]


def compute_asof_features(
    events: pd.DataFrame,
    *,
    n_users: int,
    n_items: int,
    state: AsOfState | None = None,
    return_state: bool = False,
) -> tuple[pd.DataFrame, AsOfState] | pd.DataFrame:
    """Compute leakage-free features for every event in ``events``.

    Args:
        events: Chronologically sorted, with ``user_id``, ``item_id``, ``ts``
            and ``event_type``; optionally ``price_at_event``, ``vertical``,
            ``merchant_id``.
        n_users: Total distinct users (sizes the dense state arrays).
        n_items: Total distinct items.
        state: State to continue from. Passing the state returned by the
            training window is how validation and test features get training
            history without peeking forward.
        return_state: Also return the final state.

    Returns:
        A frame with one row per input event, columns :data:`FEATURE_COLUMNS`,
        aligned to the input index.

    Raises:
        ValueError: If ``events`` is not sorted by ``ts``. The whole guarantee
            depends on that ordering, so it is checked rather than assumed.
    """
    working = state if state is not None else AsOfState(n_users, n_items)

    if events.empty:
        empty = pd.DataFrame({name: np.empty(0, dtype=np.float32) for name in FEATURE_COLUMNS})
        return (empty, working) if return_state else empty

    if not events["ts"].is_monotonic_increasing:
        raise ValueError(
            "compute_asof_features requires chronologically sorted events. "
            "Out-of-order rows would let a feature observe a later event, "
            "which is precisely the leakage this function exists to prevent."
        )

    rows = rows_from_frame(events)
    out = np.empty((len(rows), N_FEATURES), dtype=np.float32)

    # Tied timestamps are emitted before any of them is applied, so two
    # simultaneous events cannot observe each other.
    pending: list[EventRow] = []
    pending_ts = -1

    for index, row in enumerate(rows):
        if pending and pending_ts != row.now:
            for queued in pending:
                apply_event(working, queued)
            pending.clear()
        pending_ts = row.now

        emit_features(working, row, out[index])
        pending.append(row)

    for queued in pending:
        apply_event(working, queued)

    frame = pd.DataFrame(out, columns=list(FEATURE_COLUMNS), index=events.index)
    logger.info("features.asof.complete", rows=len(rows), features=N_FEATURES)
    return (frame, working) if return_state else frame


__all__ = [
    "FEATURE_COLUMNS",
    "N_FEATURES",
    "UNDEFINED",
    "AsOfState",
    "EventRow",
    "apply_event",
    "compute_asof_features",
    "emit_features",
    "rows_from_frame",
]
