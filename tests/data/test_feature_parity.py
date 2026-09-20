"""Online/offline feature parity: the training-serving skew proof.

The spec's requirement is that training-time and serving-time feature
computation stay consistent. Asserting that in prose is worthless; this
module demonstrates it.

Training/serving skew is notoriously hard to detect in production. Both
systems keep running, no exception is raised, no shape mismatches - the model
simply degrades because it is scored on features that differ subtly from the
ones it learned. By the time it is noticed, the cause is weeks of drift away.

These tests replay a stream through both paths and assert the resulting
feature vectors are bit-for-bit identical.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from mercury_rec.core.enums import EventType
from mercury_rec.features.asof import (
    FEATURE_COLUMNS,
    AsOfState,
    compute_asof_features,
    rows_from_frame,
)
from mercury_rec.features.store import (
    FEATURE_SCHEMA_VERSION,
    AsOfFeatureStore,
    FeatureStore,
    load_state,
    save_state,
)

N_USERS = 20
N_ITEMS = 30


@pytest.fixture
def catalogue() -> pd.DataFrame:
    """Per-item attributes, indexed by dense item id.

    Built FIRST and joined onto events, mirroring the real pipeline where
    price, vertical and merchant are properties of the item rather than of
    the event. Generating them per-event instead would make the event stream
    and the catalogue disagree about the same item, and the parity assertion
    below would fail for a reason that has nothing to do with the code.
    """
    rng = np.random.default_rng(11)
    return pd.DataFrame(
        {
            "price": rng.uniform(2, 80, N_ITEMS).astype("float32"),
            "vertical": rng.integers(0, 6, N_ITEMS).astype("int16"),
            "merchant_id": rng.integers(0, 12, N_ITEMS).astype("int32"),
        }
    )


@pytest.fixture
def events(catalogue: pd.DataFrame) -> pd.DataFrame:
    rng = np.random.default_rng(7)
    n = 400
    minutes = np.sort(rng.integers(0, 600, n))
    item_ids = rng.integers(0, N_ITEMS, n).astype("int32")

    return pd.DataFrame(
        {
            "user_id": rng.integers(0, N_USERS, n).astype("int32"),
            "item_id": item_ids,
            "ts": pd.Timestamp("2024-03-01", tz="UTC") + pd.to_timedelta(minutes, unit="m"),
            "event_type": rng.choice(
                [int(EventType.VIEW), int(EventType.ADD_TO_CART), int(EventType.PURCHASE)],
                n,
                p=[0.75, 0.2, 0.05],
            ).astype("int8"),
            # Joined from the catalogue, exactly as build_data does.
            "price_at_event": catalogue["price"].to_numpy()[item_ids],
            "vertical": catalogue["vertical"].to_numpy()[item_ids].astype("int8"),
            "merchant_id": catalogue["merchant_id"].to_numpy()[item_ids],
        }
    ).reset_index(drop=True)


def test_online_store_matches_offline_scan(events: pd.DataFrame, catalogue: pd.DataFrame) -> None:
    """The core parity assertion.

    The offline scan emits features for every event. The online store, given
    the same history and asked about the same (user, item) at the same instant,
    must produce the identical vector. Any divergence here is training/serving
    skew, and it would be invisible in production.
    """
    # Offline: one pass over the whole stream.
    offline = compute_asof_features(events, n_users=N_USERS, n_items=N_ITEMS)

    # Online: replay the same stream through the store, asking for each
    # pair's features immediately before observing the event - exactly the
    # sequence a live service experiences.
    store = AsOfFeatureStore(
        AsOfState(N_USERS, N_ITEMS),
        item_prices=catalogue["price"].to_numpy(dtype=np.float32),
        item_verticals=catalogue["vertical"].to_numpy(dtype=np.int16),
        item_merchants=catalogue["merchant_id"].to_numpy(dtype=np.int32),
    )

    rows = rows_from_frame(events)
    online = np.empty((len(rows), len(FEATURE_COLUMNS)), dtype=np.float32)

    index = 0
    while index < len(rows):
        # Tied timestamps must all be scored before any is observed, matching
        # the offline scan's tie handling.
        group_end = index
        while group_end < len(rows) and rows[group_end].now == rows[index].now:
            group_end += 1

        for position in range(index, group_end):
            row = rows[position]
            online[position] = store.compute_batch(row.user, np.array([row.item]), row.now)[0]

        for position in range(index, group_end):
            store.observe(rows[position])
        index = group_end

    np.testing.assert_allclose(
        offline.to_numpy(),
        online,
        rtol=1e-6,
        equal_nan=True,
        err_msg="online store diverged from the offline scan - this is training/serving skew",
    )


def test_batch_and_single_candidate_agree(events: pd.DataFrame, catalogue: pd.DataFrame) -> None:
    """Scoring many candidates at once must equal scoring them one by one.

    Serving batches 200-500 candidates per request; a batching bug would
    corrupt every ranking while leaving shapes intact.
    """
    _, state = compute_asof_features(events, n_users=N_USERS, n_items=N_ITEMS, return_state=True)
    store = AsOfFeatureStore(
        state,
        item_prices=catalogue["price"].to_numpy(dtype=np.float32),
        item_verticals=catalogue["vertical"].to_numpy(dtype=np.int16),
        item_merchants=catalogue["merchant_id"].to_numpy(dtype=np.int32),
    )

    now = int(events["ts"].max().value) + 10**9
    candidates = np.arange(N_ITEMS)

    batched = store.compute_batch(3, candidates, now)
    one_by_one = np.vstack(
        [store.compute_batch(3, np.array([item]), now)[0] for item in candidates]
    )

    np.testing.assert_allclose(batched, one_by_one, equal_nan=True)


def test_candidate_order_does_not_affect_features(
    events: pd.DataFrame, catalogue: pd.DataFrame
) -> None:
    """Row i of the output must describe item_ids[i], whatever the order."""
    _, state = compute_asof_features(events, n_users=N_USERS, n_items=N_ITEMS, return_state=True)
    store = AsOfFeatureStore(
        state,
        item_prices=catalogue["price"].to_numpy(dtype=np.float32),
        item_verticals=catalogue["vertical"].to_numpy(dtype=np.int16),
        item_merchants=catalogue["merchant_id"].to_numpy(dtype=np.int32),
    )
    now = int(events["ts"].max().value)

    forward = np.array([1, 5, 9, 12])
    reverse = forward[::-1]

    np.testing.assert_allclose(
        store.compute_batch(2, forward, now),
        store.compute_batch(2, reverse, now)[::-1],
        equal_nan=True,
    )


def test_observe_advances_state_like_the_offline_scan(
    events: pd.DataFrame, catalogue: pd.DataFrame
) -> None:
    """Events arriving at the API must update state as the scan would."""
    midpoint = len(events) // 2

    _, offline_state = compute_asof_features(
        events, n_users=N_USERS, n_items=N_ITEMS, return_state=True
    )

    _, partial_state = compute_asof_features(
        events.iloc[:midpoint], n_users=N_USERS, n_items=N_ITEMS, return_state=True
    )
    store = AsOfFeatureStore(partial_state)
    for row in rows_from_frame(events.iloc[midpoint:]):
        store.observe(row)

    np.testing.assert_array_equal(store.state.user_events, offline_state.user_events)
    np.testing.assert_array_equal(store.state.item_events, offline_state.item_events)
    np.testing.assert_array_equal(store.state.user_purchases, offline_state.user_purchases)
    assert store.state.ui_events == offline_state.ui_events


def test_store_satisfies_the_protocol(events: pd.DataFrame) -> None:
    store = AsOfFeatureStore(AsOfState(N_USERS, N_ITEMS))
    assert isinstance(store, FeatureStore)
    assert store.feature_names() == FEATURE_COLUMNS


def test_mismatched_metadata_length_is_rejected() -> None:
    """A wrong-length array would silently mis-attribute prices to items."""
    with pytest.raises(ValueError, match="expected n_items"):
        AsOfFeatureStore(AsOfState(N_USERS, N_ITEMS), item_prices=np.zeros(5, dtype=np.float32))


# --- persistence -----------------------------------------------------------


def test_state_round_trips_through_disk(events: pd.DataFrame, tmp_path: Path) -> None:
    """Serving loads the training-window state, so it must survive a round trip."""
    _, state = compute_asof_features(events, n_users=N_USERS, n_items=N_ITEMS, return_state=True)
    path = tmp_path / "state.pkl"
    save_state(state, path)
    restored = load_state(path)

    np.testing.assert_array_equal(restored.user_events, state.user_events)
    np.testing.assert_array_equal(restored.item_purchases, state.item_purchases)
    assert restored.ui_events == state.ui_events
    assert restored.user_category == state.user_category


def test_restored_state_produces_identical_features(
    events: pd.DataFrame, catalogue: pd.DataFrame, tmp_path: Path
) -> None:
    """A round trip must not perturb a single feature value."""
    _, state = compute_asof_features(events, n_users=N_USERS, n_items=N_ITEMS, return_state=True)
    path = tmp_path / "state.pkl"
    save_state(state, path)

    prices = catalogue["price"].to_numpy(dtype=np.float32)
    now = int(events["ts"].max().value)
    candidates = np.arange(N_ITEMS)

    before = AsOfFeatureStore(state, item_prices=prices).compute_batch(4, candidates, now)
    after = AsOfFeatureStore(load_state(path), item_prices=prices).compute_batch(4, candidates, now)
    np.testing.assert_allclose(before, after, equal_nan=True)


def test_schema_version_mismatch_is_rejected(events: pd.DataFrame, tmp_path: Path) -> None:
    """Serving a model features from a different schema fails silently.

    Shapes still align and scores are still produced - they are just
    meaningless. This must be a loud startup error instead.
    """
    import pickle

    _, state = compute_asof_features(events, n_users=N_USERS, n_items=N_ITEMS, return_state=True)
    path = tmp_path / "stale.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": FEATURE_SCHEMA_VERSION + 1,
                "feature_columns": list(FEATURE_COLUMNS),
                "state": state,
            },
            handle,
        )

    with pytest.raises(ValueError, match="schema version"):
        load_state(path)


def test_changed_feature_columns_are_rejected(events: pd.DataFrame, tmp_path: Path) -> None:
    import pickle

    _, state = compute_asof_features(events, n_users=N_USERS, n_items=N_ITEMS, return_state=True)
    path = tmp_path / "wrong_columns.pkl"
    with path.open("wb") as handle:
        pickle.dump(
            {
                "schema_version": FEATURE_SCHEMA_VERSION,
                "feature_columns": ["only_one_feature"],
                "state": state,
            },
            handle,
        )

    with pytest.raises(ValueError, match="different columns"):
        load_state(path)


def test_missing_state_file_names_the_fix(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError, match="mercury features build"):
        load_state(tmp_path / "absent.pkl")
