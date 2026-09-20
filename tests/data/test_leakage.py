"""The leakage proof: verify as-of features against an independent oracle.

This is the most important test in the repository. Every offline metric the
project publishes is only meaningful if the features attached to an event
reflect solely what was knowable before it. A regression here would not break
anything visibly - it would quietly *improve* every reported number, which is
the failure mode the project's honesty requirements exist to prevent.

So the fast forward-scan in :mod:`mercury_rec.features.asof` is checked
against a deliberately naive O(n^2) implementation written here, independently,
from the definition rather than from the optimised code. If the two disagree,
the optimised path is wrong.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mercury_rec.core.enums import EventType
from mercury_rec.features.asof import FEATURE_COLUMNS, compute_asof_features


@pytest.fixture
def events() -> pd.DataFrame:
    """A small, dense frame with heavy user/item reuse and tied timestamps."""
    rng = np.random.default_rng(42)
    n = 600
    n_users, n_items = 25, 40

    # Deliberately coarse timestamps so many rows share one, exercising the
    # tie-handling path.
    minutes = np.sort(rng.integers(0, 400, n))
    return pd.DataFrame(
        {
            "user_id": rng.integers(0, n_users, n).astype("int32"),
            "item_id": rng.integers(0, n_items, n).astype("int32"),
            "ts": pd.Timestamp("2024-01-01", tz="UTC") + pd.to_timedelta(minutes, unit="m"),
            "event_type": rng.choice(
                [int(EventType.VIEW), int(EventType.ADD_TO_CART), int(EventType.PURCHASE)],
                n,
                p=[0.8, 0.15, 0.05],
            ).astype("int8"),
            "price_at_event": rng.uniform(1, 100, n).astype("float32"),
            "vertical": rng.integers(0, 6, n).astype("int8"),
            "merchant_id": rng.integers(0, 15, n).astype("int32"),
        }
    ).reset_index(drop=True)


def _oracle_counts(events: pd.DataFrame) -> pd.DataFrame:
    """Brute-force reference: for each row, scan every strictly-earlier row.

    Written from the definition of "as of", not from the implementation. It is
    O(n^2) and would never be used in production - that is the point. Its only
    job is to be obviously correct.
    """
    user = events["user_id"].to_numpy()
    item = events["item_id"].to_numpy()
    ts = events["ts"].to_numpy()
    etype = events["event_type"].to_numpy()

    n = len(events)
    user_count = np.zeros(n)
    user_purchases = np.zeros(n)
    item_count = np.zeros(n)
    ui_count = np.zeros(n)
    user_distinct = np.zeros(n)

    for i in range(n):
        earlier = ts < ts[i]  # STRICTLY before: ties must not see each other
        same_user = earlier & (user == user[i])
        same_item = earlier & (item == item[i])

        user_count[i] = same_user.sum()
        user_purchases[i] = (same_user & (etype == int(EventType.PURCHASE))).sum()
        item_count[i] = same_item.sum()
        ui_count[i] = (same_user & (item == item[i])).sum()
        user_distinct[i] = len(np.unique(item[same_user]))

    return pd.DataFrame(
        {
            "user_event_count": user_count,
            "user_purchase_count": user_purchases,
            "item_event_count": item_count,
            "ui_prior_events": ui_count,
            "user_distinct_items": user_distinct,
        }
    )


def test_asof_features_match_the_brute_force_oracle(events: pd.DataFrame) -> None:
    """The optimised scan must agree exactly with the naive definition."""
    computed = compute_asof_features(events, n_users=25, n_items=40)
    oracle = _oracle_counts(events)

    for column in oracle.columns:
        np.testing.assert_allclose(
            computed[column].to_numpy(),
            oracle[column].to_numpy(),
            err_msg=f"{column} disagrees with the brute-force oracle",
        )


def test_first_event_of_a_user_has_no_history(events: pd.DataFrame) -> None:
    """A user's first event must show zero counts and undefined recency."""
    computed = compute_asof_features(events, n_users=25, n_items=40)
    joined = events.join(computed)

    first_rows = joined.sort_values("ts").drop_duplicates("user_id", keep="first")
    assert (first_rows["user_event_count"] == 0).all()
    assert (first_rows["user_purchase_count"] == 0).all()
    # No prior interaction, so "days since last" is undefined, not zero.
    assert first_rows["user_days_since_last"].isna().all()
    assert first_rows["user_conversion_rate"].isna().all()


def test_simultaneous_events_cannot_observe_each_other() -> None:
    """Ties are the subtle case.

    Two events at the identical timestamp must both be emitted against the
    state as it stood before either occurred. Applying them in row order would
    let the second see the first, which is leakage across a zero-length
    interval.
    """
    ts = pd.Timestamp("2024-01-01T12:00:00Z")
    frame = pd.DataFrame(
        {
            "user_id": np.array([0, 0, 0], dtype="int32"),
            "item_id": np.array([1, 2, 3], dtype="int32"),
            "ts": [ts, ts, ts],
            "event_type": np.array([int(EventType.VIEW)] * 3, dtype="int8"),
        }
    )
    computed = compute_asof_features(frame, n_users=1, n_items=4)

    assert (computed["user_event_count"] == 0).all(), (
        "tied events saw one another - this is leakage across a zero interval"
    )


def test_features_never_depend_on_future_rows(events: pd.DataFrame) -> None:
    """Truncating the future must not change any past row's features.

    This is the property that matters operationally: what the system computed
    in January must not change because September happened. If any feature
    shifted, some aggregate was reaching forward.
    """
    full = compute_asof_features(events, n_users=25, n_items=40)

    midpoint = len(events) // 2
    truncated = compute_asof_features(events.iloc[:midpoint], n_users=25, n_items=40)

    pd.testing.assert_frame_equal(
        full.iloc[:midpoint].reset_index(drop=True),
        truncated.reset_index(drop=True),
        check_exact=False,
        rtol=1e-6,
    )


def test_state_carries_across_calls_without_leaking(events: pd.DataFrame) -> None:
    """Validation features must use training history but not the future.

    Computing the split in two passes, carrying state forward, must give the
    same answer as one pass over everything - otherwise evaluation features
    would differ from what serving would have produced.
    """
    midpoint = len(events) // 2
    single_pass = compute_asof_features(events, n_users=25, n_items=40)

    first_half, state = compute_asof_features(
        events.iloc[:midpoint], n_users=25, n_items=40, return_state=True
    )
    second_half = compute_asof_features(events.iloc[midpoint:], n_users=25, n_items=40, state=state)

    two_pass = pd.concat([first_half, second_half])
    pd.testing.assert_frame_equal(single_pass, two_pass, check_exact=False, rtol=1e-6)


def test_unsorted_input_is_rejected(events: pd.DataFrame) -> None:
    """The guarantee depends on ordering, so it is checked, not assumed."""
    shuffled = events.sample(frac=1.0, random_state=0)
    with pytest.raises(ValueError, match="chronologically sorted"):
        compute_asof_features(shuffled, n_users=25, n_items=40)


def test_all_declared_features_are_produced(events: pd.DataFrame) -> None:
    """FEATURE_COLUMNS is the contract the ranker and serving both read."""
    computed = compute_asof_features(events, n_users=25, n_items=40)
    assert list(computed.columns) == list(FEATURE_COLUMNS)
    assert len(computed) == len(events)


def test_no_infinities_are_produced(events: pd.DataFrame) -> None:
    """Ratios must never divide by zero into inf - LightGBM handles NaN, not inf."""
    computed = compute_asof_features(events, n_users=25, n_items=40)
    assert not np.isinf(computed.to_numpy()).any()


def test_counts_are_monotonically_non_decreasing_per_user(events: pd.DataFrame) -> None:
    """A user's cumulative event count can only ever rise over time."""
    joined = events.join(compute_asof_features(events, n_users=25, n_items=40))
    for _, group in joined.sort_values("ts").groupby("user_id"):
        counts = group["user_event_count"].to_numpy()
        assert np.all(np.diff(counts) >= 0), "cumulative user count decreased"


def test_empty_input_returns_the_declared_columns() -> None:
    empty = pd.DataFrame(
        {
            "user_id": np.array([], dtype="int32"),
            "item_id": np.array([], dtype="int32"),
            "ts": pd.to_datetime([], utc=True),
            "event_type": np.array([], dtype="int8"),
        }
    )
    computed = compute_asof_features(empty, n_users=10, n_items=10)
    assert list(computed.columns) == list(FEATURE_COLUMNS)
    assert computed.empty
