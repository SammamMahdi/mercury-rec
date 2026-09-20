"""Tests for time-aware splitting and leakage prevention.

Temporal leakage is the most damaging bug available in a recommender
evaluation: it inflates every metric, and the result still looks entirely
plausible. These tests exist so that a regression fails the build instead of
quietly producing better-looking numbers.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mercury_rec.data.split import (
    assert_no_temporal_leakage,
    quantile_boundaries,
    temporal_split,
)


@pytest.fixture
def events() -> pd.DataFrame:
    """1000 events over 100 days, 50 users, each user active throughout.

    Users span the whole window deliberately: a naive per-user split would
    look fine on this fixture, so the tests below actually exercise the
    global-boundary behaviour.
    """
    rng = np.random.default_rng(0)
    n = 1000
    start = pd.Timestamp("2024-01-01", tz="UTC")
    return pd.DataFrame(
        {
            "ts": start + pd.to_timedelta(np.sort(rng.uniform(0, 100, n)), unit="D"),
            "user_id": rng.integers(0, 50, n),
            "item_id": rng.integers(0, 200, n),
        }
    )


def test_windows_are_strictly_ordered_in_time(events: pd.DataFrame) -> None:
    """The core guarantee: no training event may occur after a test event."""
    split = temporal_split(events, require_train_history=False)

    assert split.train["ts"].max() <= split.validation["ts"].min()
    assert split.validation["ts"].max() <= split.test["ts"].min()
    assert split.train["ts"].max() <= split.test["ts"].min()


def test_assert_no_temporal_leakage_accepts_a_clean_split(events: pd.DataFrame) -> None:
    assert_no_temporal_leakage(temporal_split(events, require_train_history=False))


def test_assert_no_temporal_leakage_catches_a_dirty_split(events: pd.DataFrame) -> None:
    """The guard must actually fire - a check that never fails is worthless."""
    split = temporal_split(events, require_train_history=False)
    # Smuggle the latest test event into training, simulating a regression.
    split.train = pd.concat([split.train, split.test.tail(1)], ignore_index=True)

    with pytest.raises(AssertionError, match="leakage"):
        assert_no_temporal_leakage(split)


def test_fractions_are_approximately_honoured(events: pd.DataFrame) -> None:
    split = temporal_split(
        events, train_fraction=0.70, validation_fraction=0.15, require_train_history=False
    )
    total = len(split.train) + len(split.validation) + len(split.test)
    assert total == len(events)
    assert split.summary()["train_fraction_at_boundary"] == pytest.approx(0.70, abs=0.02)
    assert split.summary()["validation_fraction_at_boundary"] == pytest.approx(0.15, abs=0.02)


def test_every_event_lands_in_exactly_one_window(events: pd.DataFrame) -> None:
    """Windows must partition the data - no gaps, no double counting."""
    split = temporal_split(events, require_train_history=False)
    combined = pd.concat([split.train, split.validation, split.test])
    assert len(combined) == len(events)
    assert not combined.index.duplicated().any()


def test_require_train_history_drops_users_absent_from_training() -> None:
    """Users with no training history cannot be personalised for.

    Scoring them in the main comparison would measure the cold-start fallback
    while appearing to measure the personalised models.
    """
    start = pd.Timestamp("2024-01-01", tz="UTC")
    frame = pd.DataFrame(
        {
            # user 1 appears only at the end, so it is absent from training.
            "ts": start + pd.to_timedelta([0, 1, 2, 3, 4, 5, 6, 7, 98, 99], unit="D"),
            "user_id": [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
            "item_id": list(range(10)),
        }
    )
    kept = temporal_split(frame, require_train_history=True)
    dropped = temporal_split(frame, require_train_history=False)

    assert 1 not in set(kept.test["user_id"])
    assert 1 in set(dropped.test["user_id"])


def test_rejects_fractions_that_leave_no_test_window(events: pd.DataFrame) -> None:
    with pytest.raises(ValueError, match="must be < 1"):
        quantile_boundaries(events, train_fraction=0.8, validation_fraction=0.3)


@pytest.mark.parametrize("train_fraction", [0.0, 1.0, -0.1, 1.5])
def test_rejects_out_of_range_fractions(events: pd.DataFrame, train_fraction: float) -> None:
    with pytest.raises(ValueError, match="train_fraction"):
        quantile_boundaries(events, train_fraction=train_fraction, validation_fraction=0.15)


def test_split_is_deterministic(events: pd.DataFrame) -> None:
    """Reproducibility: the same input must yield byte-identical windows."""
    first = temporal_split(events, require_train_history=False)
    second = temporal_split(events, require_train_history=False)
    pd.testing.assert_frame_equal(first.train, second.train)
    pd.testing.assert_frame_equal(first.test, second.test)
    assert first.boundaries == second.boundaries
