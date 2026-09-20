"""Tests for session reconstruction from inactivity gaps."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mercury_rec.data.sessionize import assign_sessions, summarise_sessions


def _frame(rows: list[tuple[int, str]]) -> pd.DataFrame:
    """Build an event frame from (user_id, iso-timestamp) rows."""
    return pd.DataFrame(
        {
            "user_id": [r[0] for r in rows],
            "ts": pd.to_datetime([r[1] for r in rows], utc=True),
            "item_id": np.arange(len(rows), dtype="int32"),
        }
    )


def test_gap_longer_than_threshold_starts_a_new_session() -> None:
    frame = _frame(
        [
            (1, "2024-01-01T10:00:00Z"),
            (1, "2024-01-01T10:20:00Z"),  # 20 min  -> same session
            (1, "2024-01-01T11:30:00Z"),  # 70 min  -> new session
        ]
    )
    result = assign_sessions(frame, inactivity_gap_minutes=30)
    assert result["session_id"].tolist() == [0, 0, 1]


def test_gap_exactly_at_threshold_stays_in_session() -> None:
    """The boundary is strictly greater-than, so 30 minutes is not a break."""
    frame = _frame([(1, "2024-01-01T10:00:00Z"), (1, "2024-01-01T10:30:00Z")])
    assert assign_sessions(frame, inactivity_gap_minutes=30)["session_id"].nunique() == 1


def test_sessions_never_span_two_users() -> None:
    """A user boundary always starts a new session, however close in time."""
    frame = _frame(
        [
            (1, "2024-01-01T10:00:00Z"),
            (2, "2024-01-01T10:00:01Z"),  # one second later, different user
        ]
    )
    result = assign_sessions(frame, inactivity_gap_minutes=30)
    assert result["session_id"].nunique() == 2


def test_session_ids_are_dense_and_zero_based() -> None:
    rng = np.random.default_rng(0)
    n = 500
    frame = pd.DataFrame(
        {
            "user_id": rng.integers(0, 20, n),
            "ts": pd.Timestamp("2024-01-01", tz="UTC")
            + pd.to_timedelta(rng.integers(0, 100_000, n), unit="s"),
            "item_id": np.arange(n, dtype="int32"),
        }
    )
    ids = assign_sessions(frame, inactivity_gap_minutes=30)["session_id"]
    assert ids.min() == 0
    assert set(ids.unique()) == set(range(ids.nunique()))


def test_oversized_sessions_are_split() -> None:
    """A bot-like run of thousands of events must not dominate sequence stats."""
    # date_range rather than repeated Timedelta addition: pandas emits a
    # DeprecationWarning for the latter, and the suite treats warnings as errors.
    stamps = pd.date_range("2024-01-01T10:00:00Z", periods=25, freq="s")
    rows = [(1, ts.isoformat()) for ts in stamps]
    result = assign_sessions(_frame(rows), inactivity_gap_minutes=30, max_events_per_session=10)

    counts = result["session_id"].value_counts()
    assert counts.max() <= 10
    assert result["session_id"].nunique() == 3  # 10 + 10 + 5


def test_summarise_sessions_reports_consistent_bounds() -> None:
    frame = _frame(
        [
            (1, "2024-01-01T10:00:00Z"),
            (1, "2024-01-01T10:10:00Z"),
            (1, "2024-01-01T12:00:00Z"),
            (2, "2024-01-01T10:00:00Z"),
        ]
    )
    sessions = summarise_sessions(assign_sessions(frame, inactivity_gap_minutes=30))

    assert (sessions["ended_at"] >= sessions["started_at"]).all()
    assert sessions["n_events"].sum() == len(frame)
    assert sessions["session_id"].is_unique


def test_empty_frame_is_rejected() -> None:
    empty = pd.DataFrame({"user_id": [], "ts": pd.to_datetime([], utc=True)})
    with pytest.raises(ValueError, match="empty"):
        assign_sessions(empty)
