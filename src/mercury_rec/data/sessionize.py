"""Derive browsing sessions from the raw event stream.

Retailrocket has no session column, so sessions are reconstructed from
inactivity gaps: a user's consecutive events belong to the same session until
they go quiet for longer than a threshold.

The 30-minute default is the long-standing web-analytics convention. Using it
rather than a tuned value keeps session statistics comparable with published
session-based recommendation work instead of idiosyncratic to this project.

Sessions matter here for three reasons: they are the unit of "current
context" at serving time, they bound the recent-item sequence fed to the user
tower, and they let the reranker reason about what the user has already seen
in this visit.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from mercury_rec.core.logging import get_logger

logger = get_logger(__name__)


def assign_sessions(
    events: pd.DataFrame,
    *,
    inactivity_gap_minutes: int = 30,
    max_events_per_session: int = 500,
) -> pd.DataFrame:
    """Add a dense ``session_id`` column to a chronologically sorted frame.

    Implemented as a vectorised scan rather than a groupby-apply: at ~800k
    rows a Python-level apply per user costs tens of seconds, while sorting
    once and taking a cumulative sum over a boolean mask is milliseconds.

    Args:
        events: Must contain ``user_id`` and ``ts``.
        inactivity_gap_minutes: Silence longer than this starts a new session.
        max_events_per_session: Hard cap; a longer run is split. Guards
            against bot-like sessions of thousands of events dominating
            sequence statistics.

    Returns:
        A copy sorted by ``(user_id, ts)`` with an added ``session_id``.
    """
    if events.empty:
        raise ValueError("Cannot sessionise an empty event frame.")

    ordered = events.sort_values(["user_id", "ts"], kind="stable").reset_index(drop=True)

    user = ordered["user_id"].to_numpy()
    timestamps = ordered["ts"].to_numpy()

    new_user = np.empty(len(ordered), dtype=bool)
    new_user[0] = True
    new_user[1:] = user[1:] != user[:-1]

    gap_ns = np.zeros(len(ordered), dtype="int64")
    gap_ns[1:] = (timestamps[1:] - timestamps[:-1]).astype("timedelta64[ns]").astype("int64")
    threshold_ns = inactivity_gap_minutes * 60 * 1_000_000_000
    timed_out = gap_ns > threshold_ns

    boundary = new_user | (timed_out & ~new_user)
    session_id = np.cumsum(boundary) - 1

    ordered["session_id"] = session_id.astype("int32")

    if max_events_per_session > 0:
        ordered = _split_oversized_sessions(ordered, max_events_per_session)

    n_sessions = int(ordered["session_id"].nunique())
    logger.info(
        "sessionize.complete",
        events=len(ordered),
        sessions=n_sessions,
        mean_events_per_session=round(len(ordered) / n_sessions, 2),
        gap_minutes=inactivity_gap_minutes,
    )
    return ordered


def _split_oversized_sessions(events: pd.DataFrame, max_events: int) -> pd.DataFrame:
    """Break any session longer than ``max_events`` into fixed-size chunks."""
    position = events.groupby("session_id", sort=False).cumcount()
    chunk = position // max_events
    if not bool((chunk > 0).any()):
        return events

    oversized = int(events.loc[chunk > 0, "session_id"].nunique())
    # Re-densify: (session, chunk) pairs become the new session identity.
    combined = events["session_id"].astype("int64") * (chunk.max() + 1) + chunk
    events = events.copy()
    events["session_id"] = pd.factorize(combined)[0].astype("int32")
    logger.info("sessionize.split_oversized", sessions_split=oversized, max_events=max_events)
    return events


def summarise_sessions(events: pd.DataFrame) -> pd.DataFrame:
    """Build the per-session dimension table from sessionised events."""
    grouped = events.groupby("session_id", sort=True)
    summary = pd.DataFrame(
        {
            "session_id": grouped["session_id"].first().astype("int32"),
            "user_id": grouped["user_id"].first().astype("int32"),
            "started_at": grouped["ts"].min(),
            "ended_at": grouped["ts"].max(),
            "n_events": grouped.size().astype("int32"),
        }
    ).reset_index(drop=True)

    if "device" in events.columns:
        summary["device"] = grouped["device"].first().to_numpy()

    logger.info("sessionize.summary", sessions=len(summary))
    return summary


__all__ = ["assign_sessions", "summarise_sessions"]
