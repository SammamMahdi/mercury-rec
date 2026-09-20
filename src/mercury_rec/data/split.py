"""Time-aware train/validation/test splitting.

Behavioural data is temporal. A random split lets the model train on events
that happened *after* the ones it is evaluated on, which inflates every metric
and produces a system that looks excellent offline and fails in production.
This is the single most common way recommender evaluations go wrong, so the
split here is strictly chronological and the boundaries are global timestamps:

    |------------- TRAIN -------------|--- VAL ---|--- TEST ---|
                                      t1          t2
                              -------------- time ------------->

Global rather than per-user boundaries, because that is what a deployed
retraining job actually does: fit on everything up to now, then serve the
future. A per-user split would let the model learn from user A's March
behaviour while being scored on user B's January behaviour, which no
deployment ever does.

Leakage prevention does not end here. Splitting separates the *labels*;
:mod:`mercury_rec.features` is responsible for ensuring that the *features*
attached to an event never incorporate anything at or after that event's
timestamp.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import cast

import pandas as pd

from mercury_rec.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class SplitBoundaries:
    """The two timestamps that define the three windows."""

    train_end: pd.Timestamp
    validation_end: pd.Timestamp

    def as_dict(self) -> dict[str, str]:
        return {
            "train_end": self.train_end.isoformat(),
            "validation_end": self.validation_end.isoformat(),
        }


@dataclass(slots=True)
class TemporalSplit:
    """The three windows plus the boundaries that produced them.

    ``*_events_at_boundary`` records each window's size **before**
    ``require_train_history`` filtering, and the ``*_events`` fields record it
    after. Both are kept because they answer different questions, and
    conflating them is misleading: the chronological boundaries genuinely sit
    at the configured quantiles (e.g. 70/15/15), but dropping evaluation
    events for users with no training history shrinks the later windows
    substantially - on the full Retailrocket build, a nominal 70/15/15 split
    yields roughly 90/6/4 of the *retained* events. Reporting only the second
    pair would misdescribe the split; reporting only the first would
    misdescribe what was actually evaluated.
    """

    train: pd.DataFrame
    validation: pd.DataFrame
    test: pd.DataFrame
    boundaries: SplitBoundaries
    train_events_at_boundary: int = 0
    validation_events_at_boundary: int = 0
    test_events_at_boundary: int = 0

    def summary(self) -> dict[str, object]:
        boundary_total = (
            self.train_events_at_boundary
            + self.validation_events_at_boundary
            + self.test_events_at_boundary
        )
        retained_total = len(self.train) + len(self.validation) + len(self.test)

        def _share(count: int, total: int) -> float | None:
            return round(count / total, 4) if total else None

        return {
            # Where the chronological boundaries actually fell.
            "train_fraction_at_boundary": _share(self.train_events_at_boundary, boundary_total),
            "validation_fraction_at_boundary": _share(
                self.validation_events_at_boundary, boundary_total
            ),
            "test_fraction_at_boundary": _share(self.test_events_at_boundary, boundary_total),
            # What survived require_train_history and was actually evaluated.
            "train_events": len(self.train),
            "validation_events": len(self.validation),
            "test_events": len(self.test),
            "train_fraction_retained": _share(len(self.train), retained_total),
            "validation_fraction_retained": _share(len(self.validation), retained_total),
            "test_fraction_retained": _share(len(self.test), retained_total),
            "train_users": int(self.train["user_id"].nunique()),
            "validation_users": int(self.validation["user_id"].nunique()),
            "test_users": int(self.test["user_id"].nunique()),
            **self.boundaries.as_dict(),
        }


def quantile_boundaries(
    events: pd.DataFrame,
    *,
    train_fraction: float,
    validation_fraction: float,
) -> SplitBoundaries:
    """Pick boundaries so each window holds the requested share of *events*.

    Event quantiles rather than a uniform division of the calendar: traffic is
    not uniform over time, so equal date ranges would produce very unequal
    window sizes and a test set whose width depends on seasonality.
    """
    if not 0 < train_fraction < 1:
        raise ValueError(f"train_fraction must be in (0, 1), got {train_fraction}")
    if not 0 < validation_fraction < 1:
        raise ValueError(f"validation_fraction must be in (0, 1), got {validation_fraction}")
    if train_fraction + validation_fraction >= 1:
        raise ValueError(
            f"train_fraction + validation_fraction must be < 1 "
            f"(got {train_fraction} + {validation_fraction}); no events would remain for test."
        )

    timestamps = events["ts"]
    # pandas-stubs types Series.quantile as float, but on a datetime Series it
    # returns a Timestamp. cast rather than weaken the dataclass annotation.
    return SplitBoundaries(
        train_end=cast("pd.Timestamp", timestamps.quantile(train_fraction)),
        validation_end=cast(
            "pd.Timestamp", timestamps.quantile(train_fraction + validation_fraction)
        ),
    )


def temporal_split(
    events: pd.DataFrame,
    *,
    train_fraction: float = 0.70,
    validation_fraction: float = 0.15,
    require_train_history: bool = True,
) -> TemporalSplit:
    """Split chronologically into train/validation/test.

    Args:
        events: Must contain ``ts`` and ``user_id``.
        train_fraction: Share of events in the training window.
        validation_fraction: Share in the validation window.
        require_train_history: Drop evaluation events for users absent from
            training. Such users cannot be personalised for by definition, so
            scoring them in the main comparison would measure the cold-start
            fallback while appearing to measure the personalised models. They
            are evaluated separately through the cold-start path.

    Returns:
        The three windows and the boundaries used.
    """
    boundaries = quantile_boundaries(
        events, train_fraction=train_fraction, validation_fraction=validation_fraction
    )

    train = events[events["ts"] <= boundaries.train_end]
    validation = events[
        (events["ts"] > boundaries.train_end) & (events["ts"] <= boundaries.validation_end)
    ]
    test = events[events["ts"] > boundaries.validation_end]

    if train.empty:
        raise ValueError("Training window is empty; check the split fractions.")

    at_boundary = (len(train), len(validation), len(test))

    if require_train_history:
        known_users = set(train["user_id"].unique())
        before_val, before_test = len(validation), len(test)
        validation = validation[validation["user_id"].isin(known_users)]
        test = test[test["user_id"].isin(known_users)]
        logger.info(
            "split.filter_unseen_users",
            validation_dropped=before_val - len(validation),
            test_dropped=before_test - len(test),
        )

    split = TemporalSplit(
        train=train.copy(),
        validation=validation.copy(),
        test=test.copy(),
        boundaries=boundaries,
        train_events_at_boundary=at_boundary[0],
        validation_events_at_boundary=at_boundary[1],
        test_events_at_boundary=at_boundary[2],
    )
    logger.info("split.complete", **split.summary())
    return split


def assert_no_temporal_leakage(split: TemporalSplit) -> None:
    """Verify the windows are strictly ordered in time.

    Cheap, and it converts the most damaging possible bug in this pipeline
    from a silently inflated metric into an immediate failure.
    """
    checks = (
        (split.train, split.validation, "training window extends past the start of validation"),
        (split.validation, split.test, "validation window extends past the start of test"),
        (split.train, split.test, "training window overlaps the test window"),
    )
    for earlier, later, message in checks:
        if earlier.empty or later.empty:
            continue
        if earlier["ts"].max() > later["ts"].min():
            raise AssertionError(f"Temporal leakage: the {message}.")


__all__ = [
    "SplitBoundaries",
    "TemporalSplit",
    "assert_no_temporal_leakage",
    "quantile_boundaries",
    "temporal_split",
]
