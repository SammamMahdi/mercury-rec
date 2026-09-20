"""Feature store: one feature definition, two access paths.

The requirement is that training-time and serving-time features agree. The
usual way that breaks is not a bug in either path -- it is that there *are*
two paths. Someone writes a Spark job for training and a Python service for
serving, they agree on day one, and they drift the first time a definition
changes on one side only. The resulting training/serving skew is notoriously
hard to detect, because both systems keep working and only the model quality
degrades.

The approach here removes the possibility rather than managing it:

- :class:`~mercury_rec.features.asof.AsOfState` holds the aggregates.
- :func:`~mercury_rec.features.asof.emit_features` is the sole definition of a
  feature vector.
- Offline, the training pipeline scans history and emits per event.
- Online, this store hydrates the state and emits per candidate.

Both call the same function. A feature added to ``emit_features`` appears in
training and serving simultaneously; there is no second place to update and
therefore nothing to forget. ``tests/data/test_feature_parity.py`` asserts the
two paths produce identical vectors.

What the state does NOT do is time travel. It holds aggregates as of the
moment it was built. Serving folds in newly-arriving events via
:func:`~mercury_rec.features.asof.apply_event`, so it advances exactly as the
offline scan would have advanced it.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Protocol, runtime_checkable

import numpy as np
import pandas as pd

from mercury_rec.core.logging import get_logger
from mercury_rec.features.asof import (
    FEATURE_COLUMNS,
    N_FEATURES,
    AsOfState,
    EventRow,
    apply_event,
    emit_features,
)

logger = get_logger(__name__)

#: Bumped whenever FEATURE_COLUMNS or a feature's semantics change. Persisted
#: alongside the state and checked on load, so a model trained against one
#: feature schema can never be served features from another -- that mismatch
#: is silent and produces plausible-looking nonsense.
FEATURE_SCHEMA_VERSION = 1


@runtime_checkable
class FeatureStore(Protocol):
    """Read path for features, as the recommendation service sees it."""

    def feature_names(self) -> tuple[str, ...]:
        """Ordered feature names matching the emitted matrix columns."""
        ...

    def compute_batch(
        self,
        user_id: int,
        item_ids: np.ndarray,
        now_ns: int,
        *,
        item_metadata: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Return a ``(len(item_ids), N_FEATURES)`` float32 matrix."""
        ...


class AsOfFeatureStore:
    """A feature store backed by an in-memory :class:`AsOfState`.

    Used by both the offline pipelines and the API process. In serving the
    state is built once at startup from the persisted training-window state
    and then advanced by incoming events, which keeps request-time work to an
    array read plus an arithmetic emit per candidate -- no database round trip
    on the hot path.
    """

    def __init__(
        self,
        state: AsOfState,
        *,
        item_prices: np.ndarray | None = None,
        item_verticals: np.ndarray | None = None,
        item_merchants: np.ndarray | None = None,
    ) -> None:
        """Args:
        state: Aggregates as of the store's current point in time.
        item_prices: Per-item price, indexed by dense item id. Supplied so
            candidate scoring needs no catalogue join per request.
        item_verticals: Per-item vertical, indexed by dense item id.
        item_merchants: Per-item merchant, indexed by dense item id.
        """
        self._state = state
        n = state.n_items
        self._prices = item_prices if item_prices is not None else np.zeros(n, dtype=np.float32)
        self._verticals = (
            item_verticals if item_verticals is not None else np.full(n, -1, dtype=np.int16)
        )
        self._merchants = (
            item_merchants if item_merchants is not None else np.full(n, -1, dtype=np.int32)
        )
        for name, array in (
            ("item_prices", self._prices),
            ("item_verticals", self._verticals),
            ("item_merchants", self._merchants),
        ):
            if len(array) != n:
                raise ValueError(f"{name} has length {len(array)}, expected n_items={n}.")

    @property
    def state(self) -> AsOfState:
        return self._state

    def feature_names(self) -> tuple[str, ...]:
        return FEATURE_COLUMNS

    def compute_batch(
        self,
        user_id: int,
        item_ids: np.ndarray,
        now_ns: int,
        *,
        item_metadata: pd.DataFrame | None = None,
    ) -> np.ndarray:
        """Emit features for one user against many candidate items.

        This is the serving hot path: typically 200-500 candidates per
        request. The loop is over candidates rather than vectorised because
        the cross-entity features are dict lookups, which do not vectorise --
        and at a few hundred rows the Python overhead is tens of microseconds,
        far below the retrieval and ranking stages it feeds.

        Args:
            user_id: Dense user id.
            item_ids: Dense candidate item ids.
            now_ns: Request time in epoch nanoseconds. Features are emitted as
                of this instant.
            item_metadata: Optional override for price/vertical/merchant,
                indexed by item id. Falls back to the arrays supplied at
                construction.

        Returns:
            ``(len(item_ids), N_FEATURES)`` float32 matrix, rows aligned to
            ``item_ids``.
        """
        candidates = np.asarray(item_ids, dtype=np.int64)
        out = np.empty((len(candidates), N_FEATURES), dtype=np.float32)

        prices, verticals, merchants = self._prices, self._verticals, self._merchants
        if item_metadata is not None:
            prices = item_metadata["price"].to_numpy(dtype=np.float32)
            verticals = item_metadata["vertical"].to_numpy(dtype=np.int16)
            merchants = item_metadata["merchant_id"].to_numpy(dtype=np.int32)

        for position, item_id in enumerate(candidates):
            item = int(item_id)
            emit_features(
                self._state,
                EventRow(
                    user=user_id,
                    item=item,
                    now=now_ns,
                    price=float(prices[item]),
                    vertical=int(verticals[item]),
                    merchant=int(merchants[item]),
                ),
                out[position],
            )
        return out

    def observe(self, row: EventRow) -> None:
        """Fold a newly-observed event into the state.

        Called by the event-ingestion endpoint so that online features reflect
        behaviour that has happened since the state was built. Uses the same
        ``apply_event`` as the offline scan, so the state evolves identically.
        """
        apply_event(self._state, row)


def save_state(state: AsOfState, path: Path) -> None:
    """Persist state so serving can start from the training-window aggregates.

    Pickle is used deliberately: the payload is a handful of numpy arrays plus
    three large int->int dicts, and the dicts are the awkward part -- parquet
    has no natural representation and JSON would inflate them severalfold
    while losing int64 keys. The file is produced and consumed only by this
    repository's own pipelines and is never accepted from an untrusted source,
    which is the condition under which pickle is safe to use.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "state": state,
    }
    with path.open("wb") as handle:
        pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
    logger.info(
        "features.state.saved",
        path=str(path),
        n_users=state.n_users,
        n_items=state.n_items,
        pairs=len(state.ui_events),
        size_mb=round(path.stat().st_size / 1024**2, 1),
    )


def load_state(path: Path) -> AsOfState:
    """Load persisted state, refusing a feature-schema mismatch.

    A model trained against one feature schema served features from another
    fails silently: shapes still line up, scores are still produced, and they
    are meaningless. Checking the version here converts that into a startup
    error naming both versions.
    """
    if not path.is_file():
        raise FileNotFoundError(f"No feature state at {path}. Run `mercury features build` first.")
    with path.open("rb") as handle:
        payload = pickle.load(handle)  # noqa: S301  (self-produced artifact only)

    version = payload.get("schema_version")
    if version != FEATURE_SCHEMA_VERSION:
        raise ValueError(
            f"Feature state at {path} has schema version {version}, but this "
            f"build expects {FEATURE_SCHEMA_VERSION}. Rebuild features and "
            "retrain, or serving would silently use mismatched definitions."
        )

    stored_columns = tuple(payload.get("feature_columns", ()))
    if stored_columns != FEATURE_COLUMNS:
        missing = set(FEATURE_COLUMNS) - set(stored_columns)
        extra = set(stored_columns) - set(FEATURE_COLUMNS)
        raise ValueError(
            f"Feature state at {path} was built with different columns "
            f"(missing: {sorted(missing)}, unexpected: {sorted(extra)}). Rebuild features."
        )

    state: AsOfState = payload["state"]
    logger.info(
        "features.state.loaded", path=str(path), n_users=state.n_users, n_items=state.n_items
    )
    return state


__all__ = [
    "FEATURE_SCHEMA_VERSION",
    "AsOfFeatureStore",
    "FeatureStore",
    "load_state",
    "save_state",
]
