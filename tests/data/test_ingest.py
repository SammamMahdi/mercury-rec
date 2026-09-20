"""Tests for ingestion: deduplication, k-core filtering and id densification."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from mercury_rec.data.ingest import apply_k_core, deduplicate, densify_ids, resolve_root_categories


def _events(rows: list[tuple[int, int, int, str]]) -> pd.DataFrame:
    """Build an event frame from (user, item, event_type, iso-timestamp) rows."""
    return pd.DataFrame(
        {
            "user_id": [r[0] for r in rows],
            "item_id": [r[1] for r in rows],
            "event_type": np.array([r[2] for r in rows], dtype="int8"),
            "ts": pd.to_datetime([r[3] for r in rows], utc=True),
        }
    )


# --- deduplication ---------------------------------------------------------


def test_deduplicate_collapses_same_second_duplicates() -> None:
    """Double-fired client events would otherwise inflate popularity."""
    frame = _events(
        [
            (1, 10, 3, "2024-01-01T00:00:00.100Z"),
            (1, 10, 3, "2024-01-01T00:00:00.400Z"),  # same user/item/type/second
            (1, 10, 3, "2024-01-01T00:00:01.000Z"),  # next second - distinct
        ]
    )
    assert len(deduplicate(frame)) == 2


def test_deduplicate_keeps_different_event_types() -> None:
    """A view and an add-to-cart in the same second are distinct signals."""
    frame = _events(
        [
            (1, 10, 3, "2024-01-01T00:00:00Z"),
            (1, 10, 6, "2024-01-01T00:00:00Z"),
        ]
    )
    assert len(deduplicate(frame)) == 2


# --- k-core ----------------------------------------------------------------


def test_k_core_enforces_both_thresholds() -> None:
    """Every surviving user and item must meet its minimum."""
    rng = np.random.default_rng(0)
    n = 4000
    frame = pd.DataFrame(
        {
            "user_id": rng.integers(0, 300, n),
            "item_id": rng.integers(0, 200, n),
            "event_type": np.full(n, 3, dtype="int8"),
            "ts": pd.to_datetime("2024-01-01", utc=True) + pd.to_timedelta(np.arange(n), unit="m"),
        }
    )
    filtered, _ = apply_k_core(frame, min_user_interactions=5, min_item_interactions=5)

    assert filtered["user_id"].value_counts().min() >= 5
    assert filtered["item_id"].value_counts().min() >= 5


def test_k_core_iterates_to_a_fixed_point() -> None:
    """One pass is not enough.

    Removing a sparse user can push an item below the item threshold, which can
    orphan another user in turn. The regression this guards against is a cap
    that stops early: on the real dataset convergence takes 11 iterations, so a
    limit of 10 silently returned a frame that still violated the thresholds.
    """
    rng = np.random.default_rng(7)
    n = 6000
    frame = pd.DataFrame(
        {
            "user_id": rng.integers(0, 500, n),
            "item_id": rng.integers(0, 400, n),
            "event_type": np.full(n, 3, dtype="int8"),
            "ts": pd.to_datetime("2024-01-01", utc=True) + pd.to_timedelta(np.arange(n), unit="m"),
        }
    )
    filtered, _ = apply_k_core(frame, min_user_interactions=4, min_item_interactions=4)

    # Re-applying must be a no-op if a true fixed point was reached.
    again, iterations = apply_k_core(filtered, min_user_interactions=4, min_item_interactions=4)
    assert len(again) == len(filtered)
    assert iterations == 1


def test_k_core_raises_rather_than_returning_unconverged() -> None:
    """Silently returning an unconverged frame would break the guarantee."""
    rng = np.random.default_rng(11)
    n = 5000
    frame = pd.DataFrame(
        {
            "user_id": rng.integers(0, 900, n),
            "item_id": rng.integers(0, 900, n),
            "event_type": np.full(n, 3, dtype="int8"),
            "ts": pd.to_datetime("2024-01-01", utc=True) + pd.to_timedelta(np.arange(n), unit="m"),
        }
    )
    with pytest.raises(RuntimeError, match="did not converge"):
        apply_k_core(frame, min_user_interactions=5, min_item_interactions=5, max_iterations=1)


def test_k_core_raises_when_thresholds_remove_everything() -> None:
    frame = _events([(1, 10, 3, "2024-01-01T00:00:00Z"), (2, 11, 3, "2024-01-01T00:01:00Z")])
    with pytest.raises(ValueError, match="removed every interaction"):
        apply_k_core(frame, min_user_interactions=50, min_item_interactions=50)


# --- densification ---------------------------------------------------------


def test_densify_produces_contiguous_zero_based_ids() -> None:
    """Embedding tables are sized by max id, so sparse ids waste memory."""
    frame = _events(
        [
            (500, 9000, 3, "2024-01-01T00:00:00Z"),
            (12, 44, 3, "2024-01-01T00:01:00Z"),
            (500, 44, 3, "2024-01-01T00:02:00Z"),
        ]
    )
    dense, user_map, item_map = densify_ids(frame)

    assert set(dense["user_id"]) == {0, 1}
    assert set(dense["item_id"]) == {0, 1}
    assert len(user_map) == 2
    assert len(item_map) == 2


def test_densify_mapping_round_trips() -> None:
    """Serving must translate an external id to an internal row and back."""
    frame = _events(
        [
            (500, 9000, 3, "2024-01-01T00:00:00Z"),
            (12, 44, 3, "2024-01-01T00:01:00Z"),
            (500, 44, 3, "2024-01-01T00:02:00Z"),
        ]
    )
    dense, user_map, item_map = densify_ids(frame)

    restored_users = dense["user_id"].map(user_map.set_index("user_id")["source_user_id"])
    restored_items = dense["item_id"].map(item_map.set_index("item_id")["source_item_id"])

    np.testing.assert_array_equal(restored_users.to_numpy(), frame["user_id"].to_numpy())
    np.testing.assert_array_equal(restored_items.to_numpy(), frame["item_id"].to_numpy())


# --- category tree ---------------------------------------------------------


def test_resolve_root_categories_walks_to_the_top() -> None:
    tree = pd.DataFrame(
        {
            "category_id": pd.array([1, 2, 3, 4], dtype="int32"),
            "parent_id": pd.array([None, 1, 2, None], dtype="Int32"),
        }
    )
    roots = resolve_root_categories(tree).set_index("category_id")["root_category_id"]

    assert roots[1] == 1  # already a root
    assert roots[2] == 1
    assert roots[3] == 1  # two levels up
    assert roots[4] == 4


def test_resolve_root_categories_survives_a_cycle() -> None:
    """A malformed tree must not hang the pipeline."""
    tree = pd.DataFrame(
        {
            "category_id": pd.array([1, 2], dtype="int32"),
            "parent_id": pd.array([2, 1], dtype="Int32"),
        }
    )
    roots = resolve_root_categories(tree)
    assert len(roots) == 2
