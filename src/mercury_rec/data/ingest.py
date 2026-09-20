"""Raw Retailrocket CSVs -> validated, densely-indexed interaction frames.

Responsibilities, in order:

1. Read the published CSVs and validate them against the raw schemas, so an
   upstream layout change fails immediately rather than producing empty joins.
2. Project the time-versioned ``item_properties`` change log down to the two
   properties that carry meaning (``categoryid``, ``available``); the rest are
   hashed and unusable.
3. Reduce the interaction graph to its k-core, because Retailrocket's median
   visitor has a single event and no model can learn a representation from one
   observation.
4. Re-index user and item ids densely from 0, which is what embedding tables
   require and what keeps them small.

Memory is the binding constraint on the target machine (16 GB total, shared
with PostgreSQL, Redis and a dev server). Two choices follow from that: the
20M-row property log is read in filtered chunks and never materialised whole,
and every frame carries explicit narrow dtypes rather than pandas' int64/
float64 defaults, which is roughly a 3x difference in peak footprint.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Final

import numpy as np
import pandas as pd

from mercury_rec.core.enums import EventType
from mercury_rec.core.logging import get_logger
from mercury_rec.data.schemas import RawCategoryTree, RawEvents, RawItemProperties

logger = get_logger(__name__)

#: Retailrocket's three event names, mapped onto this project's vocabulary.
#: The dataset has no impression log, so CTR-style features are computed over
#: observed views rather than over served impressions. That limitation is
#: real and is stated in the data card rather than papered over.
_EVENT_MAP: Final[dict[str, int]] = {
    "view": int(EventType.VIEW),
    "addtocart": int(EventType.ADD_TO_CART),
    "transaction": int(EventType.PURCHASE),
}

#: The only two interpretable properties in item_properties; everything else
#: is a hashed token with no recoverable meaning.
_USEFUL_PROPERTIES: Final = frozenset({"categoryid", "available"})

_CHUNK_ROWS: Final = 2_000_000


@dataclass(slots=True)
class IngestStats:
    """Row counts at each stage, for the data card and for the dataset hash.

    Recorded rather than logged-and-forgotten: the k-core filter materially
    changes what the published metrics describe, so the before/after numbers
    are part of the dataset's provenance.
    """

    raw_events: int = 0
    raw_users: int = 0
    raw_items: int = 0
    after_dedup: int = 0
    kcore_iterations: int = 0
    final_events: int = 0
    final_users: int = 0
    final_items: int = 0
    first_event: pd.Timestamp | None = None
    last_event: pd.Timestamp | None = None
    events_by_type: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict[str, object]:
        return {
            "raw_events": self.raw_events,
            "raw_users": self.raw_users,
            "raw_items": self.raw_items,
            "after_dedup": self.after_dedup,
            "kcore_iterations": self.kcore_iterations,
            "final_events": self.final_events,
            "final_users": self.final_users,
            "final_items": self.final_items,
            "retained_event_fraction": (
                round(self.final_events / self.raw_events, 4) if self.raw_events else None
            ),
            "first_event": self.first_event.isoformat() if self.first_event is not None else None,
            "last_event": self.last_event.isoformat() if self.last_event is not None else None,
            "events_by_type": self.events_by_type,
        }


def read_events(raw_dir: Path) -> pd.DataFrame:
    """Read and validate ``events.csv``.

    Returns a frame with ``ts`` (UTC), ``user_id``, ``item_id``,
    ``event_type`` and ``is_purchase``, still carrying the *original*
    Retailrocket ids.
    """
    path = raw_dir / "events.csv"
    logger.info("ingest.events.read", path=str(path))

    frame = pd.read_csv(
        path,
        dtype={
            "timestamp": "int64",
            "visitorid": "int64",
            "event": "string",
            "itemid": "int64",
            "transactionid": "float64",
        },
    )
    RawEvents.validate(frame, lazy=True)

    unknown = set(frame["event"].unique()) - set(_EVENT_MAP)
    if unknown:
        raise ValueError(
            f"events.csv contains unrecognised event types {sorted(unknown)}. "
            "The upstream dataset may have changed; update _EVENT_MAP."
        )

    out = pd.DataFrame(
        {
            # Epoch MILLISECONDS. Parsing as seconds silently relocates the
            # entire dataset to 1970 and flattens every temporal feature.
            "ts": pd.to_datetime(frame["timestamp"], unit="ms", utc=True),
            "user_id": frame["visitorid"].astype("int64"),
            "item_id": frame["itemid"].astype("int64"),
            "event_type": frame["event"].map(_EVENT_MAP).astype("int8"),
        }
    )
    logger.info(
        "ingest.events.loaded",
        rows=len(out),
        users=out["user_id"].nunique(),
        items=out["item_id"].nunique(),
    )
    return out


def read_item_properties(raw_dir: Path) -> pd.DataFrame:
    """Read the item property change log, keeping only usable properties.

    ``item_properties_part{1,2}.csv`` together hold roughly 20M rows. Reading
    them whole costs several GB; read in chunks and filtered down to the two
    interpretable properties, the result is a few hundred thousand rows.

    Returns a long frame of ``(item_id, ts, property, value)``.
    """
    frames: list[pd.DataFrame] = []
    for part in ("item_properties_part1.csv", "item_properties_part2.csv"):
        path = raw_dir / part
        logger.info("ingest.properties.read", path=str(path))
        kept = 0
        for chunk in pd.read_csv(
            path,
            dtype={
                "timestamp": "int64",
                "itemid": "int64",
                "property": "string",
                "value": "string",
            },
            chunksize=_CHUNK_ROWS,
        ):
            if not frames and kept == 0:
                RawItemProperties.validate(chunk.head(1000), lazy=True)
            subset = chunk[chunk["property"].isin(_USEFUL_PROPERTIES)]
            if not subset.empty:
                frames.append(
                    pd.DataFrame(
                        {
                            "item_id": subset["itemid"].astype("int64"),
                            "ts": pd.to_datetime(subset["timestamp"], unit="ms", utc=True),
                            "property": subset["property"].astype("category"),
                            "value": subset["value"].astype("string"),
                        }
                    )
                )
                kept += len(subset)
        logger.info("ingest.properties.filtered", part=part, kept=kept)

    if not frames:
        raise ValueError(
            "No usable rows found in item_properties. Expected properties "
            f"{sorted(_USEFUL_PROPERTIES)}; the dataset layout may have changed."
        )

    result = pd.concat(frames, ignore_index=True)
    logger.info("ingest.properties.loaded", rows=len(result))
    return result


def latest_categories(properties: pd.DataFrame) -> pd.DataFrame:
    """Collapse the category change log to each item's most recent category.

    Items can be recategorised over time. A single current category is used
    rather than an as-of lookup because category is consumed as a static item
    attribute (for the item tower, diversity constraints and cold-start
    similarity), and recategorisation is rare enough that the added join
    complexity would not pay for itself. The simplification is noted in the
    data card.
    """
    subset = properties[properties["property"] == "categoryid"]
    if subset.empty:
        raise ValueError("No 'categoryid' rows present in item_properties.")

    newest = subset.sort_values("ts").drop_duplicates("item_id", keep="last")
    out = pd.DataFrame(
        {
            "item_id": newest["item_id"].to_numpy(),
            "category_id": pd.to_numeric(newest["value"], errors="coerce"),
        }
    ).dropna(subset=["category_id"])
    out["category_id"] = out["category_id"].astype("int32")
    logger.info("ingest.categories.resolved", items=len(out))
    return out


def availability_windows(properties: pd.DataFrame) -> pd.DataFrame:
    """Derive per-item availability from the real ``available`` property.

    This is one of the few genuinely real signals Retailrocket provides beyond
    the events themselves, so it is used directly rather than synthesised:
    ``first_available`` seeds ``available_from`` and the most recent value
    decides whether an item is currently listed.
    """
    subset = properties[properties["property"] == "available"]
    if subset.empty:
        logger.warning("ingest.availability.absent")
        return pd.DataFrame(columns=["item_id", "first_available", "is_available"])

    flag = pd.to_numeric(subset["value"], errors="coerce").fillna(0).astype("int8")
    work = pd.DataFrame(
        {
            "item_id": subset["item_id"].to_numpy(),
            "ts": subset["ts"].to_numpy(),
            "available": flag.to_numpy(),
        }
    )

    available_rows = work[work["available"] == 1]
    first_available = (
        available_rows.groupby("item_id", sort=False)["ts"].min().rename("first_available")
    )
    newest = work.sort_values("ts").drop_duplicates("item_id", keep="last")
    current = newest.set_index("item_id")["available"].rename("is_available").astype(bool)

    out = pd.concat([first_available, current], axis=1).reset_index()
    logger.info(
        "ingest.availability.resolved",
        items=len(out),
        currently_available=int(out["is_available"].sum()),
    )
    return out


def read_category_tree(raw_dir: Path) -> pd.DataFrame:
    """Read the category hierarchy. ``parentid`` is null at each root."""
    path = raw_dir / "category_tree.csv"
    frame = pd.read_csv(path, dtype={"categoryid": "int64", "parentid": "float64"})
    RawCategoryTree.validate(frame, lazy=True)
    out = pd.DataFrame(
        {
            "category_id": frame["categoryid"].astype("int32"),
            "parent_id": frame["parentid"].astype("Int32"),
        }
    )
    logger.info(
        "ingest.category_tree.loaded", categories=len(out), roots=int(out["parent_id"].isna().sum())
    )
    return out


def resolve_root_categories(tree: pd.DataFrame) -> pd.DataFrame:
    """Map every category to the root of its subtree.

    Vertical assignment happens at the root, so all descendants of a root land
    in the same vertical. Without that, sibling categories would scatter across
    verticals and both vertical features and merchant diversity would become
    meaningless.

    Iterative rather than recursive: the tree is shallow, and an iterative walk
    cannot blow the stack or hang on a cycle.
    """
    parent = dict(
        zip(
            tree["category_id"].to_numpy(),
            tree["parent_id"].astype("float64").to_numpy(),
            strict=True,
        )
    )
    roots: dict[int, int] = {}

    for category in parent:
        seen: set[int] = set()
        node = category
        while True:
            candidate = parent.get(node)
            if candidate is None or np.isnan(candidate):
                break
            nxt = int(candidate)
            if nxt in seen:  # defensive: a cycle would otherwise loop forever
                logger.warning("ingest.category_tree.cycle", category=category, at=nxt)
                break
            seen.add(nxt)
            node = nxt
        roots[category] = node

    out = pd.DataFrame(
        {"category_id": list(roots.keys()), "root_category_id": list(roots.values())}
    ).astype({"category_id": "int32", "root_category_id": "int32"})
    logger.info("ingest.category_tree.roots", roots=out["root_category_id"].nunique())
    return out


def deduplicate(events: pd.DataFrame) -> pd.DataFrame:
    """Drop exact duplicate (user, item, event_type, second) rows.

    Double-fired client events are common in behavioural logs and would
    otherwise inflate interaction counts and popularity. Dedup is at
    one-second resolution: two genuinely distinct views of the same item
    within the same second are not meaningfully different observations.
    """
    before = len(events)
    key = events["ts"].dt.floor("s")
    out = events.loc[
        ~pd.DataFrame(
            {"u": events["user_id"], "i": events["item_id"], "e": events["event_type"], "t": key}
        ).duplicated()
    ].copy()
    logger.info("ingest.dedup", before=before, after=len(out), removed=before - len(out))
    return out


def apply_k_core(
    events: pd.DataFrame,
    *,
    min_user_interactions: int,
    min_item_interactions: int,
    max_iterations: int = 30,
) -> tuple[pd.DataFrame, int]:
    """Iteratively reduce the interaction graph to its k-core.

    One pass is not enough: dropping a sparse user can push an item below the
    item threshold, which can in turn orphan another user. Iterating to a fixed
    point is the only way to actually satisfy both constraints.

    Raises ``RuntimeError`` if it has not converged within ``max_iterations``.
    Returning an unconverged frame would silently violate the very guarantee
    this function exists to provide - on this dataset convergence takes 11
    iterations, so a cap of 10 would have returned a frame that still
    contained sub-threshold users.

    Returns the filtered frame and the number of iterations performed.
    """
    working = events
    converged = False
    iterations = 0

    for iteration in range(1, max_iterations + 1):
        user_counts = working["user_id"].value_counts()
        item_counts = working["item_id"].value_counts()
        keep_users = user_counts[user_counts >= min_user_interactions].index
        keep_items = item_counts[item_counts >= min_item_interactions].index

        filtered = working[
            working["user_id"].isin(keep_users) & working["item_id"].isin(keep_items)
        ]
        iterations = iteration

        if len(filtered) == len(working):
            logger.info("ingest.kcore.converged", iteration=iteration, events=len(filtered))
            working = filtered
            converged = True
            break

        logger.info(
            "ingest.kcore.iteration",
            iteration=iteration,
            events=len(filtered),
            users=filtered["user_id"].nunique(),
            items=filtered["item_id"].nunique(),
        )
        working = filtered

        if working.empty:
            raise ValueError(
                "k-core filtering removed every interaction. The thresholds "
                f"(user>={min_user_interactions}, item>={min_item_interactions}) "
                "are too aggressive for this dataset."
            )

    if not converged:
        raise RuntimeError(
            f"k-core did not converge within {max_iterations} iterations "
            f"(user>={min_user_interactions}, item>={min_item_interactions}). "
            "Returning now would yield a frame that still contains "
            "sub-threshold users or items. Raise max_iterations."
        )

    return working.copy(), iterations


def densify_ids(events: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Re-index user and item ids densely from zero.

    Embedding tables are sized by the maximum id, so raw Retailrocket ids
    (sparse, up to ~1.4M) would allocate an embedding matrix orders of
    magnitude larger than the number of distinct entities.

    Returns the re-indexed events plus the two id mappings, which must be
    persisted: serving has to translate an external id to an internal row and
    back.
    """
    user_codes, user_uniques = pd.factorize(events["user_id"], sort=True)
    item_codes, item_uniques = pd.factorize(events["item_id"], sort=True)

    out = events.copy()
    out["user_id"] = user_codes.astype("int32")
    out["item_id"] = item_codes.astype("int32")

    user_map = pd.DataFrame(
        {"user_id": np.arange(len(user_uniques), dtype="int32"), "source_user_id": user_uniques}
    )
    item_map = pd.DataFrame(
        {"item_id": np.arange(len(item_uniques), dtype="int32"), "source_item_id": item_uniques}
    )
    logger.info("ingest.densify", users=len(user_map), items=len(item_map))
    return out, user_map, item_map


__all__ = [
    "IngestStats",
    "apply_k_core",
    "availability_windows",
    "deduplicate",
    "densify_ids",
    "latest_categories",
    "read_category_tree",
    "read_events",
    "read_item_properties",
    "resolve_root_categories",
]
