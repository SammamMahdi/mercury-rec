"""End-to-end dataset build: raw download -> validated, split parquet.

Stages, in order::

    download -> ingest -> validate -> deduplicate -> k-core -> densify ids
      -> resolve categories/availability -> augment -> sessionise
      -> split -> persist

Everything the pipeline decided is written to ``data/processed/METADATA.json``:
row counts at each stage, the split boundaries, the config hashes and the
dataset hash. Two reasons. First, the k-core filter substantially changes what
the published metrics describe, so the before/after numbers are provenance,
not logging. Second, every MLflow run records the dataset hash, so any metric
can be traced back to the exact data that produced it.

Run via::

    mercury data build              # full
    mercury data build --preset demo
"""

from __future__ import annotations

import hashlib
import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mercury_rec.config.loader import config_hash, load_config
from mercury_rec.config.schemas import DataConfig
from mercury_rec.core.enums import DataSource, EventType
from mercury_rec.core.logging import get_logger
from mercury_rec.data import augment, ingest, sessionize, split
from mercury_rec.data.download import download_retailrocket

logger = get_logger(__name__)

#: Columns whose values are generated rather than observed. Surfaced through
#: the API and the data card so a consumer can never mistake one for the other.
SYNTHESISED_COLUMNS: tuple[str, ...] = (
    "merchant_id",
    "vertical",
    "region_id",
    "device",
    "price",
    "price_at_event",
    "base_rating",
    "delivery_radius_km",
    "avg_delivery_minutes",
)

REAL_COLUMNS: tuple[str, ...] = (
    "user_id",
    "item_id",
    "ts",
    "event_type",
    "category_id",
    "available_from",
    "session_id",
)


@dataclass(slots=True)
class BuildResult:
    """Where the build wrote, and what it produced."""

    output_dir: Path
    metadata: dict[str, Any]
    elapsed_seconds: float


def _dataset_hash(parts: dict[str, Any]) -> str:
    """Stable digest of the shape-defining facts about this build.

    Derived from row counts, boundaries and config hashes rather than from the
    parquet bytes: parquet is not byte-reproducible across library versions,
    so hashing the files would report spurious changes.
    """
    payload = json.dumps(parts, sort_keys=True, default=str).encode()
    return hashlib.blake2b(payload, digest_size=8).hexdigest()


def _augmentation_config(config: DataConfig) -> augment.AugmentationConfig:
    aug = config.augmentation
    return augment.AugmentationConfig(
        seed=config.seed,
        vertical_weights=aug.verticals.weights.model_dump(),
        region_count=aug.regions.count,
        region_zipf_s=aug.regions.population_zipf_s,
        merchants_per_vertical=aug.merchants.count_per_vertical,
        merchant_rating_beta=(aug.merchants.rating_beta_a, aug.merchants.rating_beta_b),
        delivery_radius_km=(
            aug.merchants.delivery_radius_km.min,
            aug.merchants.delivery_radius_km.max,
        ),
        delivery_minutes=(aug.merchants.delivery_minutes.min, aug.merchants.delivery_minutes.max),
        price_lognormal={k: v.model_dump() for k, v in aug.prices.lognormal.items()},
        min_price=aug.prices.min_price,
    )


def _mark_repeat_purchases(events: pd.DataFrame) -> pd.DataFrame:
    """Flag purchases of an item the user has bought before.

    Repeat purchase is the strongest preference signal available and is
    weighted separately in ``configs/events.yaml``, so it has to be derived
    rather than assumed. The cumulative count is taken over time-ordered rows,
    so a row only ever sees the user's *earlier* purchases - deriving it any
    other way would leak the future into a training label.
    """
    events = events.sort_values("ts", kind="stable").reset_index(drop=True)
    is_purchase = events["event_type"].to_numpy() == int(EventType.PURCHASE)

    repeat = np.zeros(len(events), dtype=bool)
    if is_purchase.any():
        purchases = events.loc[is_purchase, ["user_id", "item_id"]]
        prior = purchases.groupby(["user_id", "item_id"], sort=False).cumcount()
        repeat[np.flatnonzero(is_purchase)] = (prior > 0).to_numpy()

    events["is_repeat"] = repeat
    events.loc[repeat, "event_type"] = int(EventType.REPEAT_PURCHASE)
    logger.info("build.repeat_purchases", count=int(repeat.sum()))
    return events


def build_dataset(
    *,
    preset: str = "full",
    data_dir: Path | None = None,
    force_download: bool = False,
) -> BuildResult:
    """Run the full data pipeline and persist the processed dataset."""
    started = time.perf_counter()
    config = load_config("data", DataConfig)
    preset_config = config.preset(preset)

    root = data_dir or Path("data")
    raw_dir, processed_dir = root / "raw", root / "processed" / preset
    processed_dir.mkdir(parents=True, exist_ok=True)

    logger.info("build.start", preset=preset, output=str(processed_dir))

    # --- 1. acquire ------------------------------------------------------
    download_retailrocket(raw_dir, force=force_download)

    # --- 2. ingest and clean --------------------------------------------
    events = ingest.read_events(raw_dir)
    stats = ingest.IngestStats(
        raw_events=len(events),
        raw_users=int(events["user_id"].nunique()),
        raw_items=int(events["item_id"].nunique()),
    )

    events = ingest.deduplicate(events)
    stats.after_dedup = len(events)

    if config.k_core.enabled:
        events, iterations = ingest.apply_k_core(
            events,
            min_user_interactions=preset_config.min_user_interactions,
            min_item_interactions=preset_config.min_item_interactions,
            max_iterations=config.k_core.max_iterations,
        )
        stats.kcore_iterations = iterations

    # A demo preset is a SUBSET of the full build, not a differently-built
    # dataset: the most active users are kept so the small dataset still has
    # enough signal per user to train on.
    if preset_config.max_users is not None:
        keep = events["user_id"].value_counts().nlargest(preset_config.max_users).index
        events = events[events["user_id"].isin(keep)]
    if preset_config.max_items is not None:
        keep_items = events["item_id"].value_counts().nlargest(preset_config.max_items).index
        events = events[events["item_id"].isin(keep_items)]
    if preset_config.max_users is not None or preset_config.max_items is not None:
        logger.info("build.preset_capped", preset=preset, events=len(events))

    events, user_map, item_map = ingest.densify_ids(events)

    # --- 3. item attributes ---------------------------------------------
    properties = ingest.read_item_properties(raw_dir)
    categories = ingest.latest_categories(properties)
    availability = ingest.availability_windows(properties)
    tree = ingest.read_category_tree(raw_dir)
    roots = ingest.resolve_root_categories(tree)
    del properties  # ~150 MB; the pipeline is memory-bound

    items = item_map.merge(
        categories.rename(columns={"item_id": "source_item_id"}), on="source_item_id", how="left"
    )
    items = items.merge(
        availability.rename(columns={"item_id": "source_item_id"}), on="source_item_id", how="left"
    )
    items = items.merge(roots, on="category_id", how="left")

    # Items with no category row fall back to a sentinel root so vertical
    # assignment still applies; the fallback count is recorded below.
    uncategorised = int(items["root_category_id"].isna().sum())
    items["root_category_id"] = items["root_category_id"].fillna(-1).astype("int32")
    items["category_id"] = items["category_id"].fillna(-1).astype("int32")

    # --- 4. augmentation -------------------------------------------------
    aug_config = _augmentation_config(config)
    # Weight the assignment by how many items each root actually carries, so
    # the realised per-vertical item share tracks the configured weights.
    root_sizes = items.groupby("root_category_id", sort=True).size()
    root_verticals = augment.assign_verticals_to_roots(
        root_sizes.index.to_numpy().astype(np.int64),
        root_sizes.to_numpy().astype(np.int64),
        aug_config,
    )
    items = items.merge(root_verticals, on="root_category_id", how="left")
    items["vertical"] = items["vertical"].astype("int8")

    merchants = augment.build_merchants(aug_config)
    items["merchant_id"] = augment.assign_items_to_merchants(
        items["item_id"].to_numpy().astype(np.int64),
        items["vertical"].to_numpy(),
        merchants,
    )
    items["price"] = augment.assign_prices(
        items["item_id"].to_numpy().astype(np.int64), items["vertical"].to_numpy(), aug_config
    )
    items["base_rating"] = augment.assign_item_ratings(
        items["item_id"].to_numpy().astype(np.int64), aug_config
    )

    first_seen = events.groupby("item_id", sort=False)["ts"].min().rename("first_interaction")
    items = items.merge(first_seen, left_on="item_id", right_index=True, how="left")
    items["available_from"] = items["first_available"].fillna(items["first_interaction"])
    items["available_until"] = pd.NaT

    users = user_map.copy()
    user_ids = users["user_id"].to_numpy().astype(np.int64)
    users["region_id"] = augment.assign_user_regions(user_ids, aug_config)
    users["device_pref"] = augment.assign_user_devices(user_ids, aug_config)
    user_first = events.groupby("user_id", sort=False)["ts"].min().rename("signup_at")
    users = users.merge(user_first, left_on="user_id", right_index=True, how="left")

    # --- 5. attach context to events -------------------------------------
    events = events.merge(
        items[["item_id", "merchant_id", "vertical", "price"]], on="item_id", how="left"
    )
    events = events.merge(users[["user_id", "region_id", "device_pref"]], on="user_id", how="left")
    events = events.rename(columns={"device_pref": "device", "price": "price_at_event"})
    events = _mark_repeat_purchases(events)

    # --- 6. sessionise ---------------------------------------------------
    events = sessionize.assign_sessions(
        events,
        inactivity_gap_minutes=config.sessions.inactivity_gap_minutes,
        max_events_per_session=config.sessions.max_events_per_session,
    )
    sessions = sessionize.summarise_sessions(events)

    # The as-of feature engine requires chronological order, and the
    # Interactions schema asserts it.
    events = events.sort_values("ts", kind="stable").reset_index(drop=True)
    events["event_id"] = np.arange(len(events), dtype="int64")

    # --- 7. cold-start cohorts -------------------------------------------
    cold_cfg = config.augmentation.cold_start
    last_ts = events["ts"].max()
    item_cold_cut = last_ts - pd.Timedelta(days=cold_cfg.item_first_seen_within_days)
    user_cold_cut = last_ts - pd.Timedelta(days=cold_cfg.user_first_seen_within_days)
    items["is_cold"] = (items["first_interaction"] >= item_cold_cut).fillna(True)
    users["is_cold"] = (users["signup_at"] >= user_cold_cut).fillna(True)

    stats.final_events = len(events)
    stats.final_users = int(events["user_id"].nunique())
    stats.final_items = int(events["item_id"].nunique())
    stats.first_event = events["ts"].min()
    stats.last_event = events["ts"].max()
    stats.events_by_type = {
        EventType(int(k)).name: int(v)  # type: ignore[call-overload]
        for k, v in events["event_type"].value_counts().items()
    }

    # --- 8. split --------------------------------------------------------
    parts = split.temporal_split(
        events,
        train_fraction=config.split.train_fraction,
        validation_fraction=config.split.validation_fraction,
        require_train_history=config.split.require_train_history,
    )
    split.assert_no_temporal_leakage(parts)

    # --- 9. persist ------------------------------------------------------
    compression = config.output.compression
    for name, frame in (
        ("interactions_train", parts.train),
        ("interactions_validation", parts.validation),
        ("interactions_test", parts.test),
        ("items", items),
        ("users", users),
        ("merchants", merchants),
        ("sessions", sessions),
        ("user_id_map", user_map),
        ("item_id_map", item_map),
    ):
        target = processed_dir / f"{name}.parquet"
        frame.to_parquet(target, compression=compression, index=False)  # type: ignore[call-overload]
        logger.info("build.persisted", table=name, rows=len(frame), path=str(target))

    metadata: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "preset": preset,
        "source": {
            "dataset": "retailrocket/ecommerce-dataset",
            "license": "CC BY-NC-SA 4.0",
            "url": "https://www.kaggle.com/datasets/retailrocket/ecommerce-dataset",
            "provenance": DataSource.AUGMENTED.value,
            "note": (
                "Behavioural events, timestamps, item ids, categories and "
                "availability are REAL. Merchants, regions, prices, verticals, "
                "devices and delivery attributes are SYNTHESISED - see "
                "docs/data-card.md."
            ),
        },
        "columns": {"real": list(REAL_COLUMNS), "synthesised": list(SYNTHESISED_COLUMNS)},
        "ingest": stats.as_dict(),
        "split": parts.summary(),
        "counts": {
            "items": len(items),
            "users": len(users),
            "merchants": len(merchants),
            "sessions": len(sessions),
            "uncategorised_items": uncategorised,
        },
        "config_hashes": {"data": config_hash("data"), "events": config_hash("events")},
        "seed": config.seed,
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "pandas": pd.__version__,
            "numpy": np.__version__,
        },
    }
    metadata["dataset_hash"] = _dataset_hash(
        {
            "ingest": metadata["ingest"],
            "split": metadata["split"],
            "counts": metadata["counts"],
            "config_hashes": metadata["config_hashes"],
            "seed": config.seed,
        }
    )

    (processed_dir / "METADATA.json").write_text(
        json.dumps(metadata, indent=2, default=str), encoding="utf-8"
    )

    elapsed = time.perf_counter() - started
    logger.info(
        "build.complete",
        preset=preset,
        seconds=round(elapsed, 1),
        dataset_hash=metadata["dataset_hash"],
    )
    return BuildResult(output_dir=processed_dir, metadata=metadata, elapsed_seconds=elapsed)


if __name__ == "__main__":
    build_dataset()
