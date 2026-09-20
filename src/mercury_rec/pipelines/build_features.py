"""Build leakage-free feature matrices for every split.

The ordering here is the whole point and is easy to get wrong.

Features are computed in a **single chronological pass across all three
splits**, carrying state forward: train, then validation, then test. That is
not an optimisation — it is what correctness requires.

- Computing each split independently would strip validation and test rows of
  their training history. Every user would look brand new at the split
  boundary, features would be mostly undefined, and measured quality would
  collapse for a reason that has nothing to do with the models.
- Computing over the concatenated frame *without* order would let training
  rows observe test behaviour, which is the leakage the whole design prevents.

Carrying state forward gives validation and test rows the full benefit of
earlier behaviour while guaranteeing no row ever sees its own future. It is
also exactly what a deployed system experiences: it knows everything up to
now, and nothing after.

The state as of the **end of training** is persisted separately. That is the
snapshot serving starts from — a model trained on the training window must be
served features from that same window, not from state that has already
absorbed the test period.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mercury_rec.core.logging import get_logger
from mercury_rec.features.asof import FEATURE_COLUMNS, AsOfState, compute_asof_features
from mercury_rec.features.store import FEATURE_SCHEMA_VERSION, save_state

logger = get_logger(__name__)

SPLITS = ("train", "validation", "test")


@dataclass(slots=True)
class FeatureBuildResult:
    output_dir: Path
    rows_per_split: dict[str, int]
    elapsed_seconds: float
    metadata: dict[str, Any]


def _load_split(processed_dir: Path, split: str) -> pd.DataFrame:
    path = processed_dir / f"interactions_{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(
            f"Missing {path}. Run `mercury data build --preset <preset>` first."
        )
    frame = pd.read_parquet(path)
    return frame.sort_values("ts", kind="stable")


def build_features(
    *,
    preset: str = "full",
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
) -> FeatureBuildResult:
    """Compute and persist as-of features for all splits.

    Returns the output location, per-split row counts and build metadata.
    """
    started = time.perf_counter()
    root = data_dir or Path("data")
    processed_dir = root / "processed" / preset
    output_dir = (artifacts_dir or Path("artifacts")) / "features" / preset
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata_path = processed_dir / "METADATA.json"
    if not metadata_path.is_file():
        raise FileNotFoundError(f"Missing {metadata_path}. Run `mercury data build` first.")
    dataset_meta = json.loads(metadata_path.read_text(encoding="utf-8"))

    # State arrays are sized by the full id space, not by what appears in a
    # given split, so an id never falls outside the array.
    n_users = int(dataset_meta["counts"]["users"])
    n_items = int(dataset_meta["counts"]["items"])

    logger.info(
        "features.build.start",
        preset=preset,
        n_users=n_users,
        n_items=n_items,
        output=str(output_dir),
    )

    state = AsOfState(n_users, n_items)
    rows_per_split: dict[str, int] = {}
    training_state_path = output_dir / "state_train.pkl"

    for split in SPLITS:
        events = _load_split(processed_dir, split)
        if events.empty:
            logger.warning("features.build.empty_split", split=split)
            rows_per_split[split] = 0
            continue

        result = compute_asof_features(
            events, n_users=n_users, n_items=n_items, state=state, return_state=True
        )
        assert isinstance(result, tuple)
        features, state = result

        # Keys carried alongside so the ranker can join labels and group rows
        # by (user, request) without re-reading the interaction frame.
        output = pd.concat(
            [
                events[["event_id", "user_id", "item_id", "ts", "event_type"]].reset_index(
                    drop=True
                ),
                features.reset_index(drop=True),
            ],
            axis=1,
        )
        target = output_dir / f"features_{split}.parquet"
        output.to_parquet(target, compression="zstd", index=False)

        rows_per_split[split] = len(output)
        logger.info(
            "features.build.split_complete",
            split=split,
            rows=len(output),
            undefined_share=round(float(features.isna().to_numpy().mean()), 4),
        )

        # Snapshot immediately after training, before validation or test has
        # touched the state. Serving must start from exactly this.
        if split == "train":
            save_state(state, training_state_path)

    elapsed = time.perf_counter() - started
    metadata = {
        "preset": preset,
        "dataset_hash": dataset_meta.get("dataset_hash"),
        "feature_schema_version": FEATURE_SCHEMA_VERSION,
        "feature_columns": list(FEATURE_COLUMNS),
        "n_users": n_users,
        "n_items": n_items,
        "rows_per_split": rows_per_split,
        "training_state": str(training_state_path.name),
        "elapsed_seconds": round(elapsed, 2),
        "note": (
            "Computed in one chronological pass across splits, carrying state "
            "forward. No row observes an event at or after its own timestamp; "
            "see tests/data/test_leakage.py for the verification."
        ),
    }
    (output_dir / "METADATA.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    logger.info("features.build.complete", seconds=round(elapsed, 1), **rows_per_split)
    return FeatureBuildResult(
        output_dir=output_dir,
        rows_per_split=rows_per_split,
        elapsed_seconds=elapsed,
        metadata=metadata,
    )


def feature_matrix(frame: pd.DataFrame) -> np.ndarray:
    """Extract the feature columns as a float32 matrix, in canonical order."""
    missing = [name for name in FEATURE_COLUMNS if name not in frame.columns]
    if missing:
        raise ValueError(f"Feature frame is missing columns: {missing}")
    return frame[list(FEATURE_COLUMNS)].to_numpy(dtype=np.float32)


__all__ = ["SPLITS", "FeatureBuildResult", "build_features", "feature_matrix"]
