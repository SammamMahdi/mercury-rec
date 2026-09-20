"""Compare feature distributions between two windows and persist the result.

The honest framing matters here. This project has no production traffic, so
there is no "yesterday's live data" to compare against. What it does have is a
genuine time-ordered split: the training window and the held-out window that
follows it, drawn from real Retailrocket behaviour over real calendar time.

Comparing those two is not a simulation of drift detection - it is drift
detection, run over the only later data that exists. Whatever it finds is a
real property of the dataset, and it is worth knowing before blaming a model
for a gap that its inputs explain.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mercury_rec.core.logging import get_logger
from mercury_rec.features.asof import FEATURE_COLUMNS
from mercury_rec.monitoring.drift import PSI_MAJOR, PSI_MINOR, detect_drift
from mercury_rec.monitoring.metrics import set_drift_score

logger = get_logger(__name__)


@dataclass(frozen=True, slots=True)
class DriftReport:
    """Where the report landed and what it concluded."""

    output_path: Path
    n_features: int
    n_major: int
    n_minor: int


def monitor_drift(
    *,
    preset: str = "full",
    reference_split: str = "train",
    current_split: str = "test",
    artifacts_dir: Path | None = None,
    bins: int = 10,
    max_rows: int = 250_000,
) -> DriftReport:
    """Compute drift between two feature windows and write the report.

    Args:
        preset: Dataset preset.
        reference_split: The window treated as the baseline.
        current_split: The window compared against it.
        artifacts_dir: Root of ``artifacts/``.
        bins: Quantile bins for PSI and JS.
        max_rows: Cap per window. The statistics converge long before the full
            frame is consumed, and holding two 500k-row frames alongside the
            models is the difference between fitting in memory and not.

    Raises:
        FileNotFoundError: If either feature window is missing.
    """
    artifacts = artifacts_dir or Path("artifacts")
    feature_dir = artifacts / "features" / preset

    reference_path = feature_dir / f"features_{reference_split}.parquet"
    current_path = feature_dir / f"features_{current_split}.parquet"
    for path in (reference_path, current_path):
        if not path.is_file():
            raise FileNotFoundError(
                f"No feature window at {path}. Run `mercury features build` first."
            )

    columns = list(FEATURE_COLUMNS)
    reference = pd.read_parquet(reference_path, columns=columns)
    current = pd.read_parquet(current_path, columns=columns)

    # Take the TAIL of the reference window, not a random sample: the state a
    # model is deployed with is the state at the end of training, and a
    # uniform sample would compare the current window against the whole of
    # history instead.
    if len(reference) > max_rows:
        reference = reference.tail(max_rows)
    if len(current) > max_rows:
        current = current.head(max_rows)

    results = detect_drift(reference, current, features=columns, bins=bins)

    # Publish to the gauge that has existed unused until now, so a scrape
    # carries the same numbers this report does.
    for entry in results:
        set_drift_score(entry.feature, entry.psi)

    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "preset": preset,
        "reference_split": reference_split,
        "current_split": current_split,
        "reference_rows": len(reference),
        "current_rows": len(current),
        "bins": bins,
        "thresholds": {"minor": PSI_MINOR, "major": PSI_MAJOR},
        "note": (
            "Drift is measured between the training window and the held-out "
            "window that follows it in calendar time. There is no production "
            "traffic behind this system, so this is the only genuine later "
            "data available - and it is real data, not simulated drift."
        ),
        "interpretation": (
            "PSI below 0.10 is conventionally read as stable, 0.10-0.25 as a "
            "minor shift and above 0.25 as a major one. Those thresholds are "
            "an industry rule of thumb, not a property of this dataset. A "
            "shifted feature is a prompt to investigate, not evidence that "
            "the model degraded."
        ),
        "expected_drift": (
            "Time-indexed features drift here BY CONSTRUCTION. The split is "
            "chronological, so users in the later window are necessarily "
            "older (user_tenure_days) and items have necessarily existed "
            "longer (item_age_days). A large PSI on those is the split "
            "working, not a fault. The features worth reading as signal are "
            "the ones with no mechanical reason to move - affinities, "
            "conversion rates and price levels."
        ),
        "features": [entry.as_dict() for entry in results],
    }

    output_dir = artifacts / "monitoring" / preset
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "drift.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")

    n_major = sum(1 for entry in results if entry.severity == "major")
    n_minor = sum(1 for entry in results if entry.severity == "minor")
    logger.info(
        "drift.report_written",
        path=str(output_path),
        features=len(results),
        major=n_major,
        minor=n_minor,
    )
    return DriftReport(
        output_path=output_path,
        n_features=len(results),
        n_major=n_major,
        n_minor=n_minor,
    )


__all__ = ["DriftReport", "monitor_drift"]
