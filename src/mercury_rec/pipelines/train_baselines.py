"""Train and evaluate every retrieval model under one protocol.

This is what produces the comparison table. All models are fitted on the same
training window and scored by the same harness against the same users, so a
difference in the numbers is a difference in the model rather than in how it
was measured.

Results are written to ``artifacts/evaluation/<preset>/results.json`` with the
dataset hash, the evaluated population and the environment recorded alongside.
Nothing in the README or the frontend may quote a number that is not in that
file.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mercury_rec.core.logging import get_logger
from mercury_rec.evaluation.harness import EvaluationData, evaluate_model, load_split, prepare
from mercury_rec.evaluation.metrics import DEFAULT_K_VALUES
from mercury_rec.models.base import Recommender
from mercury_rec.models.item_cf import ItemCFRecommender
from mercury_rec.models.mf import BPRRecommender
from mercury_rec.models.popularity import (
    ContextualPopularityRecommender,
    PopularityRecommender,
    TrendingRecommender,
)

logger = get_logger(__name__)


@dataclass(slots=True)
class BaselineResults:
    output_path: Path
    payload: dict[str, Any]


def _build_models(n_users: int, n_items: int, *, quick: bool) -> list[Recommender]:
    """Instantiate the model line-up.

    ``quick`` shortens BPR training for smoke tests; it never changes the
    protocol, only the epoch budget, and the value used is recorded in the
    results so a quick run is never mistaken for a full one.
    """
    return [
        PopularityRecommender(n_users, n_items),
        ContextualPopularityRecommender(n_users, n_items),
        TrendingRecommender(n_users, n_items),
        ItemCFRecommender(n_users, n_items),
        BPRRecommender(n_users, n_items, epochs=5 if quick else 30),
    ]


def run_baselines(
    *,
    preset: str = "full",
    split: str = "test",
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    max_users: int | None = None,
    quick: bool = False,
) -> BaselineResults:
    """Fit and evaluate every baseline; persist the comparison."""
    started = time.perf_counter()
    root = data_dir or Path("data")
    processed_dir = root / "processed" / preset
    output_dir = (artifacts_dir or Path("artifacts")) / "evaluation" / preset
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_meta = json.loads((processed_dir / "METADATA.json").read_text(encoding="utf-8"))
    n_users = int(dataset_meta["counts"]["users"])
    n_items = int(dataset_meta["counts"]["items"])

    train = load_split(processed_dir, "train")
    evaluation = load_split(processed_dir, split)
    items = pd.read_parquet(processed_dir / "items.parquet")

    data: EvaluationData = prepare(
        train,
        evaluation,
        n_users=n_users,
        n_items=n_items,
        items=items,
        max_users=max_users,
    )

    rows: list[dict[str, Any]] = []
    for model in _build_models(n_users, n_items, quick=quick):
        logger.info("baselines.fit", model=model.name)
        fit_result = model.fit(train)
        result, per_user_ms = evaluate_model(model, data, k_values=DEFAULT_K_VALUES)

        rows.append(
            {
                **result.as_dict(),
                "train_seconds": round(fit_result.train_seconds, 3),
                "scoring_ms_per_user": round(per_user_ms, 4),
                "params": fit_result.params,
                "fit_extra": fit_result.extra,
            }
        )

    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "preset": preset,
        "split": split,
        "dataset_hash": dataset_meta.get("dataset_hash"),
        "n_users": n_users,
        "n_items": n_items,
        "train_events": len(train),
        "evaluation_events": len(evaluation),
        "users_scored": len(data.ground_truth),
        "k_values": list(DEFAULT_K_VALUES),
        "quick_mode": quick,
        "protocol": {
            "training_history_excluded": True,
            "ground_truth": "evaluation-window items not already seen in training",
            "relevance": "binary, any intent level",
            "note": (
                "scoring_ms_per_user is OFFLINE batch scoring on this machine; "
                "it excludes feature lookup, caching, network and the API layer. "
                "Serving latency is measured separately under load."
            ),
        },
        "environment": {
            "python": platform.python_version(),
            "platform": platform.platform(),
            "processor": platform.processor(),
        },
        "results": rows,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }

    output_path = output_dir / f"results_{split}.json"
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("baselines.complete", path=str(output_path), models=len(rows))
    return BaselineResults(output_path=output_path, payload=payload)


def format_table(payload: dict[str, Any], k: int = 10) -> str:
    """Render the comparison as a fixed-width table for the terminal."""
    header = (
        f"{'model':<24}{f'R@{k}':>9}{f'NDCG@{k}':>10}{f'MAP@{k}':>9}"
        f"{f'HR@{k}':>9}{'cov':>8}{'gini':>7}{'fit_s':>8}{'ms/user':>9}"
    )
    lines = [header, "-" * len(header)]
    for row in payload["results"]:
        lines.append(
            f"{row['model']:<24}{row[f'recall@{k}']:>9.4f}{row[f'ndcg@{k}']:>10.4f}"
            f"{row[f'map@{k}']:>9.4f}{row[f'hit_rate@{k}']:>9.4f}"
            f"{row['catalog_coverage']:>8.3f}{row['gini']:>7.3f}"
            f"{row['train_seconds']:>8.1f}{row['scoring_ms_per_user']:>9.3f}"
        )
    return "\n".join(lines)


__all__ = ["BaselineResults", "format_table", "run_baselines"]
