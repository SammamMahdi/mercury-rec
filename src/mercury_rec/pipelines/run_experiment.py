"""Run the offline A/B simulation and the promotion gate, and persist both.

This produces the numbers the frontend's experimentation page renders. It is
the last stage of the model lifecycle before serving:

    train -> evaluate -> compare (A/B) -> gate -> promote or reject

The comparison and the gate answer different questions and both are needed.
The A/B simulation asks "is the treatment better, and by how much, with what
uncertainty". The gate asks "is it safe to serve" - which can be false even
when the first answer is yes, because a model can improve ranking quality
while collapsing coverage or breaching the latency budget.
"""

from __future__ import annotations

import json
import time
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd

from mercury_rec.core.logging import get_logger
from mercury_rec.evaluation.harness import load_split, prepare
from mercury_rec.experimentation.ab_simulation import simulate_ab_test
from mercury_rec.experimentation.tracking import log_metrics, track_run
from mercury_rec.experimentation.validation_gate import GateConfig, evaluate_gate
from mercury_rec.models.base import Recommender
from mercury_rec.models.item_cf import ItemCFRecommender
from mercury_rec.models.mf import BPRRecommender
from mercury_rec.models.popularity import PopularityRecommender

logger = get_logger(__name__)


@dataclass(slots=True)
class ExperimentOutcome:
    output_path: Path
    payload: dict[str, Any]


def run_experiment(
    *,
    preset: str = "full",
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    control: str = "popularity",
    treatment: str = "bpr_mf",
    k: int = 10,
    n_bootstrap: int = 2000,
    track: bool = True,
) -> ExperimentOutcome:
    """Compare two models offline and run the promotion gate on the treatment."""
    started = time.perf_counter()
    root = data_dir or Path("data")
    artifacts = artifacts_dir or Path("artifacts")
    processed_dir = root / "processed" / preset
    output_dir = artifacts / "evaluation" / preset
    output_dir.mkdir(parents=True, exist_ok=True)

    metadata = json.loads((processed_dir / "METADATA.json").read_text(encoding="utf-8"))
    n_users = int(metadata["counts"]["users"])
    n_items = int(metadata["counts"]["items"])
    dataset_hash = metadata.get("dataset_hash")

    train = load_split(processed_dir, "train")
    test = load_split(processed_dir, "test")
    items = pd.read_parquet(processed_dir / "items.parquet")
    data = prepare(train, test, n_users=n_users, n_items=n_items, items=items)

    builders: dict[str, Callable[[], Recommender]] = {
        "popularity": lambda: PopularityRecommender(n_users, n_items),
        "item_cf": lambda: ItemCFRecommender(n_users, n_items),
        "bpr_mf": lambda: BPRRecommender(n_users, n_items, epochs=30),
    }
    for name in (control, treatment):
        if name not in builders:
            raise ValueError(f"Unknown variant {name!r}. Available: {sorted(builders)}.")

    recommendations: dict[str, dict[int, list[int]]] = {}
    for name in (control, treatment):
        model = builders[name]()
        model.fit(train)
        recommendations[name] = {
            user: model.recommend(
                user,
                k=k,
                exclude=data.train_history.get(user),
                context=data.user_contexts.get(user),
            )
            for user in data.ground_truth
        }
        logger.info("experiment.variant_scored", variant=name)

    simulation = simulate_ab_test(
        recommendations[control],
        recommendations[treatment],
        data.ground_truth,
        control_name=control,
        treatment_name=treatment,
        k=k,
        n_bootstrap=n_bootstrap,
    )

    # --- promotion gate ---------------------------------------------------
    # Fed from the committed comparison table so the gate sees exactly the
    # numbers published in the README, not a separately-computed set.
    results_path = output_dir / "results_test.json"
    gate_payload: dict[str, Any] = {}
    if results_path.is_file():
        rows = {
            row["model"]: row
            for row in json.loads(results_path.read_text(encoding="utf-8"))["results"]
        }
        candidate = rows.get(treatment)
        incumbent = rows.get(control)
        if candidate is not None:
            gate = evaluate_gate(
                candidate,
                incumbent,
                candidate_version=treatment,
                incumbent_version=control,
                config=GateConfig(),
            )
            gate_payload = gate.as_dict()
            gate_payload["summary"] = gate.summary()
    else:
        logger.warning("experiment.no_evaluation_results", expected=str(results_path))

    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "preset": preset,
        "dataset_hash": dataset_hash,
        "k": k,
        "simulation": simulation.as_dict(),
        "gate": gate_payload,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }

    output_path = output_dir / "experiment_results.json"
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")

    with track_run(
        run_name=f"ab_{control}_vs_{treatment}",
        params={
            "control": control,
            "treatment": treatment,
            "k": k,
            "n_bootstrap": n_bootstrap,
            "dataset_hash": dataset_hash,
        },
        tags={"kind": "ab_simulation", "is_simulated": "true"},
        enabled=track,
    ) as run:
        if run is not None:
            log_metrics({f"{c.metric}_lift": c.absolute_lift for c in simulation.comparisons})

    logger.info("experiment.complete", path=str(output_path))
    return ExperimentOutcome(output_path=output_path, payload=payload)


__all__ = ["ExperimentOutcome", "run_experiment"]
