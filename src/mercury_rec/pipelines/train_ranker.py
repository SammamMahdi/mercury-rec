"""Train the ranker on real retrieval output, and measure the two-stage system.

This pipeline is where the multi-stage architecture is actually assembled:

1. Fit the retrieval models on the training window.
2. Generate candidates for every evaluation user, exactly as serving would.
3. Label those candidates by what the user went on to interact with.
4. Train LambdaRank on the labelled candidate lists.
5. Score the full pipeline -- retrieval, ranking, business re-ranking -- and
   compare against retrieval alone.

Two numbers matter and are reported separately:

- **Retrieval recall** is the ceiling. An item never proposed cannot be
  recommended, so no ranker can recover it, and end-to-end quality can never
  exceed this.
- **Ranking lift** is how much of that ceiling the ranker converts into
  top-K quality.

Reporting only the end-to-end number hides which stage is the bottleneck, and
therefore hides where the next hour of work should go.

Candidates for the ranker's TRAINING set are generated inside the training
window, using a held-out slice of it, so the ranker never sees evaluation
data. Generating them from the evaluation window would leak the labels it is
being asked to predict.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from mercury_rec.core.enums import RetrievalSource
from mercury_rec.core.logging import get_logger
from mercury_rec.evaluation.harness import EvaluationData, load_split, prepare
from mercury_rec.evaluation.metrics import DEFAULT_K_VALUES, evaluate_recommendations
from mercury_rec.features.asof import FEATURE_COLUMNS
from mercury_rec.models.base import RecommendationContext
from mercury_rec.models.item_cf import ItemCFRecommender
from mercury_rec.models.popularity import PopularityRecommender
from mercury_rec.models.ranking.dataset import RANKING_FEATURES, build_ranking_dataset
from mercury_rec.models.ranking.ranker import LambdaRanker, RankerConfig
from mercury_rec.models.two_tower.model import TwoTowerConfig
from mercury_rec.models.two_tower.recommender import TwoTowerRecommender
from mercury_rec.reranking.pipeline import RerankConfig, ScoredItem, rerank
from mercury_rec.retrieval.candidates import (
    reciprocal_rank_fusion,
    retrieval_recall,
    retrieve_from_scores,
)

logger = get_logger(__name__)

#: Retrieval here is context-free; contextual popularity is exercised
#: separately in the model comparison.
_EMPTY_CONTEXT = RecommendationContext()


@dataclass(slots=True)
class RankerPipelineResult:
    output_path: Path
    payload: dict[str, Any]


@dataclass(slots=True)
class _Retrievers:
    """The fitted retrieval stack, shared by candidate generation everywhere."""

    popularity: PopularityRecommender
    item_cf: ItemCFRecommender
    two_tower: TwoTowerRecommender
    user_embeddings: dict[int, np.ndarray]


def _lookup_int(table: pd.DataFrame, key: Any, column: str) -> int | None:
    """Read one int cell, or None when the row is absent.

    pandas types a .at lookup as a broad Scalar union, so the int-ness is
    asserted here once rather than at every call site.
    """
    if key not in table.index:
        return None
    return int(cast("int", table.at[key, column]))


def _generate_candidates(
    retrievers: _Retrievers,
    user_id: int,
    *,
    exclude: set[int],
    per_source_k: int,
    max_candidates: int,
) -> Any:
    """Run every retrieval source for one user and fuse the results."""
    results = []

    started = time.perf_counter()
    popularity_scores = retrievers.popularity.item_scores
    results.append(
        retrieve_from_scores(
            popularity_scores,
            source=RetrievalSource.POPULARITY,
            k=per_source_k,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
    )

    started = time.perf_counter()
    cf_scores = retrievers.item_cf._score(
        user_id, np.arange(retrievers.item_cf.n_items), _EMPTY_CONTEXT
    )
    results.append(
        retrieve_from_scores(
            cf_scores,
            source=RetrievalSource.ITEM_CF,
            k=per_source_k,
            elapsed_ms=(time.perf_counter() - started) * 1000.0,
        )
    )

    embedding = retrievers.user_embeddings.get(user_id)
    if embedding is not None:
        started = time.perf_counter()
        tower_scores = retrievers.two_tower.item_embeddings @ embedding
        results.append(
            retrieve_from_scores(
                tower_scores,
                source=RetrievalSource.TWO_TOWER,
                k=per_source_k,
                elapsed_ms=(time.perf_counter() - started) * 1000.0,
            )
        )

    return reciprocal_rank_fusion(results, max_candidates=max_candidates, exclude=exclude)


def _candidate_frame(
    retrievers: _Retrievers,
    users: list[int],
    *,
    data: EvaluationData,
    user_features: pd.DataFrame,
    truth: dict[int, set[int]],
    per_source_k: int,
    max_candidates: int,
) -> tuple[pd.DataFrame, list[float]]:
    """Build the labelled candidate frame, plus per-user retrieval recall."""
    feature_lookup = user_features.set_index("user_id")
    rows: list[pd.DataFrame] = []
    recalls: list[float] = []

    for user_id in users:
        relevant = truth.get(user_id, set())
        candidates = _generate_candidates(
            retrievers,
            user_id,
            exclude=data.train_history.get(user_id, set()),
            per_source_k=per_source_k,
            max_candidates=max_candidates,
        )
        if len(candidates) == 0:
            continue

        recalls.append(retrieval_recall(candidates.item_ids, relevant))

        frame = pd.DataFrame({"user_id": user_id, "item_id": candidates.item_ids})
        frame["label"] = np.isin(
            candidates.item_ids, np.fromiter(relevant, dtype=np.int64, count=len(relevant))
        ).astype(np.int32)

        # The user's as-of feature vector is constant across their candidates;
        # the item-side columns vary, but those are already folded into the
        # retrieval scores at this stage. Broadcasting is what makes building
        # hundreds of thousands of candidate rows tractable.
        if user_id in feature_lookup.index:
            user_row = feature_lookup.loc[user_id]
            for name in FEATURE_COLUMNS:
                frame[name] = float(user_row[name]) if name in user_row else np.nan
        else:
            for name in FEATURE_COLUMNS:
                frame[name] = np.nan

        for source in ("two_tower", "item_cf", "popularity"):
            frame[f"retrieval_score_{source}"] = candidates.per_source_scores.get(
                source, np.full(len(candidates), np.nan, dtype=np.float32)
            )
        frame["retrieval_rank_fused"] = np.arange(1, len(candidates) + 1, dtype=np.float32)
        frame["retrieval_n_sources"] = candidates.sources_per_item.astype(np.float32)
        rows.append(frame)

    if not rows:
        raise ValueError("Candidate generation produced no rows.")
    return pd.concat(rows, ignore_index=True), recalls


def run_ranker_pipeline(
    *,
    preset: str = "full",
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    per_source_k: int = 200,
    max_candidates: int = 400,
    max_train_users: int = 4000,
    quick: bool = False,
) -> RankerPipelineResult:
    """Fit retrieval, train the ranker on its output, and score end to end."""
    started = time.perf_counter()
    root = data_dir or Path("data")
    processed_dir = root / "processed" / preset
    artifacts = artifacts_dir or Path("artifacts")
    feature_dir = artifacts / "features" / preset
    output_dir = artifacts / "evaluation" / preset
    output_dir.mkdir(parents=True, exist_ok=True)

    dataset_meta = json.loads((processed_dir / "METADATA.json").read_text(encoding="utf-8"))
    n_users = int(dataset_meta["counts"]["users"])
    n_items = int(dataset_meta["counts"]["items"])

    train = load_split(processed_dir, "train")
    validation = load_split(processed_dir, "validation")
    test = load_split(processed_dir, "test")
    items = pd.read_parquet(processed_dir / "items.parquet")
    users_frame = pd.read_parquet(processed_dir / "users.parquet")

    train_features = pd.read_parquet(feature_dir / "features_train.parquet")
    validation_features = pd.read_parquet(feature_dir / "features_validation.parquet")
    test_features = pd.read_parquet(feature_dir / "features_test.parquet")

    # --- 1. fit retrieval ------------------------------------------------
    logger.info("ranker_pipeline.fit_retrieval")
    popularity = PopularityRecommender(n_users, n_items)
    popularity.fit(train)
    item_cf = ItemCFRecommender(n_users, n_items)
    item_cf.fit(train)
    two_tower = TwoTowerRecommender(
        n_users,
        n_items,
        items=items,
        users=users_frame,
        # 25 epochs, selected on the VALIDATION split. Training longer keeps
        # lowering the loss (1.60 -> 1.10 at 60 epochs) while validation
        # Recall@10 FALLS (0.0404 -> 0.0357): the model is memorising the
        # training interactions. Loss alone would have chosen the worse model.
        config=TwoTowerConfig(epochs=5 if quick else 25),
    )
    two_tower.fit(train_features)

    def _entering(features: pd.DataFrame) -> pd.DataFrame:
        return features.sort_values("ts").drop_duplicates("user_id", keep="last")

    # --- 2. ranker training set, from WITHIN the training window ---------
    # Candidates are generated for users in the validation window and labelled
    # from it. Using the test window would leak the very labels the ranker is
    # asked to predict.
    validation_data = prepare(train, validation, n_users=n_users, n_items=n_items, items=items)
    validation_entering = _entering(validation_features)
    two_tower.set_user_embeddings(
        validation_entering["user_id"].to_numpy(),
        two_tower.encode_users(validation_entering),
    )
    retrievers = _Retrievers(
        popularity=popularity,
        item_cf=item_cf,
        two_tower=two_tower,
        user_embeddings={
            int(uid): vec
            for uid, vec in zip(
                validation_entering["user_id"].to_numpy(),
                two_tower.encode_users(validation_entering),
                strict=True,
            )
        },
    )

    train_users = sorted(validation_data.ground_truth)[:max_train_users]
    logger.info("ranker_pipeline.generate_train_candidates", users=len(train_users))
    train_frame, train_recalls = _candidate_frame(
        retrievers,
        train_users,
        data=validation_data,
        user_features=validation_entering,
        truth=validation_data.ground_truth,
        per_source_k=per_source_k,
        max_candidates=max_candidates,
    )
    ranking_train = build_ranking_dataset(train_frame)

    # --- 3. train the ranker ---------------------------------------------
    ranker = LambdaRanker(RankerConfig(n_estimators=100 if quick else 500))
    ranker_fit = ranker.fit(ranking_train)
    ranker.save(artifacts / "models" / preset / "ranker.txt")

    # --- 4. evaluate end to end on TEST ----------------------------------
    test_data = prepare(train, test, n_users=n_users, n_items=n_items, items=items)
    test_entering = _entering(test_features)
    test_embeddings = two_tower.encode_users(test_entering)
    two_tower.set_user_embeddings(test_entering["user_id"].to_numpy(), test_embeddings)
    retrievers.user_embeddings = {
        int(uid): vec
        for uid, vec in zip(test_entering["user_id"].to_numpy(), test_embeddings, strict=True)
    }

    test_users = sorted(test_data.ground_truth)
    logger.info("ranker_pipeline.generate_test_candidates", users=len(test_users))
    test_frame, test_recalls = _candidate_frame(
        retrievers,
        test_users,
        data=test_data,
        user_features=test_entering,
        truth=test_data.ground_truth,
        per_source_k=per_source_k,
        max_candidates=max_candidates,
    )

    scores = ranker.score(test_frame[list(RANKING_FEATURES)].to_numpy(dtype=np.float32))
    test_frame = test_frame.assign(ml_score=scores)

    item_lookup = items.set_index("item_id")
    popularity_rank = np.argsort(np.argsort(-popularity.item_scores)) / max(n_items - 1, 1)

    retrieval_only: dict[int, list[int]] = {}
    ranked_only: dict[int, list[int]] = {}
    reranked: dict[int, list[int]] = {}

    largest_k = max(DEFAULT_K_VALUES)
    for raw_user_id, group in test_frame.groupby("user_id", sort=False):
        # A groupby key is typed as Hashable; it is an int by construction here.
        user_id = int(cast("int", raw_user_id))
        retrieval_only[user_id] = group["item_id"].tolist()[:largest_k]

        ordered = group.sort_values("ml_score", ascending=False)
        ranked_only[user_id] = ordered["item_id"].tolist()[:largest_k]

        # Column-wise rather than itertuples: numpy arrays are faster over a
        # few hundred rows and carry real types, where namedtuple attributes
        # are opaque to the checker.
        head = ordered.head(100)
        head_items = head["item_id"].to_numpy(dtype=np.int64)
        head_scores = head["ml_score"].to_numpy(dtype=np.float64)
        scored_items = [
            ScoredItem(
                item_id=int(head_items[position]),
                ml_relevance_score=float(head_scores[position]),
                merchant_id=_lookup_int(item_lookup, int(head_items[position]), "merchant_id"),
                category_id=_lookup_int(item_lookup, int(head_items[position]), "category_id"),
            )
            for position in range(len(head_items))
        ]
        result = rerank(
            scored_items,
            k=largest_k,
            config=RerankConfig(),
            popularity_rank=popularity_rank,
        )
        reranked[user_id] = [item.item_id for item in result.items]

    stages = {}
    for label, recommendations in (
        ("retrieval_only", retrieval_only),
        ("two_tower_plus_ranker", ranked_only),
        ("two_tower_ranker_reranked", reranked),
    ):
        evaluation = evaluate_recommendations(
            recommendations,
            test_data.ground_truth,
            model=label,
            k_values=DEFAULT_K_VALUES,
            n_catalog_items=n_items,
            item_popularity=test_data.item_popularity,
            item_categories=test_data.item_categories,
        )
        stages[label] = evaluation.as_dict()

    payload: dict[str, Any] = {
        "generated_at": datetime.now(UTC).isoformat(),
        "preset": preset,
        "dataset_hash": dataset_meta.get("dataset_hash"),
        "quick_mode": quick,
        "retrieval": {
            "per_source_k": per_source_k,
            "max_candidates": max_candidates,
            "mean_recall_at_candidates_test": round(float(np.mean(test_recalls)), 5),
            "mean_recall_at_candidates_train": round(float(np.mean(train_recalls)), 5),
            "note": (
                "Retrieval recall is the CEILING on end-to-end quality: an item "
                "never proposed cannot be ranked or recommended."
            ),
        },
        "ranker": {
            **ranker_fit.as_dict(),
            "train_groups": ranking_train.n_groups,
            "train_rows": len(ranking_train.labels),
            "positive_rate": ranking_train.positive_rate,
        },
        "stages": stages,
        "elapsed_seconds": round(time.perf_counter() - started, 2),
    }

    output_path = output_dir / "ranking_results.json"
    output_path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")
    logger.info("ranker_pipeline.complete", path=str(output_path))
    return RankerPipelineResult(output_path=output_path, payload=payload)


__all__ = ["RankerPipelineResult", "run_ranker_pipeline"]
