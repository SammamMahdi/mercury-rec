"""Load trained artifacts into a servable bundle.

Everything the engine needs is loaded once at startup and swapped as a single
object. Loading models individually would allow a ranker from one training run
to be served against embeddings from another, which does not raise -- it
produces plausible scores that are quietly wrong.

Loading is deliberately tolerant. A missing ranker means the service runs as
retrieval-only; a missing two-tower means the remaining sources serve the
request. What is *not* tolerated is a feature-schema mismatch, because that
one is silent: shapes still align, scores are still produced, and they are
meaningless.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, cast

import numpy as np
import pandas as pd

from mercury_rec.core.logging import get_logger
from mercury_rec.features.store import FEATURE_SCHEMA_VERSION, AsOfFeatureStore, load_state
from mercury_rec.models.item_cf import ItemCFRecommender
from mercury_rec.models.mf import BPRRecommender
from mercury_rec.models.popularity import ContextualPopularityRecommender, PopularityRecommender
from mercury_rec.models.ranking.ranker import LambdaRanker
from mercury_rec.models.two_tower.recommender import TwoTowerRecommender
from mercury_rec.recommender.engine import ArtifactBundle

logger = get_logger(__name__)


class ArtifactsMissingError(RuntimeError):
    """Raised when the artifacts required to serve are absent."""


def _load_metadata(processed_dir: Path) -> dict[str, Any]:
    path = processed_dir / "METADATA.json"
    if not path.is_file():
        raise ArtifactsMissingError(
            f"No dataset metadata at {path}. Run `mercury data build` first."
        )
    return json.loads(path.read_text(encoding="utf-8"))  # type: ignore[no-any-return]


def load_bundle(
    *,
    preset: str = "full",
    data_dir: Path | None = None,
    artifacts_dir: Path | None = None,
    fit_retrieval: bool = True,
) -> ArtifactBundle:
    """Assemble a servable :class:`ArtifactBundle`.

    Args:
        preset: Dataset preset to serve.
        data_dir: Root of ``data/``.
        artifacts_dir: Root of ``artifacts/``.
        fit_retrieval: Fit the lightweight retrieval models (popularity,
            item-CF, MF) from the training window at startup. They train in
            seconds and have no separate persisted form, so refitting is
            simpler and less error-prone than maintaining another artifact
            that could fall out of sync with the data.

    Raises:
        ArtifactsMissingError: When the dataset or feature state is absent.
    """
    root = data_dir or Path("data")
    artifacts = artifacts_dir or Path("artifacts")
    processed_dir = root / "processed" / preset
    feature_dir = artifacts / "features" / preset

    metadata = _load_metadata(processed_dir)
    n_users = int(metadata["counts"]["users"])
    n_items = int(metadata["counts"]["items"])
    dataset_hash = metadata.get("dataset_hash")

    items = pd.read_parquet(processed_dir / "items.parquet").sort_values("item_id")
    user_map = pd.read_parquet(processed_dir / "user_id_map.parquet")
    item_map = pd.read_parquet(processed_dir / "item_id_map.parquet")
    train = pd.read_parquet(processed_dir / "interactions_train.parquet")

    # --- feature state ---------------------------------------------------
    state_path = feature_dir / "state_train.pkl"
    if not state_path.is_file():
        raise ArtifactsMissingError(
            f"No feature state at {state_path}. Run `mercury features build` first."
        )
    state = load_state(state_path)  # raises on a schema mismatch

    feature_store = AsOfFeatureStore(
        state,
        item_prices=items["price"].to_numpy(dtype=np.float32),
        item_verticals=items["vertical"].to_numpy(dtype=np.int16),
        item_merchants=items["merchant_id"].to_numpy(dtype=np.int32),
    )

    # --- retrieval -------------------------------------------------------
    popularity = PopularityRecommender(n_users, n_items)
    contextual = ContextualPopularityRecommender(n_users, n_items)
    item_cf: ItemCFRecommender | None = None
    matrix_factorization: BPRRecommender | None = None

    if fit_retrieval:
        popularity.fit(train)
        contextual.fit(train)
        item_cf = ItemCFRecommender(n_users, n_items)
        item_cf.fit(train)
        matrix_factorization = BPRRecommender(n_users, n_items, epochs=30)
        matrix_factorization.fit(train)

    # --- two-tower -------------------------------------------------------
    # Loaded from persisted embeddings, not rebuilt: the towers are an offline
    # cost, and a replica that had to reconstruct a torch graph and a CUDA
    # context to answer a dot product would be paying for the wrong half of
    # the architecture.
    two_tower: TwoTowerRecommender | None = None
    two_tower_path = artifacts / "models" / preset / "two_tower.npz"
    if two_tower_path.is_file():
        try:
            two_tower = TwoTowerRecommender.load_serving_state(
                two_tower_path,
                items=items,
                users=pd.read_parquet(processed_dir / "users.parquet"),
            )
        except (ValueError, OSError, KeyError) as exc:
            # Retrieval degrades to the remaining sources. A mismatched or
            # unreadable embedding file must not take the service down, but it
            # must also never be served, because wrong-by-one embeddings
            # produce confident nonsense rather than an error.
            logger.warning("artifacts.two_tower_load_failed", error=str(exc)[:200])
    else:
        logger.info("artifacts.two_tower_absent", expected=str(two_tower_path))

    # --- ranker ----------------------------------------------------------
    ranker: LambdaRanker | None = None
    ranker_path = artifacts / "models" / preset / "ranker.txt"
    if ranker_path.is_file():
        try:
            ranker = LambdaRanker.load(ranker_path)
            # Pay SHAP's ~900ms TreeExplainer construction here rather than on
            # the first request that asks for an explanation.
            ranker.warmup()
            logger.info("artifacts.ranker_loaded", path=str(ranker_path))
        except Exception as exc:  # noqa: BLE001
            # Retrieval-only is a valid deployment; refusing to start because
            # the ranker is unreadable would be worse than serving without it.
            logger.warning("artifacts.ranker_load_failed", error=str(exc)[:160])
    else:
        logger.warning("artifacts.ranker_absent", expected=str(ranker_path))

    # --- lookups ---------------------------------------------------------
    user_index = {
        str(source): int(internal)
        for source, internal in zip(
            user_map["source_user_id"].to_numpy(), user_map["user_id"].to_numpy(), strict=True
        )
    }
    item_index = {
        int(source): int(internal)
        for source, internal in zip(
            item_map["source_item_id"].to_numpy(), item_map["item_id"].to_numpy(), strict=True
        )
    }
    user_history = {
        int(cast("int", user)): set(history)
        for user, history in train.groupby("user_id")["item_id"].apply(set).items()
    }

    popularity_scores = (
        popularity.item_scores
        if fit_retrieval
        else np.bincount(train["item_id"].to_numpy(dtype=np.int64), minlength=n_items).astype(
            np.float64
        )
    )
    popularity_rank = np.argsort(np.argsort(-popularity_scores)) / max(n_items - 1, 1)

    bundle = ArtifactBundle(
        model_version=f"{preset}@{dataset_hash}",
        dataset_hash=dataset_hash,
        n_users=n_users,
        n_items=n_items,
        feature_store=feature_store,
        popularity=popularity,
        contextual_popularity=contextual if fit_retrieval else None,
        item_cf=item_cf,
        matrix_factorization=matrix_factorization,
        two_tower=two_tower,
        ranker=ranker,
        items=items.reset_index(drop=True),
        user_index=user_index,
        item_index=item_index,
        user_history=user_history,
        popularity_rank=popularity_rank,
    )
    logger.info(
        "artifacts.bundle_loaded",
        model_version=bundle.model_version,
        n_users=n_users,
        n_items=n_items,
        feature_schema=FEATURE_SCHEMA_VERSION,
        has_ranker=ranker is not None,
        has_two_tower=two_tower is not None,
    )
    return bundle


__all__ = ["ArtifactsMissingError", "load_bundle"]
