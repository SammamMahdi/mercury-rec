"""Run models against a held-out window under identical conditions.

The point of a harness is that every model is measured the same way. If
popularity were evaluated over all users and the two-tower model over only
users it happened to cover, the comparison table would be measuring
evaluation protocol rather than model quality.

Two protocol decisions matter and are applied to every model equally:

1. **Training history is excluded from recommendations.** Re-recommending
   something a user already interacted with during training inflates every
   metric, because those are precisely the items the model has the most
   evidence for. A model that simply replays history would otherwise look
   excellent.
2. **Ground truth is the items a user interacted with in the evaluation window
   that they had NOT already interacted with during training.** This follows
   necessarily from decision 1: if training history is excluded from the
   candidate pool, then leaving those items in the denominator makes recall
   mathematically unreachable. Measured on this dataset, 16.4% of test pairs
   are repeats of training pairs, and 14.9% of evaluation users have *only*
   repeats - scored against the raw truth set they are guaranteed zeros, and
   every reported metric is depressed by a population no model could ever
   serve. Those users are skipped and the count is reported.

3. **All intent levels count as relevant**, not just purchases. At this
   dataset's 1.3% conversion rate a purchase-only truth set would leave most
   users empty, and the metric would describe the few hundred users who
   happened to buy rather than the system.
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass
from pathlib import Path
from typing import cast

import numpy as np
import pandas as pd

from mercury_rec.core.logging import get_logger
from mercury_rec.evaluation.metrics import (
    DEFAULT_K_VALUES,
    EvaluationResult,
    evaluate_recommendations,
)
from mercury_rec.models.base import RecommendationContext, Recommender

logger = get_logger(__name__)


def _grouped_sets(frame: pd.DataFrame, key: str, value: str) -> dict[int, set[int]]:
    """Group ``value`` into a set per ``key``, with the key type preserved.

    pandas types a groupby key as ``Hashable``, so the resulting dict loses
    its ``int`` key type at every call site. Converting once here keeps the
    three callers below readable and correctly typed.
    """
    grouped = frame.groupby(key)[value].apply(set)
    return {int(cast("int", k)): set(v) for k, v in grouped.items()}


@dataclass(slots=True)
class EvaluationData:
    """Everything the harness needs, prepared once and shared by all models."""

    train: pd.DataFrame
    evaluation: pd.DataFrame
    n_users: int
    n_items: int
    ground_truth: dict[int, set[int]]
    train_history: dict[int, set[int]]
    item_popularity: np.ndarray
    item_categories: np.ndarray
    user_contexts: dict[int, RecommendationContext]


def prepare(
    train: pd.DataFrame,
    evaluation: pd.DataFrame,
    *,
    n_users: int,
    n_items: int,
    items: pd.DataFrame | None = None,
    max_users: int | None = None,
    seed: int = 42,
) -> EvaluationData:
    """Build shared evaluation inputs.

    Args:
        train: Training-window interactions.
        evaluation: Held-out interactions (validation or test).
        n_users: Catalogue user count.
        n_items: Catalogue item count.
        items: Item dimension, for category-based diversity.
        max_users: Sample this many evaluation users. Sampling is seeded, and
            the same sample is reused across models - a different sample per
            model would make the comparison meaningless.
        seed: Sampling seed.
    """
    raw_truth = _grouped_sets(evaluation, "user_id", "item_id")

    # Training history is excluded from the candidate pool, so it must also be
    # excluded from the relevance denominator - otherwise recall has a ceiling
    # below 1.0 that no model can reach, and the metric is internally
    # inconsistent. Users left with nothing novel are dropped here and
    # surfaced as `n_users_skipped` rather than silently scored zero.
    all_history = _grouped_sets(train[train["user_id"].isin(set(raw_truth))], "user_id", "item_id")
    ground_truth = {user: items - all_history.get(user, set()) for user, items in raw_truth.items()}
    repeat_only = [user for user, items in ground_truth.items() if not items]
    ground_truth = {user: items for user, items in ground_truth.items() if items}
    logger.info(
        "evaluation.ground_truth",
        users_with_novel_items=len(ground_truth),
        users_repeat_only_dropped=len(repeat_only),
    )

    if max_users is not None and len(ground_truth) > max_users:
        rng = np.random.default_rng(seed)
        sampled = rng.choice(np.array(sorted(ground_truth)), size=max_users, replace=False)
        ground_truth = {int(u): ground_truth[int(u)] for u in sampled}
        logger.info("evaluation.sampled_users", n=len(ground_truth), seed=seed)

    relevant_users = set(ground_truth)
    history_frame = train[train["user_id"].isin(relevant_users)]
    train_history = _grouped_sets(history_frame, "user_id", "item_id")

    # Popularity for the bias metrics comes from TRAINING only. Deriving it
    # from the evaluation window would leak the answer into a reported number.
    item_popularity = np.bincount(
        train["item_id"].to_numpy(dtype=np.int64), minlength=n_items
    ).astype(np.float64)

    item_categories = np.full(n_items, -1, dtype=np.int64)
    if items is not None and "category_id" in items.columns:
        ids = items["item_id"].to_numpy(dtype=np.int64)
        valid = (ids >= 0) & (ids < n_items)
        item_categories[ids[valid]] = items["category_id"].to_numpy(dtype=np.int64)[valid]

    # Each user's context is taken from their FIRST evaluation event, so a
    # contextual model is asked the same question the data actually posed.
    contexts: dict[int, RecommendationContext] = {}
    first_events = (
        evaluation[evaluation["user_id"].isin(relevant_users)]
        .sort_values("ts")
        .drop_duplicates("user_id", keep="first")
    )
    # Column-wise rather than itertuples: numpy arrays are both faster over
    # thousands of rows and actually typed, where a namedtuple attribute is
    # opaque to the checker.
    ctx_users = first_events["user_id"].to_numpy(dtype=np.int64)
    ctx_stamps = first_events["ts"]
    ctx_hours = ctx_stamps.dt.hour.to_numpy(dtype=np.int64)
    ctx_weekdays = ctx_stamps.dt.dayofweek.to_numpy(dtype=np.int64)
    ctx_ns = ctx_stamps.to_numpy(dtype="datetime64[ns]").astype(np.int64)
    ctx_regions = first_events["region_id"].fillna(-1).to_numpy(dtype=np.int64)
    ctx_verticals = first_events["vertical"].fillna(-1).to_numpy(dtype=np.int64)

    for position in range(len(first_events)):
        region = int(ctx_regions[position])
        vertical = int(ctx_verticals[position])
        contexts[int(ctx_users[position])] = RecommendationContext(
            hour=int(ctx_hours[position]),
            weekday=int(ctx_weekdays[position]),
            region_id=region if region >= 0 else None,
            vertical=vertical if vertical >= 0 else None,
            now_ns=int(ctx_ns[position]),
        )

    logger.info(
        "evaluation.prepared",
        users=len(ground_truth),
        train_events=len(train),
        eval_events=len(evaluation),
    )
    return EvaluationData(
        train=train,
        evaluation=evaluation,
        n_users=n_users,
        n_items=n_items,
        ground_truth=ground_truth,
        train_history=train_history,
        item_popularity=item_popularity,
        item_categories=item_categories,
        user_contexts=contexts,
    )


def evaluate_model(
    model: Recommender,
    data: EvaluationData,
    *,
    k_values: Sequence[int] = DEFAULT_K_VALUES,
    use_context: bool = True,
) -> tuple[EvaluationResult, float]:
    """Score one model. Returns the result and mean per-user latency in ms.

    The latency figure is offline batch scoring on this machine, NOT a serving
    measurement - it excludes feature lookup, caching, network and the API
    layer entirely. It is useful for comparing models against one another and
    is labelled as such wherever it is reported. Real serving latency is
    measured separately under load.
    """
    largest_k = max(k_values)
    recommendations: dict[int, list[int]] = {}

    started = time.perf_counter()
    for user_id in data.ground_truth:
        context = data.user_contexts.get(user_id) if use_context else None
        recommendations[user_id] = model.recommend(
            user_id,
            k=largest_k,
            exclude=data.train_history.get(user_id),
            context=context,
        )
    elapsed = time.perf_counter() - started
    per_user_ms = (elapsed / max(len(data.ground_truth), 1)) * 1000

    result = evaluate_recommendations(
        recommendations,
        data.ground_truth,
        model=model.name,
        k_values=k_values,
        n_catalog_items=data.n_items,
        item_popularity=data.item_popularity,
        item_categories=data.item_categories,
    )
    logger.info(
        "evaluation.model_scored",
        model=model.name,
        seconds=round(elapsed, 2),
        per_user_ms=round(per_user_ms, 3),
    )
    return result, per_user_ms


def load_split(processed_dir: Path, split: str) -> pd.DataFrame:
    path = processed_dir / f"interactions_{split}.parquet"
    if not path.is_file():
        raise FileNotFoundError(f"Missing {path}. Run `mercury data build` first.")
    return pd.read_parquet(path).sort_values("ts", kind="stable")


__all__ = ["EvaluationData", "evaluate_model", "load_split", "prepare"]
