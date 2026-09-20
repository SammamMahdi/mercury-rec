"""The recommender interface every retrieval model implements.

A single interface across popularity, item-CF, matrix factorisation and the
two-tower model is what makes the comparison table meaningful: the evaluation
harness runs identical code against each, so a difference in the numbers is a
difference in the model rather than a difference in how it was measured.

Two conventions are load-bearing:

- ``fit`` receives only the **training window**. Passing evaluation data would
  leak, so the harness never offers it.
- ``recommend`` receives the items to exclude. Re-recommending something the
  user already interacted with in training inflates every metric, because
  those items are exactly the ones the model has the most evidence for.
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from mercury_rec.core.logging import get_logger

logger = get_logger(__name__)


@dataclass(slots=True)
class RecommendationContext:
    """Request-time context available to a model.

    Contextual personalisation is a stated requirement, so context is part of
    the interface rather than a special case bolted onto one model. Models
    that ignore it (plain popularity, matrix factorisation) simply do not read
    these fields, which keeps the comparison honest: every model is offered
    the same information.
    """

    hour: int | None = None
    weekday: int | None = None
    region_id: int | None = None
    vertical: int | None = None
    session_items: tuple[int, ...] = ()
    now_ns: int | None = None


@dataclass(slots=True)
class FitResult:
    """What fitting produced, for experiment tracking."""

    model: str
    train_seconds: float
    n_users: int
    n_items: int
    n_interactions: int
    params: dict[str, Any] = field(default_factory=dict)
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "model": self.model,
            "train_seconds": round(self.train_seconds, 3),
            "n_users": self.n_users,
            "n_items": self.n_items,
            "n_interactions": self.n_interactions,
            **self.params,
            **self.extra,
        }


class Recommender(ABC):
    """Base class for every candidate-retrieval model."""

    #: Stable identifier used in metric tables, MLflow runs and the API.
    name: str = "recommender"

    def __init__(self, n_users: int, n_items: int) -> None:
        self.n_users = n_users
        self.n_items = n_items
        self._fitted = False
        self._fit_result: FitResult | None = None

    @property
    def is_fitted(self) -> bool:
        return self._fitted

    @property
    def fit_result(self) -> FitResult:
        if self._fit_result is None:
            raise RuntimeError(f"{self.name} has not been fitted yet.")
        return self._fit_result

    def _require_fitted(self) -> None:
        if not self._fitted:
            raise RuntimeError(f"{self.name} must be fitted before recommending. Call fit() first.")

    @abstractmethod
    def _fit(self, interactions: pd.DataFrame) -> dict[str, Any]:
        """Train on the training window. Returns tracking metadata."""

    @abstractmethod
    def _score(
        self, user_id: int, candidates: np.ndarray, context: RecommendationContext
    ) -> np.ndarray:
        """Score candidate items for a user. Higher is better."""

    def fit(self, interactions: pd.DataFrame) -> FitResult:
        """Train, timing the run and recording metadata for MLflow.

        Timing lives here rather than in each subclass so the reported
        training time is measured identically for every model.
        """
        started = time.perf_counter()
        extra = self._fit(interactions)
        elapsed = time.perf_counter() - started

        self._fitted = True
        self._fit_result = FitResult(
            model=self.name,
            train_seconds=elapsed,
            n_users=self.n_users,
            n_items=self.n_items,
            n_interactions=len(interactions),
            params=self.params(),
            extra=extra,
        )
        logger.info("model.fit.complete", model=self.name, seconds=round(elapsed, 2), **extra)
        return self._fit_result

    def params(self) -> dict[str, Any]:
        """Hyperparameters, logged with every experiment run."""
        return {}

    def recommend(
        self,
        user_id: int,
        k: int = 10,
        *,
        exclude: set[int] | None = None,
        candidates: np.ndarray | None = None,
        context: RecommendationContext | None = None,
    ) -> list[int]:
        """Return the top-``k`` item ids for a user, best first.

        Args:
            user_id: Dense user id.
            k: How many items to return.
            exclude: Items to suppress - normally everything the user already
                interacted with during training. Omitting this inflates every
                metric, because those items are the ones the model has the
                most evidence for.
            candidates: Restrict scoring to this subset. Used by the ranking
                stage, which scores only what retrieval proposed.
            context: Request-time context.
        """
        self._require_fitted()
        ctx = context or RecommendationContext()

        pool = np.arange(self.n_items) if candidates is None else np.asarray(candidates)
        if exclude:
            mask = ~np.isin(pool, np.fromiter(exclude, dtype=np.int64, count=len(exclude)))
            pool = pool[mask]
        if pool.size == 0:
            return []

        scores = self._score(user_id, pool, ctx)
        if scores.shape != pool.shape:
            raise RuntimeError(
                f"{self.name}._score returned shape {scores.shape}, expected {pool.shape}."
            )

        top_k = min(k, pool.size)
        # argpartition is O(n) to find the top-k, then only those are sorted.
        # Sorting the full candidate array would be O(n log n) for no gain
        # when the catalogue is 36k items and k is 10.
        partitioned = np.argpartition(-scores, top_k - 1)[:top_k]
        ordered = partitioned[np.argsort(-scores[partitioned], kind="stable")]
        return [int(pool[i]) for i in ordered]

    def recommend_batch(
        self,
        user_ids: list[int],
        k: int = 10,
        *,
        exclude: dict[int, set[int]] | None = None,
        context: RecommendationContext | None = None,
    ) -> dict[int, list[int]]:
        """Recommend for many users. Subclasses may override to vectorise."""
        excludes = exclude or {}
        return {
            user_id: self.recommend(user_id, k, exclude=excludes.get(user_id), context=context)
            for user_id in user_ids
        }

    def save(self, path: Path) -> None:
        raise NotImplementedError(f"{self.name} does not implement save().")

    @classmethod
    def load(cls, path: Path) -> Recommender:
        raise NotImplementedError(f"{cls.__name__} does not implement load().")


__all__ = ["FitResult", "RecommendationContext", "Recommender"]
