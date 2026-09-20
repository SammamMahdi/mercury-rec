"""LightGBM LambdaRank ranker with real feature attributions.

Why LambdaRank rather than binary classification
------------------------------------------------
A classifier optimises calibrated per-item probability. A ranker optimises the
*order*, which is the only thing the user experiences: nothing in a top-10 list
depends on whether the model thought an item scored 0.02 or 0.20, only on
whether it placed it above the alternatives.

LambdaRank makes that explicit by weighting each pairwise swap by the NDCG it
would change, so effort concentrates where mistakes are expensive -- at the top
of the list. On this data, where roughly 1 candidate in 300 is relevant, a
classifier also spends most of its capacity modelling the overwhelming negative
class rather than the distinctions that decide the ranking.

Why LightGBM rather than a neural ranker
----------------------------------------
The ranking stage is tabular, has a few hundred rows per request and must
answer in single-digit milliseconds on CPU inside the API process. Gradient-
boosted trees handle heterogeneous tabular features and native NaN without
scaling or imputation, train in seconds, and are directly explainable through
exact TreeSHAP. A neural ranker would be slower to serve, slower to train and
harder to explain, for no demonstrated gain at this scale. That is a
trade-off, and it is recorded in ``docs/design-decisions.md`` rather than
presented as the only option.

Explanations
------------
``explain`` returns exact TreeSHAP values, so the UI's "why was this
recommended?" panel reports the model's actual per-feature contributions to
*this* score. It is not a plausible-sounding narrative generated after the
fact, which the specification explicitly rules out.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np

from mercury_rec.core.logging import get_logger
from mercury_rec.models.ranking.dataset import RANKING_FEATURES, RankingDataset

logger = get_logger(__name__)


@dataclass(slots=True)
class RankerConfig:
    """LambdaRank hyperparameters.

    Defaults are deliberately conservative. The ranking set here is a few
    hundred thousand rows with a ~0.3% positive rate, and on that shape an
    unconstrained GBDT memorises the training groups within a few dozen
    iterations.
    """

    objective: str = "lambdarank"
    metric: str = "ndcg"
    ndcg_eval_at: tuple[int, ...] = (5, 10, 20)
    n_estimators: int = 500
    learning_rate: float = 0.05
    num_leaves: int = 31
    min_child_samples: int = 50
    """Raised from LightGBM's default of 20: with so few positives, small
    leaves fit individual users rather than transferable patterns."""
    subsample: float = 0.8
    subsample_freq: int = 1
    colsample_bytree: float = 0.8
    reg_lambda: float = 1.0
    max_bin: int = 127
    """Halved from the default 255. Memory during Dataset construction scales
    with bin count, and on a 16 GB machine shared with Postgres, Redis and a
    dev server that headroom matters more than the marginal split precision."""
    early_stopping_rounds: int = 50
    seed: int = 42
    n_jobs: int = 8

    def to_params(self) -> dict[str, Any]:
        return {
            "objective": self.objective,
            "metric": self.metric,
            "ndcg_eval_at": list(self.ndcg_eval_at),
            "learning_rate": self.learning_rate,
            "num_leaves": self.num_leaves,
            "min_child_samples": self.min_child_samples,
            "subsample": self.subsample,
            "subsample_freq": self.subsample_freq,
            "colsample_bytree": self.colsample_bytree,
            "reg_lambda": self.reg_lambda,
            "max_bin": self.max_bin,
            "seed": self.seed,
            "num_threads": self.n_jobs,
            "verbosity": -1,
        }


@dataclass(slots=True)
class RankerFitResult:
    best_iteration: int
    best_score: dict[str, float]
    train_seconds: float
    feature_importance: dict[str, float] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        return {
            "best_iteration": self.best_iteration,
            "best_score": self.best_score,
            "train_seconds": round(self.train_seconds, 3),
            "top_features": dict(
                sorted(self.feature_importance.items(), key=lambda kv: -kv[1])[:10]
            ),
        }


class LambdaRanker:
    """Learning-to-rank over retrieval candidates."""

    name = "lambdarank"

    def __init__(self, config: RankerConfig | None = None) -> None:
        self.config = config or RankerConfig()
        self._booster: Any = None
        self._explainer: Any = None
        self.feature_names: tuple[str, ...] = RANKING_FEATURES

    @property
    def is_fitted(self) -> bool:
        return self._booster is not None

    def _require_fitted(self) -> None:
        if self._booster is None:
            raise RuntimeError("LambdaRanker must be fitted before scoring.")

    def fit(
        self, train: RankingDataset, validation: RankingDataset | None = None
    ) -> RankerFitResult:
        """Train, using ``validation`` for early stopping when supplied.

        Early stopping is on held-out NDCG rather than a fixed iteration
        count: the right number of trees depends on the data, and fixing it
        either underfits or silently memorises.
        """
        import time

        import lightgbm as lgb

        self.feature_names = train.feature_names

        train_set = lgb.Dataset(
            train.features,
            label=train.labels,
            group=train.groups,
            feature_name=list(train.feature_names),
            free_raw_data=True,
        )
        valid_sets = [train_set]
        valid_names = ["train"]
        callbacks: list[Any] = [lgb.log_evaluation(period=0)]

        if validation is not None:
            valid_set = lgb.Dataset(
                validation.features,
                label=validation.labels,
                group=validation.groups,
                feature_name=list(validation.feature_names),
                reference=train_set,
                free_raw_data=True,
            )
            valid_sets.append(valid_set)
            valid_names.append("validation")
            callbacks.append(lgb.early_stopping(self.config.early_stopping_rounds, verbose=False))

        started = time.perf_counter()
        booster = lgb.train(
            self.config.to_params(),
            train_set,
            num_boost_round=self.config.n_estimators,
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks,
        )
        elapsed = time.perf_counter() - started
        self._booster = booster
        self._explainer = None  # invalidated by refitting

        gains = booster.feature_importance(importance_type="gain")
        importance = {
            name: float(value) for name, value in zip(self.feature_names, gains, strict=True)
        }

        best_score: dict[str, float] = {}
        for split, metrics in (booster.best_score or {}).items():
            for metric, value in metrics.items():
                best_score[f"{split}_{metric}"] = float(value)

        result = RankerFitResult(
            best_iteration=int(booster.best_iteration or self.config.n_estimators),
            best_score=best_score,
            train_seconds=elapsed,
            feature_importance=importance,
        )
        logger.info("ranker.fit.complete", **result.as_dict())
        return result

    def score(self, features: np.ndarray) -> np.ndarray:
        """Score candidates. Higher ranks higher; values are not probabilities.

        LambdaRank output is an ordinal utility, not a calibrated probability,
        and the API reports it as ``ml_relevance_score`` rather than implying
        it is one.
        """
        self._require_fitted()
        scores: np.ndarray = self._booster.predict(
            features, num_iteration=self._booster.best_iteration
        )
        return scores.astype(np.float32)

    def rank(self, features: np.ndarray, item_ids: np.ndarray, k: int = 10) -> list[int]:
        """Return the top-``k`` item ids by model score."""
        scores = self.score(features)
        top_k = min(k, len(item_ids))
        order = np.argpartition(-scores, top_k - 1)[:top_k]
        order = order[np.argsort(-scores[order], kind="stable")]
        return [int(item_ids[i]) for i in order]

    def explain(self, features: np.ndarray, *, top_n: int = 5) -> list[dict[str, float]]:
        """Exact per-feature SHAP contributions for each row.

        TreeSHAP is exact for tree ensembles rather than a sampled
        approximation, so these are the model's genuine contributions to this
        specific score - which is what makes the UI's explanation panel a
        report rather than a story.

        Returns one dict per row, holding the ``top_n`` features by absolute
        contribution. Signed values are preserved: a feature that pushed the
        score *down* is as informative as one that pushed it up.
        """
        self._require_fitted()
        import shap

        if self._explainer is None:
            self._explainer = shap.TreeExplainer(self._booster)

        values = np.asarray(self._explainer.shap_values(features))
        if values.ndim == 3:  # some versions return (classes, rows, features)
            values = values[0]

        explanations: list[dict[str, float]] = []
        for row in values:
            ordered = np.argsort(-np.abs(row))[:top_n]
            explanations.append({self.feature_names[i]: round(float(row[i]), 6) for i in ordered})
        return explanations

    def warmup(self) -> dict[str, float]:
        """Pay the one-time initialisation costs before serving traffic.

        Two lazy costs otherwise land on whichever request arrives first:

        - ``shap.TreeExplainer`` walks the whole ensemble when constructed.
          Measured at ~900 ms for this 500-tree model, against ~2.7 ms for a
          warm explanation of three items.
        - LightGBM allocates its prediction buffers on the first ``predict``.

        Together they made the first request's ranking stage take 2.8 seconds
        while every later one took single-digit milliseconds - a cold-start
        cliff that a p99 latency target would catch only after deploy. Calling
        this at startup moves the cost out of the request path entirely.

        Returns the measured warm-up cost per component, which is logged so a
        regression in it is visible.
        """
        import time

        self._require_fitted()
        timings: dict[str, float] = {}
        probe = np.zeros((1, len(self.feature_names)), dtype=np.float32)

        started = time.perf_counter()
        self.score(probe)
        timings["predict_ms"] = (time.perf_counter() - started) * 1000.0

        started = time.perf_counter()
        try:
            self.explain(probe, top_n=1)
            timings["explainer_ms"] = (time.perf_counter() - started) * 1000.0
        except Exception as exc:  # noqa: BLE001
            # Explanations are optional; a service that cannot build them
            # should still serve recommendations.
            logger.warning("ranker.warmup_explainer_failed", error=str(exc)[:120])
            timings["explainer_ms"] = -1.0

        logger.info("ranker.warmup", **{k: round(v, 2) for k, v in timings.items()})
        return timings

    def save(self, path: Path) -> None:
        self._require_fitted()
        path.parent.mkdir(parents=True, exist_ok=True)
        self._booster.save_model(str(path), num_iteration=self._booster.best_iteration)
        # The feature order is part of the model's contract: scoring a matrix
        # whose columns are in a different order produces plausible garbage
        # rather than an error.
        path.with_suffix(".meta.json").write_text(
            json.dumps(
                {"feature_names": list(self.feature_names), "config": self.config.to_params()},
                indent=2,
            ),
            encoding="utf-8",
        )
        logger.info("ranker.saved", path=str(path))

    @classmethod
    def load(cls, path: Path) -> LambdaRanker:
        import lightgbm as lgb

        if not path.is_file():
            raise FileNotFoundError(f"No ranker at {path}. Run `mercury train ranker` first.")

        ranker = cls()
        ranker._booster = lgb.Booster(model_file=str(path))

        meta_path = path.with_suffix(".meta.json")
        if meta_path.is_file():
            meta = json.loads(meta_path.read_text(encoding="utf-8"))
            ranker.feature_names = tuple(meta["feature_names"])
        return ranker


__all__ = ["LambdaRanker", "RankerConfig", "RankerFitResult"]
