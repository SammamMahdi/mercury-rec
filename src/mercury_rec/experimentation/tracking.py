"""MLflow experiment tracking.

Every run records the dataset hash and the feature-schema version alongside
the parameters and metrics. That is the part that makes tracking useful
months later: a recorded NDCG of 0.0415 means nothing without knowing which
data produced it, and "the model from March" is not an answer when the
pipeline has been rebuilt since.

A **SQLite backend store** rather than a bare ``file://`` path, for two
reasons. The model registry does not work against a file backend at all. And
on Windows, MLflow's file store has a long history of bugs with drive letters
and backslashes in artifact URIs, which surface as confusing path errors
rather than as anything pointing at the cause.

Tracking is optional by design. If MLflow is unavailable the training
pipelines still run and still write their JSON artifacts; they simply do not
log. A tracking server being down should never stop a model being trained.
"""

from __future__ import annotations

import contextlib
import platform
from collections.abc import Iterator, Mapping
from pathlib import Path
from typing import Any

from mercury_rec.core.logging import get_logger

logger = get_logger(__name__)


def resolve_tracking_uri(configured: str | None = None) -> str:
    """Resolve a tracking URI that works on Windows.

    Paths are converted through ``as_uri``/POSIX form rather than
    concatenated, because a Windows path embedded raw in a URI produces
    backslash-and-drive-letter forms that MLflow parses inconsistently.
    """
    if configured:
        return configured

    database = Path("mlflow.db").resolve()
    return f"sqlite:///{database.as_posix()}"


@contextlib.contextmanager
def track_run(
    *,
    run_name: str,
    experiment: str = "mercury-rec",
    tracking_uri: str | None = None,
    params: Mapping[str, Any] | None = None,
    tags: Mapping[str, str] | None = None,
    enabled: bool = True,
) -> Iterator[Any]:
    """Context manager wrapping one MLflow run.

    Yields the active run, or ``None`` when tracking is disabled or MLflow is
    unavailable. Callers therefore guard on the yielded value, and a missing
    tracking server degrades to "no logging" rather than to a failed training
    job.
    """
    if not enabled:
        yield None
        return

    try:
        import mlflow
    except ImportError:
        logger.warning("tracking.mlflow_unavailable")
        yield None
        return

    try:
        mlflow.set_tracking_uri(resolve_tracking_uri(tracking_uri))
        mlflow.set_experiment(experiment)

        with mlflow.start_run(run_name=run_name) as run:
            # Recorded on every run so a result is reproducible from the run
            # alone rather than from someone's memory of the environment.
            mlflow.set_tags(
                {
                    "python_version": platform.python_version(),
                    "platform": platform.platform(),
                    **(tags or {}),
                }
            )
            if params:
                mlflow.log_params(_flatten(params))
            yield run
    except Exception as exc:  # noqa: BLE001
        # A tracking outage must not fail a training run.
        logger.warning("tracking.run_failed", error=str(exc)[:200])
        yield None


def log_metrics(metrics: Mapping[str, float], *, step: int | None = None) -> None:
    """Log metrics to the active run, if there is one."""
    try:
        import mlflow

        if mlflow.active_run() is None:
            return
        numeric = {
            name: float(value)
            for name, value in metrics.items()
            if isinstance(value, int | float) and not isinstance(value, bool)
        }
        mlflow.log_metrics(numeric, step=step)
    except Exception as exc:  # noqa: BLE001
        logger.debug("tracking.log_metrics_failed", error=str(exc)[:120])


def log_artifact(path: Path, *, artifact_path: str | None = None) -> None:
    """Attach a file to the active run."""
    try:
        import mlflow

        if mlflow.active_run() is None or not path.exists():
            return
        mlflow.log_artifact(str(path), artifact_path=artifact_path)
    except Exception as exc:  # noqa: BLE001
        logger.debug("tracking.log_artifact_failed", error=str(exc)[:120])


def _flatten(params: Mapping[str, Any], prefix: str = "") -> dict[str, Any]:
    """Flatten nested parameters.

    MLflow parameters are a flat string map, so a nested dict silently
    stringifies into an unsearchable blob. Flattening keeps each value
    individually filterable in the UI.
    """
    flat: dict[str, Any] = {}
    for key, value in params.items():
        name = f"{prefix}{key}"
        if isinstance(value, Mapping):
            flat.update(_flatten(value, prefix=f"{name}."))
        elif isinstance(value, list | tuple):
            flat[name] = ",".join(str(v) for v in value)
        else:
            flat[name] = value
    return flat


__all__ = ["log_artifact", "log_metrics", "resolve_tracking_uri", "track_run"]
