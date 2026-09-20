"""Typed loading of the versioned YAML configuration under ``configs/``.

Experiment inputs live in YAML rather than in code or environment variables so
that a run is reproducible from a repository checkout alone: the config file is
committed, hashed, and logged to MLflow alongside the metrics it produced.

Each config is parsed into a frozen pydantic model, so a typo in a YAML key is
a load-time error naming the offending field, rather than a silent default that
quietly changes a result.
"""

from __future__ import annotations

import hashlib
from functools import cache
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from mercury_rec.config.settings import PROJECT_ROOT

CONFIG_DIR = PROJECT_ROOT / "configs"


class ConfigError(RuntimeError):
    """Raised when a configuration file is missing or malformed."""


def read_yaml(path: Path) -> dict[str, Any]:
    """Read a YAML mapping, with errors that identify the file."""
    if not path.is_file():
        raise ConfigError(
            f"Configuration file not found: {path}\n"
            f"Expected it under {CONFIG_DIR}. See configs/ in the repository."
        )
    try:
        loaded = yaml.safe_load(path.read_text(encoding="utf-8"))
    except yaml.YAMLError as exc:
        raise ConfigError(f"{path} is not valid YAML: {exc}") from exc

    if loaded is None:
        return {}
    if not isinstance(loaded, dict):
        raise ConfigError(
            f"{path} must contain a mapping at the top level, got {type(loaded).__name__}."
        )
    return loaded


def load_config[ModelT: BaseModel](name: str, schema: type[ModelT]) -> ModelT:
    """Load ``configs/{name}.yaml`` and validate it against ``schema``.

    Args:
        name: File stem, e.g. ``"retrieval"`` for ``configs/retrieval.yaml``.
        schema: The pydantic model describing the file's expected shape.
    """
    path = CONFIG_DIR / f"{name}.yaml"
    payload = read_yaml(path)
    try:
        return schema.model_validate(payload)
    except Exception as exc:  # pydantic.ValidationError, re-raised with context
        raise ConfigError(f"{path} failed validation against {schema.__name__}:\n{exc}") from exc


@cache
def config_hash(name: str) -> str:
    """Stable 12-character digest of a config file's bytes.

    Logged with every experiment so a recorded metric can be traced back to the
    exact configuration that produced it. Hashing raw bytes rather than parsed
    values means a comment-only edit also changes the hash — which is the
    conservative choice for provenance.
    """
    path = CONFIG_DIR / f"{name}.yaml"
    if not path.is_file():
        raise ConfigError(f"Cannot hash missing config: {path}")
    digest = hashlib.blake2b(path.read_bytes(), digest_size=6)
    return digest.hexdigest()


__all__ = ["CONFIG_DIR", "ConfigError", "config_hash", "load_config", "read_yaml"]
