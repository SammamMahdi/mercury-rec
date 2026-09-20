"""Tests for settings and YAML configuration loading."""

from __future__ import annotations

from pathlib import Path

import pytest
from pydantic import BaseModel

from mercury_rec.config.loader import ConfigError, config_hash, load_config, read_yaml
from mercury_rec.config.settings import Environment, Settings, get_settings


class _Sample(BaseModel):
    name: str
    count: int


def test_settings_have_no_secret_defaults() -> None:
    """Connection strings must come from the environment, never from source.

    A usable default DSN is how credentials end up committed: it works on the
    author's machine, so nobody notices until it is in the history.
    """
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.postgres_dsn is None
    assert settings.redis_dsn is None


def test_missing_dsn_raises_actionable_error() -> None:
    """The error must say what to set and how, not just that it is missing."""
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(RuntimeError, match="MERCURY_POSTGRES_DSN"):
        settings.require_postgres_dsn()
    with pytest.raises(RuntimeError, match="MERCURY_REDIS_DSN"):
        settings.require_redis_dsn()


def test_settings_are_frozen() -> None:
    """Configuration must not mutate at runtime.

    Mutable global settings make behaviour depend on import order, which is
    the kind of bug that only appears under a load test.
    """
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    with pytest.raises(Exception, match=r"frozen|immutable"):
        settings.api_port = 9999  # type: ignore[misc]


def test_get_settings_is_cached() -> None:
    """Every caller must observe the same configuration object."""
    assert get_settings() is get_settings()


def test_environment_is_local_by_default() -> None:
    settings = Settings(_env_file=None)  # type: ignore[call-arg]
    assert settings.environment is Environment.LOCAL
    assert settings.is_production_like is False


def test_read_yaml_rejects_non_mapping(tmp_path: Path) -> None:
    path = tmp_path / "bad.yaml"
    path.write_text("- just\n- a\n- list\n", encoding="utf-8")
    with pytest.raises(ConfigError, match="mapping"):
        read_yaml(path)


def test_read_yaml_missing_file_names_the_path(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        read_yaml(tmp_path / "absent.yaml")


def test_load_config_validates_against_schema(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    monkeypatch.setattr("mercury_rec.config.loader.CONFIG_DIR", tmp_path)
    (tmp_path / "sample.yaml").write_text("name: x\ncount: 3\n", encoding="utf-8")
    assert load_config("sample", _Sample) == _Sample(name="x", count=3)


def test_load_config_reports_the_file_on_validation_failure(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """A typo in YAML must name the file and the field, not just fail."""
    monkeypatch.setattr("mercury_rec.config.loader.CONFIG_DIR", tmp_path)
    (tmp_path / "sample.yaml").write_text("name: x\ncount: not-an-int\n", encoding="utf-8")
    with pytest.raises(ConfigError, match=r"sample\.yaml"):
        load_config("sample", _Sample)


def test_config_hash_is_stable_and_content_sensitive(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The hash ties a recorded metric to the exact config that produced it."""
    monkeypatch.setattr("mercury_rec.config.loader.CONFIG_DIR", tmp_path)
    path = tmp_path / "h.yaml"

    path.write_text("a: 1\n", encoding="utf-8")
    config_hash.cache_clear()
    first = config_hash("h")
    config_hash.cache_clear()
    assert config_hash("h") == first, "hash must be deterministic"

    path.write_text("a: 2\n", encoding="utf-8")
    config_hash.cache_clear()
    assert config_hash("h") != first, "hash must change when content changes"


def test_event_weights_config_is_valid_and_ordered() -> None:
    """The shipped event weights must load and respect intent ordering.

    The weights encode "more costly action implies stronger preference". If a
    future edit made a click outrank a purchase, every implicit-feedback model
    would quietly degrade while still training successfully.
    """
    from mercury_rec.config.loader import CONFIG_DIR

    weights = read_yaml(CONFIG_DIR / "events.yaml")["weights"]

    assert weights["impression"] < weights["click"] < weights["add_to_cart"]
    assert weights["add_to_cart"] < weights["purchase"] < weights["repeat_purchase"]
    assert weights["remove_from_cart"] < weights["add_to_cart"]
    assert all(value >= 0 for value in weights.values()), "weights must be non-negative"
