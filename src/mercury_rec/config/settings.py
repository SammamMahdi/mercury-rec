"""Environment-driven application settings.

All deployment-varying values — connection strings, credentials, ports, cache
TTLs — arrive from the environment or a local ``.env`` file. No secret has a
usable default; ``.env.example`` documents every key with placeholder values.

Model and pipeline hyperparameters deliberately do **not** live here. Those
belong in the versioned YAML under ``configs/`` (see
:mod:`mercury_rec.config.loader`), because they are experiment inputs that must
be reproducible from the repository, not environment state. The dividing line:
*if changing it changes a metric, it belongs in YAML; if changing it changes
where the process connects, it belongs here.*
"""

from __future__ import annotations

from enum import StrEnum
from functools import lru_cache
from pathlib import Path

from pydantic import Field, PostgresDsn, RedisDsn, computed_field
from pydantic_settings import BaseSettings, SettingsConfigDict

PROJECT_ROOT = Path(__file__).resolve().parents[3]


class Environment(StrEnum):
    LOCAL = "local"
    CI = "ci"
    DOCKER = "docker"


class Settings(BaseSettings):
    """Runtime configuration, resolved from environment variables."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        env_prefix="MERCURY_",
        extra="ignore",
        frozen=True,
    )

    environment: Environment = Environment.LOCAL
    debug: bool = False

    # --- paths -------------------------------------------------------------
    project_root: Path = PROJECT_ROOT
    data_dir: Path = PROJECT_ROOT / "data"
    artifacts_dir: Path = PROJECT_ROOT / "artifacts"

    # --- api ---------------------------------------------------------------
    api_host: str = "0.0.0.0"  # noqa: S104  (container-facing; compose maps the port)
    api_port: int = 8000
    api_workers: int = 1
    """Workers for the serving process.

    Deliberately 1 by default. On Windows each uvicorn worker is a fresh
    ``spawn`` that re-loads the full model bundle, so a high worker count
    exhausts memory on a 16 GB machine long before it improves throughput.
    Benchmarks record the value actually used.
    """

    cors_origins: list[str] = Field(default_factory=lambda: ["http://localhost:3000"])
    max_request_bytes: int = 1_048_576
    allow_as_of_override: bool = False
    """Permit an ``as_of`` query parameter on the recommendation endpoint.

    Powers the frontend's User Journey replay. It lets a caller ask what the
    system would have returned at an arbitrary past time, so it stays off
    outside local and demo environments.
    """

    # --- postgres ----------------------------------------------------------
    postgres_dsn: PostgresDsn | None = None
    db_pool_size: int = 10
    db_max_overflow: int = 5
    db_echo: bool = False

    # --- redis -------------------------------------------------------------
    redis_dsn: RedisDsn | None = None
    cache_enabled: bool = True
    cache_ttl_seconds: int = 300
    """Hard TTL for a cached recommendation payload."""
    cache_soft_ttl_seconds: int = 120
    """Age past which a cached payload is served but refreshed in the
    background (stale-while-revalidate), so a popular user cannot trigger a
    cache stampede."""
    cache_jitter_ratio: float = 0.15
    """Random TTL spread, preventing synchronised mass expiry."""

    # --- mlflow ------------------------------------------------------------
    mlflow_tracking_uri: str | None = None
    mlflow_experiment: str = "mercury-rec"

    # --- data --------------------------------------------------------------
    random_seed: int = 42
    kaggle_dataset: str = "retailrocket/ecommerce-dataset"

    @computed_field  # type: ignore[prop-decorator]
    @property
    def is_production_like(self) -> bool:
        """Whether unsafe conveniences must stay disabled."""
        return self.environment is not Environment.LOCAL

    def require_postgres_dsn(self) -> str:
        """Return the Postgres DSN or explain precisely what is missing."""
        if self.postgres_dsn is None:
            raise RuntimeError(
                "MERCURY_POSTGRES_DSN is not set. Copy .env.example to .env and "
                "fill it in, or start the database with `docker compose up postgres`."
            )
        return str(self.postgres_dsn)

    def require_redis_dsn(self) -> str:
        """Return the Redis DSN or explain precisely what is missing."""
        if self.redis_dsn is None:
            raise RuntimeError(
                "MERCURY_REDIS_DSN is not set. Copy .env.example to .env and "
                "fill it in, or start the cache with `docker compose up redis`."
            )
        return str(self.redis_dsn)


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return the process-wide settings singleton.

    Cached so that every FastAPI dependency and CLI command observes identical
    configuration. Tests clear it via ``get_settings.cache_clear()``.
    """
    return Settings()


__all__ = ["Environment", "Settings", "get_settings"]
