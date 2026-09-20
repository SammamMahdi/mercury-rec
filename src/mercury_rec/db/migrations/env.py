"""Alembic environment.

The connection URL is taken from ``MERCURY_POSTGRES_DSN`` rather than from
``alembic.ini``. Two reasons: a real credential must never live in a committed
file, and migrations should run against whatever the application is configured
for, so there is no way for the two to disagree.

``compare_type`` and ``compare_server_default`` are enabled. Without them
autogenerate silently ignores a column whose type or default changed, which
produces an empty migration that looks like "no changes needed" - the most
misleading possible outcome.
"""

from __future__ import annotations

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool

from mercury_rec.config.settings import get_settings
from mercury_rec.db.models import Base

config = context.config

if config.config_file_name is not None:
    fileConfig(config.config_file_name)

target_metadata = Base.metadata


def _database_url() -> str:
    """Resolve the URL from settings, failing with an actionable message."""
    settings = get_settings()
    if settings.postgres_dsn is None:
        raise RuntimeError(
            "MERCURY_POSTGRES_DSN is not set. Copy .env.example to .env, or start "
            "the database with `docker compose up -d postgres`."
        )
    return str(settings.postgres_dsn)


def run_migrations_offline() -> None:
    """Emit SQL without connecting.

    Useful for producing a script a DBA can review before it touches a
    database nobody wants an ORM connecting to directly.
    """
    context.configure(
        url=_database_url(),
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )
    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Apply migrations against a live connection."""
    configuration = config.get_section(config.config_ini_section, {})
    configuration["sqlalchemy.url"] = _database_url()

    connectable = engine_from_config(
        configuration,
        prefix="sqlalchemy.",
        # NullPool: a migration run is a short-lived process, and holding a
        # pool open past the last statement just delays the exit.
        poolclass=pool.NullPool,
    )

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            # Both default to False, and with them off autogenerate quietly
            # misses a changed type or default and writes an empty migration.
            compare_type=True,
            compare_server_default=True,
        )
        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
