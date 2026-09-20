"""Database engine and session management.

Connection pooling is configured explicitly rather than left at SQLAlchemy's
defaults, because the defaults assume a long-lived application server with
plenty of headroom, and this one shares a 16 GB machine with training, Redis
and a dev server.

``pool_pre_ping`` is on. Without it, a connection that the server closed while
idle - which Postgres does, and which a container restart guarantees - is
handed to the application and fails on first use. The symptom is an
intermittent ``OperationalError`` that is very hard to reproduce; the fix
costs one round trip per checkout.
"""

from __future__ import annotations

from collections.abc import Iterator
from contextlib import contextmanager

from sqlalchemy import Engine, create_engine, text
from sqlalchemy.orm import Session, sessionmaker

from mercury_rec.core.logging import get_logger
from mercury_rec.db.models import Base

logger = get_logger(__name__)

_engine: Engine | None = None
_session_factory: sessionmaker[Session] | None = None


def create_db_engine(
    dsn: str,
    *,
    pool_size: int = 10,
    max_overflow: int = 5,
    echo: bool = False,
) -> Engine:
    """Build an engine with pooling suited to this deployment.

    Args:
        dsn: SQLAlchemy URL, e.g. ``postgresql+psycopg://...``.
        pool_size: Connections held open. Postgres is configured for
            ``max_connections=50``, so this must stay well below it once
            multiplied by the worker count.
        max_overflow: Extra connections allowed under burst.
        echo: Log every statement. Extremely verbose; debugging only.
    """
    return create_engine(
        dsn,
        pool_size=pool_size,
        max_overflow=max_overflow,
        # Validate a pooled connection before handing it out. Postgres closes
        # idle connections and a container restart drops all of them; without
        # this the application receives a dead socket and fails on first use.
        pool_pre_ping=True,
        # Recycle below any typical proxy or server idle timeout, so the pool
        # rotates connections before something upstream closes them.
        pool_recycle=1800,
        echo=echo,
        future=True,
    )


def init_engine(dsn: str, **kwargs: object) -> Engine:
    """Create and cache the process-wide engine."""
    global _engine, _session_factory
    _engine = create_db_engine(dsn, **kwargs)  # type: ignore[arg-type]
    _session_factory = sessionmaker(bind=_engine, expire_on_commit=False)
    # Pool.size() exists on QueuePool but not on the Pool base class that
    # SQLAlchemy types the attribute as, so it is read defensively.
    logger.info("db.engine_created", pool=type(_engine.pool).__name__)
    return _engine


def get_engine() -> Engine:
    if _engine is None:
        raise RuntimeError("Database engine is not initialised. Call init_engine(dsn) first.")
    return _engine


@contextmanager
def session_scope() -> Iterator[Session]:
    """A transactional scope: commit on success, roll back on failure.

    ``expire_on_commit=False`` on the factory so objects stay usable after the
    block exits; the default expires every attribute, which then triggers a
    lazy reload against a closed session.
    """
    if _session_factory is None:
        raise RuntimeError("Database engine is not initialised. Call init_engine(dsn) first.")

    session = _session_factory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()


def create_all(engine: Engine | None = None) -> None:
    """Create every table.

    For tests and local bootstrapping only. Production schema changes go
    through Alembic, because ``create_all`` cannot alter an existing table and
    silently does nothing when one already exists - which looks like success.
    """
    Base.metadata.create_all(engine or get_engine())
    logger.info("db.tables_created", tables=len(Base.metadata.tables))


def healthcheck(engine: Engine | None = None) -> tuple[bool, str | None]:
    """Probe the database for ``/ready``."""
    try:
        with (engine or get_engine()).connect() as connection:
            connection.execute(text("SELECT 1"))
        return True, None
    except Exception as exc:  # noqa: BLE001
        return False, str(exc)[:160]


__all__ = [
    "create_all",
    "create_db_engine",
    "get_engine",
    "healthcheck",
    "init_engine",
    "session_scope",
]
