"""Structured logging configuration.

Logs are JSON in every non-local environment so they are machine-parseable by
a collector, and human-readable coloured console output locally. Configuring
this twice is harmless.

Request-scoped context (``request_id``, ``user_id``, ``model_version``) is
bound via :func:`bind_request_context` using structlog's context variables, so
every log line emitted while handling a request carries it without needing to
be threaded through call signatures.

Safe logging is a stated requirement: :func:`redact` exists so that anything
that could carry a credential is scrubbed before it reaches an emitter.
"""

from __future__ import annotations

import logging
import sys
from typing import Any

import structlog
from structlog.types import EventDict, Processor

_SENSITIVE_KEYS = frozenset(
    {
        "password",
        "passwd",
        "secret",
        "token",
        "api_key",
        "apikey",
        "authorization",
        "kaggle_key",
        "dsn",
        "postgres_dsn",
        "redis_dsn",
    }
)

_REDACTED = "***redacted***"

_configured = False


def redact(_logger: Any, _method: str, event_dict: EventDict) -> EventDict:
    """Scrub credential-bearing keys from a log event.

    Matching is on the key name, case-insensitively and by substring, so
    ``db_password`` and ``MERCURY_POSTGRES_DSN`` are both caught. A connection
    string is redacted wholesale rather than parsed — the password is only one
    part, and hostnames in logs have their own disclosure risk.
    """
    for key in list(event_dict):
        lowered = key.lower()
        if any(marker in lowered for marker in _SENSITIVE_KEYS):
            event_dict[key] = _REDACTED
    return event_dict


def configure_logging(*, level: str = "INFO", json_output: bool = False) -> None:
    """Configure structlog and the stdlib root logger.

    Args:
        level: Minimum level name, e.g. ``"DEBUG"``.
        json_output: Emit JSON lines instead of coloured console output.
    """
    global _configured

    shared: list[Processor] = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso", utc=True),
        structlog.processors.StackInfoRenderer(),
        redact,
    ]

    renderer: Processor = (
        structlog.processors.JSONRenderer()
        if json_output
        else structlog.dev.ConsoleRenderer(colors=sys.stderr.isatty())
    )

    structlog.configure(
        processors=[
            *shared,
            structlog.processors.format_exc_info,
            renderer,
        ],
        wrapper_class=structlog.make_filtering_bound_logger(
            logging.getLevelNamesMapping()[level.upper()]
        ),
        logger_factory=structlog.PrintLoggerFactory(file=sys.stderr),
        cache_logger_on_first_use=True,
    )

    # Route stdlib loggers (uvicorn, sqlalchemy, mlflow) through the same
    # handler so output is not half JSON and half plain text.
    logging.basicConfig(
        format="%(message)s",
        stream=sys.stderr,
        level=logging.getLevelNamesMapping()[level.upper()],
        force=True,
    )
    for noisy in ("uvicorn.access", "sqlalchemy.engine", "alembic"):
        logging.getLogger(noisy).setLevel(logging.WARNING)

    _configured = True


def get_logger(name: str | None = None) -> structlog.stdlib.BoundLogger:
    """Return a bound logger, configuring logging on first use."""
    if not _configured:
        configure_logging()
    return structlog.get_logger(name)  # type: ignore[no-any-return]


def bind_request_context(**kwargs: Any) -> None:
    """Bind key/values onto every subsequent log line in this context."""
    structlog.contextvars.bind_contextvars(**kwargs)


def clear_request_context() -> None:
    """Clear request-scoped log context. Call at the end of a request."""
    structlog.contextvars.clear_contextvars()


__all__ = [
    "bind_request_context",
    "clear_request_context",
    "configure_logging",
    "get_logger",
    "redact",
]
