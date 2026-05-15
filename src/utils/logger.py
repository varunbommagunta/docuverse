"""Structured logging setup using structlog.

Provides a single configure_logging() function that must be called once at
application startup (in FastAPI's lifespan). After that, any module can call
structlog.get_logger() to get a bound logger that emits JSON to stdout.

JSON output makes logs trivially parseable by log aggregators (Datadog,
Loki, CloudWatch). In development you can pipe through `python -m json.tool`
or `jq` for readable output.
"""

import logging
import sys

import structlog

from config.settings import get_settings


def configure_logging() -> None:
    """Configure structlog for JSON output at the level specified in Settings.

    Call exactly once, at process startup. Idempotent — safe to call in tests.
    """
    settings = get_settings()
    log_level = getattr(logging, settings.log_level.upper(), logging.INFO)

    logging.basicConfig(
        format="%(message)s",
        stream=sys.stdout,
        level=log_level,
    )

    structlog.configure(
        processors=[
            structlog.contextvars.merge_contextvars,
            structlog.stdlib.add_log_level,
            structlog.processors.TimeStamper(fmt="iso"),
            structlog.processors.StackInfoRenderer(),
            structlog.processors.format_exc_info,
            structlog.processors.JSONRenderer(),
        ],
        wrapper_class=structlog.make_filtering_bound_logger(log_level),
        context_class=dict,
        logger_factory=structlog.PrintLoggerFactory(),
        cache_logger_on_first_use=True,
    )
