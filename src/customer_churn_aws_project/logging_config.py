"""Logging configuration shared by every pipeline module.

Logs are written to stdout in a human-readable format so that DVC surfaces
them directly while a stage runs. The default level is ``INFO`` and can be
overridden either with the ``--log-level`` CLI flag or the ``CHURN_LOG_LEVEL``
environment variable::

    CHURN_LOG_LEVEL=DEBUG dvc repro
    uv run python -m customer_churn_aws_project.stages train --log-level DEBUG
"""

from __future__ import annotations

import logging
import os
import sys

LOGGER_NAME = "customer_churn_aws_project"
LOG_FORMAT = "%(asctime)s | %(levelname)-7s | %(name)s | %(message)s"
DATE_FORMAT = "%Y-%m-%d %H:%M:%S"
DEFAULT_LEVEL = "INFO"

_VALID_LEVELS = ("DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL")


def resolve_level(level: str | int | None = None) -> str:
    """Resolve the effective log level (CLI arg > env var > default)."""
    if isinstance(level, int):
        return logging.getLevelName(level)
    if level:
        return str(level).upper()
    return os.getenv("CHURN_LOG_LEVEL", DEFAULT_LEVEL).upper()


def configure_logging(level: str | int | None = None) -> logging.Logger:
    """Configure and return the package logger. Safe to call repeatedly."""
    logger = logging.getLogger(LOGGER_NAME)
    logger.setLevel(resolve_level(level))
    logger.propagate = False

    for handler in list(logger.handlers):
        logger.removeHandler(handler)

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter(LOG_FORMAT, datefmt=DATE_FORMAT))
    logger.addHandler(handler)

    return logger


def get_logger(name: str) -> logging.Logger:
    """Return a child logger that propagates to the package logger."""
    return logging.getLogger(name)
