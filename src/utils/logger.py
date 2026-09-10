# ============================================================
# Shared Logging Utility
# ============================================================
"""
Project-wide logging setup.

Usage:
    from src.utils.logger import get_logger

    logger = get_logger(__name__)
    logger.info("Pipeline started")
    logger.warning("Missing data for ticker %s", ticker)
"""

from __future__ import annotations

import logging
import os
import sys

_LOG_FORMAT = "%(asctime)s | %(levelname)-8s | %(name)s:%(funcName)s:%(lineno)d — %(message)s"
_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"

# Allow override via env var (set in .env or shell)
_DEFAULT_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()


def get_logger(name: str, level: str | None = None) -> logging.Logger:
    """Return a configured logger.

    Parameters
    ----------
    name : str
        Logger name — typically ``__name__`` of the calling module.
    level : str | None
        Override log level (DEBUG, INFO, WARNING, ERROR, CRITICAL).
        Falls back to the ``LOG_LEVEL`` env var, then INFO.

    Returns
    -------
    logging.Logger
    """
    logger = logging.getLogger(name)

    if not logger.handlers:
        resolved_level = getattr(logging, (level or _DEFAULT_LEVEL), logging.INFO)
        logger.setLevel(resolved_level)

        # Console handler (stderr)
        console = logging.StreamHandler(sys.stderr)
        console.setLevel(resolved_level)
        console.setFormatter(logging.Formatter(_LOG_FORMAT, datefmt=_DATE_FORMAT))
        logger.addHandler(console)

        # Prevent duplicate logs if root logger is also configured
        logger.propagate = False

    return logger
