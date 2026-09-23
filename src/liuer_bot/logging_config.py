"""Minimal standard-library console logging configuration."""

from __future__ import annotations

import logging

from .config import VALID_LOG_LEVELS


def configure_logging(log_level: str) -> None:
    """Configure console logging for a validated Phase 0 log level."""

    normalized = log_level.strip().upper()
    if normalized not in VALID_LOG_LEVELS:
        raise ValueError("invalid log level")
    logging.basicConfig(
        level=getattr(logging, normalized),
        format="%(levelname)s %(name)s: %(message)s",
        force=True,
    )
    # Keep application DEBUG diagnostics while preventing low-level HTTP wire
    # tracing from handling authorization headers or large request bodies.
    logging.getLogger("httpcore").setLevel(logging.INFO)
