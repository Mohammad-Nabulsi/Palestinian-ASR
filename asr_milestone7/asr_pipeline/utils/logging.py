"""Logging helpers for the ASR pipeline."""

from __future__ import annotations

import logging


def get_logger(name: str = "asr_pipeline", level: int | str = logging.INFO) -> logging.Logger:
    """Return a simple stdout logger without duplicate handlers."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.propagate = False

    if not logger.handlers:
        handler = logging.StreamHandler()
        handler.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s"))
        logger.addHandler(handler)

    return logger
