"""Unified Palestinian-ASR data pipeline.

One config-driven runner over five stages -- ``ingest``, ``clean``, ``assemble``,
``dialect``, ``split`` -- replacing the per-dataset scripts and notebooks documented
in ``DATA_CURATION.md``. See ``PIPELINE.md`` for the stage/script mapping.
"""
from __future__ import annotations

__all__ = ["__version__"]

__version__ = "1.0.0"
