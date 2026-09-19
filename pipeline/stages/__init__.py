"""Stage registry for the unified pipeline."""
from __future__ import annotations

from typing import Callable

from ..config import StageSpec
from ..context import RunContext, StageResult
from . import assemble, clean, dialect, ingest, speaker_select, split

StageFn = Callable[[RunContext, StageSpec], StageResult]

STAGES: dict[str, StageFn] = {
    "ingest": ingest.run,
    "clean": clean.run,
    "assemble": assemble.run,
    "dialect": dialect.run,
    "speaker_select": speaker_select.run,
    "split": split.run,
}


def get_stage(name: str) -> StageFn:
    try:
        return STAGES[name]
    except KeyError:
        raise SystemExit(
            f"Unknown stage type {name!r}. Known: {sorted(STAGES)}"
        ) from None
