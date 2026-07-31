"""Run context: logging, report writing, and stage result bookkeeping."""
from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from .config import PipelineConfig


@dataclass
class StageResult:
    """What a stage reports back to the runner."""

    stage_id: str
    stage: str
    status: str = "ok"
    counts: dict[str, Any] = field(default_factory=dict)
    outputs: dict[str, str] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    duration_sec: float = 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "stage_id": self.stage_id,
            "stage": self.stage,
            "status": self.status,
            "duration_sec": round(self.duration_sec, 3),
            "counts": self.counts,
            "outputs": self.outputs,
            "notes": self.notes,
        }


class RunContext:
    """Holds the run directory, the shared log, and per-stage report output."""

    def __init__(self, config: PipelineConfig, run_root: Path, dry_run: bool = False) -> None:
        self.config = config
        self.run_root = run_root
        self.dry_run = dry_run
        self.reports_dir = run_root / "reports"
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        self.log_path = self.reports_dir / "run.log"
        self.results: list[StageResult] = []
        self._stage_id: str | None = None

    # ---- logging -------------------------------------------------------

    def log(self, msg: str) -> None:
        ts = time.strftime("%Y-%m-%d %H:%M:%S")
        prefix = f"[{ts}]"
        if self._stage_id:
            prefix += f" [{self._stage_id}]"
        line = f"{prefix} {msg}"
        print(line, flush=True)
        with self.log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")

    def bind_stage(self, stage_id: str | None) -> None:
        self._stage_id = stage_id

    # ---- reports -------------------------------------------------------

    def stage_reports_dir(self, stage_id: str) -> Path:
        path = self.reports_dir / stage_id
        path.mkdir(parents=True, exist_ok=True)
        return path

    def write_json(self, path: Path, payload: Any) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(
            json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
            encoding="utf-8",
        )
        return path

    def write_text(self, path: Path, text: str) -> Path:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(text if text.endswith("\n") else text + "\n", encoding="utf-8")
        return path

    def record(self, result: StageResult) -> None:
        self.results.append(result)
        self.write_json(
            self.stage_reports_dir(result.stage_id) / "stage_result.json",
            result.to_dict(),
        )

    def write_manifest(self) -> Path:
        payload = {
            "pipeline": self.config.name,
            "config": str(self.config.source_path),
            "run_root": str(self.run_root),
            "vars": self.config.vars,
            "stages": [r.to_dict() for r in self.results],
            "generated_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        }
        return self.write_json(self.reports_dir / "pipeline_manifest.json", payload)
