"""Config loading for the unified pipeline.

A config is a YAML document with three top-level keys:

.. code-block:: yaml

    name: sample
    vars:                 # free-form string substitutions usable as {name}
      root: /path/to/run
    stages:               # ordered list; each entry has `id` and `stage`
      - id: ingest
        stage: ingest
        ...

Every string value anywhere in the document is ``str.format``-ed against ``vars``
(plus ``repo_root``), so paths can be written once and reused.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parent.parent


@dataclass
class StageSpec:
    """One entry from the config's ``stages`` list."""

    id: str
    stage: str
    enabled: bool = True
    options: dict[str, Any] = field(default_factory=dict)

    def get(self, key: str, default: Any = None) -> Any:
        return self.options.get(key, default)

    def require(self, key: str) -> Any:
        if key not in self.options:
            raise KeyError(f"Stage {self.id!r} ({self.stage}) is missing required key {key!r}")
        return self.options[key]

    def path(self, key: str, default: Any = None) -> Path | None:
        value = self.options.get(key, default)
        return None if value is None else Path(value)

    def require_path(self, key: str) -> Path:
        return Path(self.require(key))


@dataclass
class PipelineConfig:
    name: str
    vars: dict[str, str]
    stages: list[StageSpec]
    source_path: Path

    def stage_ids(self) -> list[str]:
        return [s.id for s in self.stages]

    def select(self, only: list[str] | None, skip: list[str] | None) -> list[StageSpec]:
        chosen = [s for s in self.stages if s.enabled]
        if only:
            unknown = set(only) - set(self.stage_ids())
            if unknown:
                raise SystemExit(
                    f"Unknown stage id(s): {sorted(unknown)}. Known: {self.stage_ids()}"
                )
            chosen = [s for s in self.stages if s.id in only]
        if skip:
            chosen = [s for s in chosen if s.id not in skip]
        return chosen


def _substitute(value: Any, mapping: dict[str, str]) -> Any:
    if isinstance(value, str):
        try:
            return value.format(**mapping)
        except KeyError as exc:
            raise SystemExit(
                f"Config references undefined var {exc} in value: {value!r}"
            ) from None
    if isinstance(value, list):
        return [_substitute(v, mapping) for v in value]
    if isinstance(value, dict):
        return {k: _substitute(v, mapping) for k, v in value.items()}
    return value


def _resolve_vars(raw_vars: dict[str, str]) -> dict[str, str]:
    """Resolve vars that reference other vars, iterating to a fixed point."""
    mapping = {"repo_root": str(REPO_ROOT), **{k: str(v) for k, v in raw_vars.items()}}
    for _ in range(10):
        updated = {k: _substitute(v, mapping) for k, v in mapping.items()}
        if updated == mapping:
            return updated
        mapping = updated
    raise SystemExit("Config vars appear to reference each other cyclically")


def load_config(path: str | Path) -> PipelineConfig:
    path = Path(path)
    raw = yaml.safe_load(path.read_text(encoding="utf-8")) or {}

    mapping = _resolve_vars(raw.get("vars") or {})
    raw = _substitute(raw, mapping)

    stages: list[StageSpec] = []
    seen_ids: set[str] = set()
    for entry in raw.get("stages") or []:
        if "id" not in entry or "stage" not in entry:
            raise SystemExit(f"Every stage needs 'id' and 'stage' keys; got {entry!r}")
        stage_id = entry["id"]
        if stage_id in seen_ids:
            raise SystemExit(f"Duplicate stage id: {stage_id!r}")
        seen_ids.add(stage_id)
        options = {k: v for k, v in entry.items() if k not in {"id", "stage", "enabled"}}
        stages.append(
            StageSpec(
                id=stage_id,
                stage=entry["stage"],
                enabled=bool(entry.get("enabled", True)),
                options=options,
            )
        )

    if not stages:
        raise SystemExit(f"Config {path} defines no stages")

    return PipelineConfig(
        name=raw.get("name", path.stem),
        vars=mapping,
        stages=stages,
        source_path=path.resolve(),
    )
