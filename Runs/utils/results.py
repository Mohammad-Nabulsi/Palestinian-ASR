"""Shared result-file helpers for all model runs."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def upsert_model_stage_result(
    results_path: str | Path,
    model_key: str,
    stage: str,
    metrics: dict[str, Any],
    predictions_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Insert or update one model stage in a shared results JSON file."""

    path = Path(results_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = _read_results(path)
    payload.setdefault("schema_version", 1)
    payload["updated_at"] = datetime.now(timezone.utc).isoformat()
    model_payload = payload.setdefault("models", {}).setdefault(model_key, {})
    model_payload[stage] = {
        "metrics": metrics,
        "predictions_path": str(predictions_path) if predictions_path else None,
        "metadata": metadata or {},
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    with path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    return path


def _read_results(path: Path) -> dict[str, Any]:
    """Read an existing results file or return an empty payload."""

    if not path.exists():
        return {}
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)
