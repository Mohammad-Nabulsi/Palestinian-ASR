"""High-level prediction API and prediction caching.

Milestone 6 keeps one public ``predict(...)`` function for both base and tuned
adapter prediction. This module owns cache keys, JSONL paths, tuned/base schema
validation, and output writing. Architecture-specific model/checkpoint behavior
stays inside adapters.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from asr_pipeline.config import ASRConfig
from asr_pipeline.registry import get_model_family
from asr_pipeline.utils.hashing import config_hash, stable_hash
from asr_pipeline.utils.io import ensure_dir, read_json, read_jsonl, write_jsonl
from asr_pipeline.utils.logging import get_logger

LOGGER = get_logger(__name__)

PREDICTION_SCHEMA_VERSION = 2
PREDICTION_REQUIRED_COLUMNS = {
    "uid",
    "reference",
    "prediction",
    "model_name",
    "model_family",
    "tuned_or_base",
    "run_name",
    "config_hash",
}

TUNED_PREDICTION_REQUIRED_COLUMNS = PREDICTION_REQUIRED_COLUMNS | {
    "tuned_adapter_path",
    "training_config_hash",
}


def _safe_name(value: str) -> str:
    """Return a filesystem-safe identifier while keeping names readable."""
    return (
        value.replace("/", "__")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("\\", "__")
    )


def _rows_from_prepared_data(prepared_data: Any, split: str) -> list[dict[str, Any]]:
    """Accept either the milestone-2 prepare result or a raw prepared split dict."""
    if isinstance(prepared_data, Mapping) and "prepared" in prepared_data:
        prepared = prepared_data["prepared"]
    else:
        prepared = prepared_data

    if not isinstance(prepared, Mapping):
        raise TypeError(
            "prepared_data must be the dict returned by prepare_data_and_collator() "
            "or a mapping of split name to prepared rows."
        )
    if split not in prepared:
        raise ValueError(f"Prepared data does not contain split {split!r}.")

    rows = prepared[split]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError(f"Prepared split {split!r} must be a sequence of row dictionaries.")

    normalized = [dict(row) for row in rows]
    for idx, row in enumerate(normalized):
        if "uid" not in row:
            raise ValueError(f"Prepared row {idx} in split {split!r} is missing 'uid'.")
        if "text" not in row and "reference" not in row:
            raise ValueError(
                f"Prepared row {idx} in split {split!r} is missing reference text "
                "('text' or 'reference')."
            )
    return normalized


def _prepared_identity(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Create a small deterministic identity for a prepared split."""
    return {
        "row_count": len(rows),
        "rows": [
            {
                "uid": row.get("uid"),
                "reference": row.get("reference", row.get("text")),
                "audio_path": row.get("audio_path", row.get("input_audio_path")),
                "duration": row.get("duration"),
                "sample_rate": row.get("sample_rate"),
                "architecture": row.get("architecture"),
            }
            for row in rows
        ],
    }


def _file_sha256(path: Path) -> str:
    """Hash a single file for tuned-checkpoint cache invalidation."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _path_fingerprint(path: str | Path | None) -> dict[str, Any] | None:
    """Return a stable fingerprint for a tuned adapter path.

    Smoke checkpoints are tiny, so we include content hashes for files. For real
    checkpoints this still works, but later milestones may replace it with a
    manifest-sidecar hash to avoid hashing very large weight files repeatedly.
    """
    if path is None:
        return None
    resolved = Path(path).expanduser().resolve()
    if not resolved.exists():
        raise FileNotFoundError(f"Tuned adapter path does not exist: {resolved}")

    if resolved.is_file():
        stat = resolved.stat()
        return {
            "path": str(resolved),
            "kind": "file",
            "size": stat.st_size,
            "mtime_ns": stat.st_mtime_ns,
            "sha256": _file_sha256(resolved),
        }

    files: list[dict[str, Any]] = []
    for file_path in sorted(p for p in resolved.rglob("*") if p.is_file()):
        stat = file_path.stat()
        files.append(
            {
                "relative_path": str(file_path.relative_to(resolved)),
                "size": stat.st_size,
                "mtime_ns": stat.st_mtime_ns,
                "sha256": _file_sha256(file_path),
            }
        )
    return {
        "path": str(resolved),
        "kind": "directory",
        "file_count": len(files),
        "files": files,
    }


def _training_config_path_for_adapter(tuned_adapter_path: str | Path | None) -> Path | None:
    """Find the training_config.json associated with a tuned checkpoint."""
    if tuned_adapter_path is None:
        return None
    path = Path(tuned_adapter_path).expanduser().resolve()
    candidates = [
        path / "training_config.json",
        path.parent / "training_config.json",
        path.parent.parent / "training_config.json",
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return None


def training_config_hash_for_adapter(tuned_adapter_path: str | Path | None) -> str | None:
    """Return the stored training config hash or a file-content fallback hash."""
    training_config_path = _training_config_path_for_adapter(tuned_adapter_path)
    if training_config_path is None:
        return None
    try:
        payload = read_json(training_config_path)
        stored = payload.get("config_hash")
        if stored:
            return str(stored)
    except Exception:
        # Fall back to the content hash below; callers should not need to know
        # the exact internal shape of training_config.json.
        pass
    return _file_sha256(training_config_path)[:16]


def prediction_cache_key(
    config: ASRConfig,
    *,
    family: str,
    split: str,
    rows: Sequence[Mapping[str, Any]],
    tuned_adapter_path: str | Path | None = None,
    tuned_adapter_fingerprint: Mapping[str, Any] | None = None,
    training_config_hash: str | None = None,
) -> str:
    """Create the prediction cache key for base or tuned predictions."""
    tuned_or_base = "tuned" if tuned_adapter_path else "base"
    payload = {
        "schema_version": PREDICTION_SCHEMA_VERSION,
        "model_family": family,
        "model_name": config.model_name,
        "split": split,
        "tuned_or_base": tuned_or_base,
        "tuned_adapter_path": str(Path(tuned_adapter_path).expanduser().resolve()) if tuned_adapter_path else None,
        "tuned_adapter_fingerprint": tuned_adapter_fingerprint,
        "training_config_hash": training_config_hash,
        "run_name": config.run_name,
        "config_hash": config_hash(config),
        "prepared_identity": _prepared_identity(rows),
    }
    return stable_hash(payload, length=16)


def prediction_output_path(
    config: ASRConfig,
    *,
    family: str,
    split: str,
    key: str,
    tuned_adapter_path: str | Path | None = None,
) -> Path:
    """Return the JSONL path for base or tuned predictions."""
    tuned_or_base = "tuned" if tuned_adapter_path else "base"
    safe_model = _safe_name(config.model_name)
    safe_run = _safe_name(config.run_name)
    filename = f"{family}__{safe_model}__{safe_run}__{split}__{tuned_or_base}__{key}.jsonl"
    return Path(config.output_dir) / "predictions" / tuned_or_base / filename


def _normalize_prediction_rows(
    raw_predictions: Sequence[Mapping[str, Any]],
    *,
    source_rows: Sequence[Mapping[str, Any]],
    config: ASRConfig,
    family: str,
    tuned_adapter_path: str | Path | None,
) -> list[dict[str, Any]]:
    """Enforce the public prediction JSONL schema."""
    tuned_or_base = "tuned" if tuned_adapter_path else "base"
    cfg_hash = config_hash(config)
    by_uid = {str(row.get("uid")): row for row in source_rows}

    output: list[dict[str, Any]] = []
    for idx, pred in enumerate(raw_predictions):
        uid = str(pred.get("uid", ""))
        if not uid:
            raise ValueError(f"Prediction row {idx} is missing 'uid'.")
        source = by_uid.get(uid, {})
        reference = pred.get("reference", source.get("reference", source.get("text", "")))
        prediction_text = pred.get("prediction")
        if prediction_text is None:
            raise ValueError(f"Prediction row {idx} for uid {uid!r} is missing 'prediction'.")

        row = {
            "uid": uid,
            "reference": str(reference),
            "prediction": str(prediction_text),
            "model_name": config.model_name,
            "model_family": family,
            "tuned_or_base": tuned_or_base,
            "run_name": config.run_name,
            "config_hash": cfg_hash,
        }
        if tuned_adapter_path:
            row["tuned_adapter_path"] = str(Path(tuned_adapter_path).expanduser().resolve())
            row["training_config_hash"] = training_config_hash_for_adapter(tuned_adapter_path) or "unknown"

        required = TUNED_PREDICTION_REQUIRED_COLUMNS if tuned_adapter_path else PREDICTION_REQUIRED_COLUMNS
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(f"Internal prediction schema error. Missing: {', '.join(missing)}")
        output.append(row)

    return output


def predict(
    config: ASRConfig,
    adapter: Any,
    prepared_data: Any,
    split: str = "test",
    tuned_adapter_path: str | Path | None = None,
) -> dict[str, Any]:
    """Run or load cached predictions through the selected architecture adapter.

    The notebook should call this function, not adapter-specific prediction code.
    In smoke mode the adapters return deterministic mock predictions without
    downloading models or running real inference.
    """
    family = getattr(adapter, "family", None) or get_model_family(config.model_name)
    resolved_family = get_model_family(config.model_name)
    if family != resolved_family:
        raise ValueError(
            f"Adapter family {family!r} does not match model_name family {resolved_family!r}."
        )

    rows = _rows_from_prepared_data(prepared_data, split=split)
    tuned_adapter_fingerprint = _path_fingerprint(tuned_adapter_path) if tuned_adapter_path else None
    training_cfg_hash = training_config_hash_for_adapter(tuned_adapter_path) if tuned_adapter_path else None
    key = prediction_cache_key(
        config,
        family=family,
        split=split,
        rows=rows,
        tuned_adapter_path=tuned_adapter_path,
        tuned_adapter_fingerprint=tuned_adapter_fingerprint,
        training_config_hash=training_cfg_hash,
    )
    output_path = prediction_output_path(
        config,
        family=family,
        split=split,
        key=key,
        tuned_adapter_path=tuned_adapter_path,
    )

    if output_path.exists():
        cached_rows = read_jsonl(output_path)
        LOGGER.info("Loaded prediction cache: %s", output_path)
        return {
            "predictions": cached_rows,
            "metadata": {
                "predicted_new": False,
                "loaded_from_cache": True,
                "path": str(output_path),
                "cache_key": key,
                "model_family": family,
                "model_name": config.model_name,
                "split": split,
                "tuned_or_base": "tuned" if tuned_adapter_path else "base",
                "config_hash": config_hash(config),
                "tuned_adapter_path": str(Path(tuned_adapter_path).expanduser().resolve()) if tuned_adapter_path else None,
                "training_config_hash": training_cfg_hash,
            },
        }

    raw_predictions = adapter.predict(
        rows,
        split=split,
        tuned_adapter_path=tuned_adapter_path,
        smoke_mode=config.smoke_mode,
        local_model_cache_dir=config.local_model_cache_dir,
    )
    normalized = _normalize_prediction_rows(
        raw_predictions,
        source_rows=rows,
        config=config,
        family=family,
        tuned_adapter_path=tuned_adapter_path,
    )
    ensure_dir(output_path.parent)
    write_jsonl(output_path, normalized)
    LOGGER.info("Created prediction cache: %s", output_path)

    return {
        "predictions": normalized,
        "metadata": {
            "predicted_new": True,
            "loaded_from_cache": False,
            "path": str(output_path),
            "cache_key": key,
            "model_family": family,
            "model_name": config.model_name,
            "split": split,
            "tuned_or_base": "tuned" if tuned_adapter_path else "base",
            "config_hash": config_hash(config),
            "tuned_adapter_path": str(Path(tuned_adapter_path).expanduser().resolve()) if tuned_adapter_path else None,
            "training_config_hash": training_cfg_hash,
        },
    }


def clear_prediction_cache(config: ASRConfig, *, tuned_or_base: str | None = None) -> None:
    """Remove milestone-3 prediction caches for deterministic notebook checks."""
    predictions_dir = Path(config.output_dir) / "predictions"
    targets: list[Path]
    if tuned_or_base in {"base", "tuned"}:
        targets = [predictions_dir / tuned_or_base]
    elif tuned_or_base is None:
        targets = [predictions_dir / "base", predictions_dir / "tuned"]
    else:
        raise ValueError("tuned_or_base must be one of: None, 'base', 'tuned'.")

    for target in targets:
        if target.exists():
            for child in target.iterdir():
                if child.is_file() and child.suffix == ".jsonl":
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
