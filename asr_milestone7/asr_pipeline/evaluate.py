"""High-level ASR prediction evaluation, metric caching, and comparison.

Milestone 6 evaluates base or tuned prediction JSONL files produced by
``asr_pipeline.predict``. It computes WER/CER variants with conservative
normalization, caches metrics under ``outputs/metrics/{base,tuned}``, and writes
base-vs-tuned comparison reports under ``outputs/reports``.
"""

from __future__ import annotations

import hashlib
import shutil
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

from asr_pipeline.config import ASRConfig
from asr_pipeline.normalization import basic_normalize, loose_normalize
from asr_pipeline.utils.hashing import config_hash, stable_hash
from asr_pipeline.utils.io import ensure_dir, read_json, read_jsonl, write_json
from asr_pipeline.utils.logging import get_logger

LOGGER = get_logger(__name__)

METRIC_SCHEMA_VERSION = 2
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

COMPARISON_METRIC_NAMES = [
    "wer",
    "cer",
    "normalized_wer",
    "normalized_cer",
    "loose_wer",
    "loose_cer",
]


def _safe_name(value: str) -> str:
    """Return a filesystem-safe identifier while keeping names readable."""
    return (
        value.replace("/", "__")
        .replace(" ", "_")
        .replace(":", "_")
        .replace("\\", "__")
    )


def _file_sha256(path: str | Path) -> str:
    """Hash a prediction file so metric cache invalidates if predictions change."""
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _edit_distance(reference: Sequence[Any], hypothesis: Sequence[Any]) -> int:
    """Compute Levenshtein edit distance using two rolling rows."""
    if reference == hypothesis:
        return 0
    if not reference:
        return len(hypothesis)
    if not hypothesis:
        return len(reference)

    previous = list(range(len(hypothesis) + 1))
    for i, ref_item in enumerate(reference, start=1):
        current = [i]
        for j, hyp_item in enumerate(hypothesis, start=1):
            substitution_cost = 0 if ref_item == hyp_item else 1
            current.append(
                min(
                    previous[j] + 1,          # deletion
                    current[j - 1] + 1,       # insertion
                    previous[j - 1] + substitution_cost,
                )
            )
        previous = current
    return previous[-1]


def _word_units(text: str) -> list[str]:
    """Tokenize text into whitespace-separated word units."""
    text = str(text).strip()
    return text.split() if text else []


def _char_units(text: str) -> list[str]:
    """Tokenize text into character units after removing spaces."""
    return [ch for ch in str(text).replace(" ", "")]


def _safe_rate(errors: int, total_reference_units: int, total_hypothesis_units: int) -> float:
    """Return an error rate with a defined behavior for empty references."""
    if total_reference_units == 0:
        return 0.0 if total_hypothesis_units == 0 else 1.0
    return errors / total_reference_units


def _error_rate(pairs: Iterable[tuple[str, str]], *, unit: str) -> dict[str, Any]:
    """Compute aggregate edit-distance error rate over all pairs."""
    if unit == "word":
        unit_fn = _word_units
    elif unit == "char":
        unit_fn = _char_units
    else:
        raise ValueError("unit must be 'word' or 'char'.")

    errors = 0
    total_reference_units = 0
    total_hypothesis_units = 0
    for reference, prediction in pairs:
        ref_units = unit_fn(reference)
        hyp_units = unit_fn(prediction)
        errors += _edit_distance(ref_units, hyp_units)
        total_reference_units += len(ref_units)
        total_hypothesis_units += len(hyp_units)

    return {
        "value": _safe_rate(errors, total_reference_units, total_hypothesis_units),
        "errors": errors,
        "reference_units": total_reference_units,
        "hypothesis_units": total_hypothesis_units,
    }


def _validate_prediction_rows(rows: Sequence[Mapping[str, Any]], prediction_path: str | Path) -> None:
    """Validate the base/tuned prediction schema."""
    if not rows:
        raise ValueError(f"Prediction file is empty: {prediction_path}")

    for idx, row in enumerate(rows):
        state = str(row.get("tuned_or_base"))
        required = TUNED_PREDICTION_REQUIRED_COLUMNS if state == "tuned" else PREDICTION_REQUIRED_COLUMNS
        missing = sorted(required - set(row))
        if missing:
            raise ValueError(
                f"Prediction row {idx} is missing required column(s): {', '.join(missing)}"
            )
        if state not in {"base", "tuned"}:
            raise ValueError(
                f"Prediction row {idx} has invalid tuned_or_base={row.get('tuned_or_base')!r}"
            )

    stable_fields = ["model_name", "model_family", "tuned_or_base", "run_name", "config_hash"]
    first = rows[0]
    for field in stable_fields:
        values = {str(row[field]) for row in rows}
        if len(values) != 1:
            raise ValueError(f"Prediction file mixes multiple values for {field!r}: {sorted(values)}")
        if str(first[field]) == "":
            raise ValueError(f"Prediction metadata field {field!r} cannot be empty.")


def _prediction_metadata(rows: Sequence[Mapping[str, Any]]) -> dict[str, str]:
    first = rows[0]
    meta = {
        "model_name": str(first["model_name"]),
        "model_family": str(first["model_family"]),
        "tuned_or_base": str(first["tuned_or_base"]),
        "run_name": str(first["run_name"]),
        "prediction_config_hash": str(first["config_hash"]),
    }
    if meta["tuned_or_base"] == "tuned":
        meta["tuned_adapter_path"] = str(first.get("tuned_adapter_path", ""))
        meta["training_config_hash"] = str(first.get("training_config_hash", ""))
    return meta


def metric_cache_key(config: ASRConfig, prediction_path: str | Path, prediction_sha256: str) -> str:
    """Create a cache key from config identity and prediction file content."""
    path = Path(prediction_path)
    payload = {
        "schema_version": METRIC_SCHEMA_VERSION,
        "prediction_path": str(path.resolve()),
        "prediction_sha256": prediction_sha256,
        "config_hash": config_hash(config),
    }
    return stable_hash(payload, length=16)


def metric_output_path(config: ASRConfig, rows: Sequence[Mapping[str, Any]], key: str) -> Path:
    """Return the metrics JSON path for base or tuned predictions."""
    meta = _prediction_metadata(rows)
    safe_model = _safe_name(meta["model_name"])
    safe_run = _safe_name(meta["run_name"])
    state = meta["tuned_or_base"]
    filename = f"{meta['model_family']}__{safe_model}__{safe_run}__{state}__{key}.json"
    return Path(config.output_dir) / "metrics" / state / filename


def compute_metrics(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    """Compute raw, normalized, and loose ASR metrics for prediction rows."""
    raw_pairs = [(str(row["reference"]), str(row["prediction"])) for row in rows]
    normalized_pairs = [(basic_normalize(ref), basic_normalize(pred)) for ref, pred in raw_pairs]
    loose_pairs = [(loose_normalize(ref), loose_normalize(pred)) for ref, pred in raw_pairs]

    wer = _error_rate(raw_pairs, unit="word")
    cer = _error_rate(raw_pairs, unit="char")
    normalized_wer = _error_rate(normalized_pairs, unit="word")
    normalized_cer = _error_rate(normalized_pairs, unit="char")
    loose_wer = _error_rate(loose_pairs, unit="word")
    loose_cer = _error_rate(loose_pairs, unit="char")

    return {
        "wer": wer["value"],
        "cer": cer["value"],
        "normalized_wer": normalized_wer["value"],
        "normalized_cer": normalized_cer["value"],
        "loose_wer": loose_wer["value"],
        "loose_cer": loose_cer["value"],
        "details": {
            "raw": {"wer": wer, "cer": cer},
            "normalized": {"wer": normalized_wer, "cer": normalized_cer},
            "loose": {"wer": loose_wer, "cer": loose_cer},
        },
    }


def evaluate_predictions(config: ASRConfig, prediction_path: str | Path) -> dict[str, Any]:
    """Evaluate a base or tuned prediction JSONL file and cache metrics.

    Returns a dictionary containing ``metrics`` and ``metadata``. If the matching
    metrics JSON already exists, it is loaded and recomputation is skipped.
    """
    prediction_path = Path(prediction_path)
    if not prediction_path.exists():
        raise FileNotFoundError(f"Prediction JSONL does not exist: {prediction_path}")

    rows = read_jsonl(prediction_path)
    _validate_prediction_rows(rows, prediction_path)
    prediction_sha256 = _file_sha256(prediction_path)
    key = metric_cache_key(config, prediction_path, prediction_sha256)
    metrics_path = metric_output_path(config, rows, key)

    if metrics_path.exists():
        cached_payload = read_json(metrics_path)
        LOGGER.info("Loaded metric cache: %s", metrics_path)
        return {
            "metrics": cached_payload["metrics"],
            "metadata": {
                **cached_payload.get("metadata", {}),
                "computed_new": False,
                "loaded_from_cache": True,
                "metrics_path": str(metrics_path),
                "cache_key": key,
            },
        }

    meta = _prediction_metadata(rows)
    metrics = compute_metrics(rows)
    metadata = {
        "computed_new": True,
        "loaded_from_cache": False,
        "metrics_path": str(metrics_path),
        "cache_key": key,
        "prediction_path": str(prediction_path),
        "prediction_sha256": prediction_sha256,
        "config_hash": config_hash(config),
        "row_count": len(rows),
        **meta,
    }
    payload = {
        "schema_version": METRIC_SCHEMA_VERSION,
        "metrics": metrics,
        "metadata": metadata,
        "normalization": {
            "basic": [
                "trim whitespace",
                "collapse repeated whitespace",
                "remove common Arabic/Latin punctuation",
            ],
            "loose": [
                "remove all Unicode punctuation",
                "remove Arabic tatweel",
                "remove Arabic diacritics",
                "preserve Arabic letter identities and dialect words",
            ],
            "explicitly_not_done": [
                "no dialect-to-MSA conversion",
                "no global أ/إ/آ to ا conversion",
                "no global ى to ي conversion",
                "no global ه to ة conversion",
            ],
        },
    }
    ensure_dir(metrics_path.parent)
    write_json(metrics_path, payload)
    LOGGER.info("Created metric cache: %s", metrics_path)

    return {"metrics": metrics, "metadata": metadata}


def _load_metric_payload(path: str | Path) -> dict[str, Any]:
    """Load a metrics JSON payload produced by evaluate_predictions."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Metrics JSON does not exist: {path}")
    payload = read_json(path)
    if "metrics" not in payload or "metadata" not in payload:
        raise ValueError(f"Metrics file does not look like an evaluate_predictions output: {path}")
    return payload


def _relative_improvement(base_value: float, tuned_value: float) -> float | None:
    """Return relative improvement where lower error rate is better."""
    if base_value == 0:
        return None
    return (base_value - tuned_value) / base_value


def _comparison_output_paths(base_metrics_path: Path, tuned_metrics_path: Path) -> tuple[Path, Path, str]:
    """Create deterministic report paths under outputs/reports."""
    try:
        output_dir = base_metrics_path.resolve().parents[2]
    except IndexError:
        output_dir = Path("outputs").resolve()
    reports_dir = ensure_dir(output_dir / "reports")
    key = stable_hash(
        {
            "schema_version": METRIC_SCHEMA_VERSION,
            "base_metrics_path": str(base_metrics_path.resolve()),
            "tuned_metrics_path": str(tuned_metrics_path.resolve()),
            "base_mtime_ns": base_metrics_path.stat().st_mtime_ns,
            "tuned_mtime_ns": tuned_metrics_path.stat().st_mtime_ns,
        },
        length=16,
    )
    return (
        reports_dir / f"base_vs_tuned__{key}.json",
        reports_dir / f"base_vs_tuned__{key}.md",
        key,
    )


def _render_comparison_markdown(comparison: Mapping[str, Any]) -> str:
    """Render a compact Markdown comparison report."""
    meta = comparison["metadata"]
    lines = [
        "# Base vs tuned ASR comparison",
        "",
        f"- Model: `{meta.get('model_name', '')}`",
        f"- Family: `{meta.get('model_family', '')}`",
        f"- Run: `{meta.get('run_name', '')}`",
        f"- Base metrics: `{meta.get('base_metrics_path', '')}`",
        f"- Tuned metrics: `{meta.get('tuned_metrics_path', '')}`",
        "",
        "| metric | base | tuned | absolute improvement | relative improvement |",
        "|---|---:|---:|---:|---:|",
    ]
    for metric, row in comparison["metrics"].items():
        rel = row["relative_improvement"]
        rel_text = "n/a" if rel is None else f"{rel:.6f}"
        lines.append(
            f"| {metric} | {row['base']:.6f} | {row['tuned']:.6f} | "
            f"{row['absolute_improvement']:.6f} | {rel_text} |"
        )
    lines.extend(
        [
            "",
            "Positive improvement means the tuned error rate is lower than the base error rate.",
            "",
        ]
    )
    return "\n".join(lines)


def compare_base_vs_tuned(base_metrics_path: str | Path, tuned_metrics_path: str | Path) -> dict[str, Any]:
    """Compare base and tuned metric JSON files and save JSON/Markdown reports.

    Improvement is computed as ``base - tuned`` because lower WER/CER is better.
    Relative improvement is ``(base - tuned) / base`` and is ``None`` when the
    base metric is zero.
    """
    base_metrics_path = Path(base_metrics_path)
    tuned_metrics_path = Path(tuned_metrics_path)
    base_payload = _load_metric_payload(base_metrics_path)
    tuned_payload = _load_metric_payload(tuned_metrics_path)

    base_meta = base_payload["metadata"]
    tuned_meta = tuned_payload["metadata"]
    if base_meta.get("tuned_or_base") != "base":
        raise ValueError(f"Expected base metrics file, got: {base_meta.get('tuned_or_base')!r}")
    if tuned_meta.get("tuned_or_base") != "tuned":
        raise ValueError(f"Expected tuned metrics file, got: {tuned_meta.get('tuned_or_base')!r}")

    comparisons: dict[str, dict[str, Any]] = {}
    for metric_name in COMPARISON_METRIC_NAMES:
        base_value = float(base_payload["metrics"][metric_name])
        tuned_value = float(tuned_payload["metrics"][metric_name])
        comparisons[metric_name] = {
            "base": base_value,
            "tuned": tuned_value,
            "absolute_improvement": base_value - tuned_value,
            "relative_improvement": _relative_improvement(base_value, tuned_value),
        }

    json_path, md_path, key = _comparison_output_paths(base_metrics_path, tuned_metrics_path)
    comparison = {
        "schema_version": METRIC_SCHEMA_VERSION,
        "metadata": {
            "comparison_key": key,
            "base_metrics_path": str(base_metrics_path),
            "tuned_metrics_path": str(tuned_metrics_path),
            "comparison_json_path": str(json_path),
            "comparison_markdown_path": str(md_path),
            "model_name": tuned_meta.get("model_name", base_meta.get("model_name")),
            "model_family": tuned_meta.get("model_family", base_meta.get("model_family")),
            "run_name": tuned_meta.get("run_name", base_meta.get("run_name")),
            "base_prediction_path": base_meta.get("prediction_path"),
            "tuned_prediction_path": tuned_meta.get("prediction_path"),
            "tuned_adapter_path": tuned_meta.get("tuned_adapter_path"),
            "training_config_hash": tuned_meta.get("training_config_hash"),
            "improvement_definition": "base_metric - tuned_metric; positive means tuned is better",
        },
        "metrics": comparisons,
    }
    write_json(json_path, comparison)
    md_path.write_text(_render_comparison_markdown(comparison), encoding="utf-8")
    LOGGER.info("Saved comparison reports: %s and %s", json_path, md_path)
    return {
        "comparison": comparison,
        "metadata": {
            "comparison_json_path": str(json_path),
            "comparison_markdown_path": str(md_path),
            "comparison_key": key,
        },
    }


def clear_metric_cache(config: ASRConfig, *, tuned_or_base: str | None = None) -> None:
    """Remove metric caches for deterministic notebook checks."""
    metrics_dir = Path(config.output_dir) / "metrics"
    targets: list[Path]
    if tuned_or_base in {"base", "tuned"}:
        targets = [metrics_dir / tuned_or_base]
    elif tuned_or_base is None:
        targets = [metrics_dir / "base", metrics_dir / "tuned"]
    else:
        raise ValueError("tuned_or_base must be one of: None, 'base', 'tuned'.")

    for target in targets:
        if target.exists():
            for child in target.iterdir():
                if child.is_file() and child.suffix == ".json":
                    child.unlink()
                elif child.is_dir():
                    shutil.rmtree(child)
