"""High-level notebook APIs for the Whisper Large run."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from Runs.utils import (
    create_smoke_asr_dataset,
    evaluate_prediction_records,
    load_prediction_records,
    upsert_model_stage_result,
)

from .config import WhisperLargeRunConfig
from .inference import generate_predictions
from .modeling import WhisperBundle
from .training import train_with_lora


def create_smoke_dataset_for_config(config: WhisperLargeRunConfig) -> WhisperLargeRunConfig:
    """Create smoke manifests and attach them to the config."""

    print("[stage] Creating smoke dataset and attaching manifests to config.")
    manifests = create_smoke_asr_dataset(config.output_dir / "smoke_data")
    config.train_manifest = manifests["train"]
    config.validation_manifest = manifests["validation"]
    config.test_manifest = manifests["test"]
    return config


def baseline_predictions_path(config: WhisperLargeRunConfig) -> Path:
    """Return the canonical baseline predictions path for this run."""

    return config.output_dir / "predictions" / "baseline_test_predictions.jsonl"


def tuned_predictions_path(config: WhisperLargeRunConfig) -> Path:
    """Return the canonical tuned-model predictions path for this run."""

    return config.output_dir / "predictions" / "tuned_test_predictions.jsonl"


def run_baseline_predictions_once(
    config: WhisperLargeRunConfig,
    bundle: WhisperBundle | None = None,
) -> dict[str, Any]:
    """Generate baseline predictions once, then reuse the saved file."""

    if config.test_manifest is None:
        raise ValueError("test_manifest must be set before running baseline predictions.")
    predictions_path = baseline_predictions_path(config)
    was_cached = predictions_path.exists()
    if was_cached:
        print(f"[baseline] Using cached baseline predictions: {predictions_path}")
    else:
        print("[baseline] Cached predictions not found. Generating baseline predictions now.")
        generate_predictions(config, config.test_manifest, predictions_path, bundle=bundle)
    return {"predictions_path": predictions_path, "cached": was_cached}


def run_baseline_once(config: WhisperLargeRunConfig) -> dict[str, Any]:
    """Run baseline predictions once and reuse the saved file on later notebook runs."""

    baseline = run_baseline_predictions_once(config)
    metrics = evaluate_saved_predictions(config, baseline["predictions_path"], stage="baseline")
    return {**baseline, "metrics": metrics}


def train_lora_with_early_stopping(config: WhisperLargeRunConfig) -> dict[str, Any]:
    """Train or resume LoRA fine-tuning with early stopping and checkpointing."""

    return train_with_lora(config)


def run_tuned_predictions(config: WhisperLargeRunConfig, checkpoint_path: str | Path | None = None) -> dict[str, Any]:
    """Generate tuned-model predictions on the test set and save them locally."""

    if config.test_manifest is None:
        raise ValueError("test_manifest must be set before running tuned predictions.")
    predictions_path = tuned_predictions_path(config)
    print(f"[tuned] Generating tuned predictions with checkpoint: {checkpoint_path}")
    if config.smoke_mode:
        generate_predictions(config, config.test_manifest, predictions_path)
    else:
        generate_predictions(config, config.test_manifest, predictions_path, adapter_path=checkpoint_path)
    return {"predictions_path": predictions_path, "checkpoint_path": checkpoint_path}


def evaluate_saved_predictions(
    config: WhisperLargeRunConfig,
    predictions_path: str | Path,
    stage: str,
) -> dict[str, Any]:
    """Evaluate saved predictions, persist metrics, and return the metrics payload."""

    print(f"[eval] Evaluating {stage} predictions: {predictions_path}")
    records = load_prediction_records(predictions_path)
    metrics = evaluate_prediction_records(records)
    metrics_path = config.output_dir / "metrics" / f"{stage}_metrics.json"
    with metrics_path.open("w", encoding="utf-8") as handle:
        json.dump(metrics, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"[eval] {stage} metrics: WER={metrics['wer']:.6f}, CER={metrics['cer']:.6f}, RTF={metrics['rtf']}")
    print(f"[eval] Wrote metrics: {metrics_path}")
    return metrics


def write_stage_result(
    config: WhisperLargeRunConfig,
    stage: str,
    metrics: dict[str, Any],
    predictions_path: str | Path | None = None,
    metadata: dict[str, Any] | None = None,
) -> Path:
    """Write one stage result into the shared model results JSON file."""

    print(f"[results] Writing {stage} result into: {config.results_path}")
    path = upsert_model_stage_result(
        results_path=config.results_path,
        model_key=config.model_key,
        stage=stage,
        metrics=metrics,
        predictions_path=predictions_path,
        metadata={"config": config.to_dict(), **(metadata or {})},
    )
    print(f"[results] Updated shared results file: {path}")
    return path


def run_full_workflow(config: WhisperLargeRunConfig) -> dict[str, Any]:
    """Run baseline, train LoRA, run tuned predictions, evaluate, and save results."""

    config.resolved()
    if config.smoke_mode:
        create_smoke_dataset_for_config(config)
    baseline = run_baseline_once(config)
    write_stage_result(
        config,
        stage="baseline",
        metrics=baseline["metrics"],
        predictions_path=baseline["predictions_path"],
        metadata={"cached": baseline["cached"]},
    )
    training = train_lora_with_early_stopping(config)
    tuned = run_tuned_predictions(config, checkpoint_path=training.get("best_checkpoint"))
    tuned_metrics = evaluate_saved_predictions(config, tuned["predictions_path"], stage="tuned")
    write_stage_result(
        config,
        stage="tuned",
        metrics=tuned_metrics,
        predictions_path=tuned["predictions_path"],
        metadata={"training": training},
    )
    return {
        "baseline": baseline,
        "training": training,
        "tuned": {**tuned, "metrics": tuned_metrics},
        "results_path": config.results_path,
    }
