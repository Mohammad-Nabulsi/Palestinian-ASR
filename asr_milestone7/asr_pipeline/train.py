"""High-level training API with smoke-safe LoRA training scaffolding.

Milestone 5 adds the public ``train(config, adapter, prepared_data, collator)``
entry point.  The notebook calls this API only; architecture-specific behavior
is dispatched through adapter ``train`` methods.

Real model fine-tuning is intentionally deferred.  In ``smoke_mode=True`` the
adapters use a deterministic mock LoRA loop that exercises the full pipeline:
model-loading hook, LoRA backend selection, collator call, mock train step,
validation metric computation, checkpoint writing, best-checkpoint selection,
and W&B/no-op logging.
"""

from __future__ import annotations

import importlib.util
import shutil
from pathlib import Path
from typing import Any, Mapping, Sequence

from asr_pipeline.config import ASRConfig
from asr_pipeline.evaluate import compute_metrics
from asr_pipeline.registry import get_model_family
from asr_pipeline.utils.hashing import config_hash, stable_hash
from asr_pipeline.utils.io import ensure_dir, read_json, write_json
from asr_pipeline.utils.logging import get_logger
from asr_pipeline.utils.wandb import WandbRun

LOGGER = get_logger(__name__)

TRAINING_SCHEMA_VERSION = 1
DEFAULT_LORA_CONFIG: dict[str, Any] = {
    "r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "bias": "none",
    "task_type": "ASR_PLACEHOLDER",
}
FAMILY_TARGET_MODULES: dict[str, list[str]] = {
    "whisper": ["q_proj", "v_proj", "k_proj", "out_proj"],
    "qwen": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj"],
    "omni": ["q_proj", "k_proj", "v_proj", "out_proj"],
}


def _safe_name(value: str) -> str:
    """Return a filesystem-safe identifier while keeping names readable."""
    return value.replace("/", "__").replace(" ", "_").replace(":", "_").replace("\\", "__")


def _rows_from_prepared_data(prepared_data: Any, split: str) -> list[dict[str, Any]]:
    """Accept the milestone-2 prepare result or a raw prepared split mapping."""
    if isinstance(prepared_data, Mapping) and "prepared" in prepared_data:
        prepared = prepared_data["prepared"]
    else:
        prepared = prepared_data

    if not isinstance(prepared, Mapping):
        raise TypeError("prepared_data must be a prepare_data_and_collator() result or split mapping.")
    if split not in prepared:
        raise ValueError(f"Prepared data does not contain split {split!r}.")

    rows = prepared[split]
    if not isinstance(rows, Sequence) or isinstance(rows, (str, bytes)):
        raise TypeError(f"Prepared split {split!r} must be a sequence of row dictionaries.")
    return [dict(row) for row in rows]


def _prepared_training_identity(prepared_data: Any) -> dict[str, Any]:
    """Create a deterministic identity for train/val rows used by training."""
    identity: dict[str, Any] = {}
    for split in ("train", "val"):
        rows = _rows_from_prepared_data(prepared_data, split)
        identity[split] = {
            "row_count": len(rows),
            "uids": [row.get("uid") for row in rows],
            "references": [row.get("reference", row.get("text", row.get("target_text", ""))) for row in rows],
        }
    return identity


def training_run_key(config: ASRConfig, *, family: str, prepared_data: Any) -> str:
    """Stable key for a training run's checkpoint directory."""
    payload = {
        "schema_version": TRAINING_SCHEMA_VERSION,
        "model_family": family,
        "model_name": config.model_name,
        "run_name": config.run_name,
        "config_hash": config_hash(config),
        "training_identity": _prepared_training_identity(prepared_data),
        "training_settings": {
            "learning_rate": config.learning_rate,
            "max_epochs": config.max_epochs,
            "early_stopping_patience": config.early_stopping_patience,
            "smoke_mode": config.smoke_mode,
        },
    }
    return stable_hash(payload, length=16)


def training_output_dir(config: ASRConfig, *, family: str, key: str) -> Path:
    """Return the checkpoint root for this model/run/training identity."""
    safe_model = _safe_name(config.model_name)
    safe_run = _safe_name(config.run_name)
    return Path(config.output_dir) / "checkpoints" / f"{family}__{safe_model}__{safe_run}__{key}"


def lora_config_for_family(family: str) -> dict[str, Any]:
    """Return the placeholder LoRA config for an architecture family."""
    if family not in FAMILY_TARGET_MODULES:
        raise ValueError(f"Unsupported model family for LoRA config: {family!r}")
    cfg = dict(DEFAULT_LORA_CONFIG)
    cfg["target_modules"] = FAMILY_TARGET_MODULES[family]
    return cfg


def select_lora_backend(family: str, *, smoke_mode: bool) -> dict[str, Any]:
    """Choose Unsloth where supported, otherwise PEFT fallback.

    Qwen is the only family marked as Unsloth-capable in this scaffold.  Whisper
    and Omni use the PEFT fallback path.  In smoke mode, missing dependencies do
    not fail the run; the backend is logged as a mock-compatible PEFT/Unsloth
    path so the full pipeline remains testable without downloads.
    """
    peft_available = importlib.util.find_spec("peft") is not None
    unsloth_available = importlib.util.find_spec("unsloth") is not None

    if family == "qwen":
        if unsloth_available:
            return {
                "backend": "unsloth",
                "backend_mode": "available",
                "unsloth_supported": True,
                "unsloth_available": True,
                "peft_available": peft_available,
                "message": "Using Unsloth LoRA path for Qwen.",
            }
        return {
            "backend": "peft",
            "backend_mode": "mock" if smoke_mode and not peft_available else "available",
            "unsloth_supported": True,
            "unsloth_available": False,
            "peft_available": peft_available,
            "message": "Unsloth is supported for Qwen but unavailable here; using PEFT fallback/mock path.",
        }

    return {
        "backend": "peft",
        "backend_mode": "mock" if smoke_mode and not peft_available else "available",
        "unsloth_supported": False,
        "unsloth_available": unsloth_available,
        "peft_available": peft_available,
        "message": f"Unsloth is not used for {family}; using PEFT fallback/mock path.",
    }


def _reference_text(row: Mapping[str, Any]) -> str:
    return str(row.get("reference", row.get("target_text", row.get("labels_text", row.get("text", "")))))


def _make_validation_prediction(reference: str, *, family: str, epoch: int, smoke_epochs: int) -> str:
    """Deterministic validation prediction used by the smoke training path."""
    if epoch >= smoke_epochs:
        return reference
    return f"[placeholder train {family}]"


def _mock_forward_backward_step(
    *,
    family: str,
    epoch: int,
    train_rows: Sequence[Mapping[str, Any]],
    collator: Any,
    learning_rate: float,
    state: dict[str, float],
) -> tuple[float, dict[str, Any]]:
    """A tiny dependency-free stand-in for a forward/backward/optimizer step."""
    batch = collator(list(train_rows)) if train_rows else collator([])
    batch_size = int(batch.get("batch_size", len(train_rows)))
    text_units = sum(len(_reference_text(row).split()) for row in train_rows) or 1

    pseudo_forward_value = state["weight"] * (batch_size + text_units + epoch)
    pseudo_gradient = pseudo_forward_value / (10.0 + epoch)
    state["weight"] -= learning_rate * pseudo_gradient
    train_loss = max(0.001, (1.0 / (epoch + 1)) + (0.01 * len(family)))
    step_info = {
        "mock_forward_value": pseudo_forward_value,
        "mock_gradient": pseudo_gradient,
        "updated_weight": state["weight"],
        "batch_size": batch_size,
        "collator_family": batch.get("family"),
        "train_step_kind": "mock_forward_backward_optimizer_step",
    }
    return train_loss, step_info


def _write_epoch_checkpoint(
    *,
    checkpoint_dir: Path,
    epoch: int,
    family: str,
    config: ASRConfig,
    lora_backend: Mapping[str, Any],
    lora_config: Mapping[str, Any],
    metrics: Mapping[str, Any],
) -> Path:
    """Write a minimal adapter/model checkpoint for one epoch."""
    epoch_dir = ensure_dir(checkpoint_dir / f"checkpoint-epoch-{epoch:04d}")
    adapter_dir = ensure_dir(epoch_dir / "adapter_model")
    write_json(
        adapter_dir / "adapter_config.json",
        {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "model_name": config.model_name,
            "model_family": family,
            "smoke_checkpoint": True,
            "lora_backend": dict(lora_backend),
            "lora_config": dict(lora_config),
        },
    )
    write_json(
        adapter_dir / "adapter_model_placeholder.json",
        {
            "note": "Smoke-mode placeholder adapter weights. Real LoRA weights are added in a later milestone.",
            "epoch": epoch,
            "model_name": config.model_name,
            "model_family": family,
            "metrics": dict(metrics),
        },
    )
    write_json(epoch_dir / "epoch_metrics.json", dict(metrics))
    return epoch_dir


def run_smoke_lora_training(
    *,
    config: ASRConfig,
    family: str,
    prepared_data: Any,
    collator: Any,
    checkpoint_dir: str | Path,
) -> dict[str, Any]:
    """Run deterministic smoke LoRA training and write epoch checkpoints."""
    checkpoint_dir = ensure_dir(checkpoint_dir)
    train_rows = _rows_from_prepared_data(prepared_data, "train")
    val_rows = _rows_from_prepared_data(prepared_data, "val")
    if not train_rows:
        raise ValueError("Training split is empty after preparation.")
    if not val_rows:
        raise ValueError("Validation split is empty after preparation.")

    # Exercise the adapter model-loading path without downloading anything.
    smoke_model = {
        "family": family,
        "model_name": config.model_name,
        "smoke_model": True,
        "local_model_cache_dir": config.local_model_cache_dir,
    }

    lora_backend = select_lora_backend(family, smoke_mode=config.smoke_mode)
    lora_config = lora_config_for_family(family)
    LOGGER.info("%s LoRA backend: %s", family, lora_backend["message"])

    max_epochs = int(config.max_epochs)
    if max_epochs <= 0:
        raise ValueError("max_epochs must be positive.")
    smoke_epochs = max(1, min(2, max_epochs))
    patience = max(1, int(config.early_stopping_patience))
    learning_rate = float(config.learning_rate)

    history: list[dict[str, Any]] = []
    checkpoints: list[str] = []
    best_epoch: int | None = None
    best_wer = float("inf")
    best_checkpoint_path: str | None = None
    epochs_without_improvement = 0
    state = {"weight": 1.0}

    for epoch in range(1, smoke_epochs + 1):
        train_loss, step_info = _mock_forward_backward_step(
            family=family,
            epoch=epoch,
            train_rows=train_rows,
            collator=collator,
            learning_rate=learning_rate,
            state=state,
        )
        val_loss = max(0.001, train_loss * 0.9)

        val_metric_rows = [
            {
                "uid": str(row.get("uid", idx)),
                "reference": _reference_text(row),
                "prediction": _make_validation_prediction(
                    _reference_text(row), family=family, epoch=epoch, smoke_epochs=smoke_epochs
                ),
            }
            for idx, row in enumerate(val_rows)
        ]
        val_metrics = compute_metrics(val_metric_rows)
        epoch_metrics = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "wer": val_metrics["wer"],
            "cer": val_metrics["cer"],
            "normalized_wer": val_metrics["normalized_wer"],
            "normalized_cer": val_metrics["normalized_cer"],
            "loose_wer": val_metrics["loose_wer"],
            "loose_cer": val_metrics["loose_cer"],
            "step_info": step_info,
        }
        epoch_dir = _write_epoch_checkpoint(
            checkpoint_dir=checkpoint_dir,
            epoch=epoch,
            family=family,
            config=config,
            lora_backend=lora_backend,
            lora_config=lora_config,
            metrics=epoch_metrics,
        )
        checkpoints.append(str(epoch_dir))
        history.append(epoch_metrics)

        if epoch_metrics["wer"] < best_wer:
            best_wer = float(epoch_metrics["wer"])
            best_epoch = epoch
            best_checkpoint_path = str(epoch_dir)
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            if epochs_without_improvement >= patience:
                LOGGER.info("Smoke early stopping triggered at epoch %s", epoch)
                break

    if best_epoch is None or best_checkpoint_path is None:
        raise RuntimeError("Smoke training did not produce a best checkpoint.")

    return {
        "train_status": "success",
        "training_mode": "smoke_mock_lora_training",
        "used_mock_training": True,
        "model_loading": smoke_model,
        "lora_backend": lora_backend["backend"],
        "lora_backend_mode": lora_backend["backend_mode"],
        "lora_backend_info": lora_backend,
        "lora_config": lora_config,
        "history": history,
        "checkpoints": checkpoints,
        "best_epoch": best_epoch,
        "best_checkpoint_path": best_checkpoint_path,
        "best_wer": best_wer,
        "last_metrics": history[-1],
        "early_stopping": {
            "patience": patience,
            "selection_metric": "val_wer",
            "selection_rule": "lowest validation WER",
            "triggered": len(history) < smoke_epochs,
        },
    }


def _existing_artifact_paths(config: ASRConfig) -> dict[str, list[str]]:
    """Find existing prediction/metric artifacts from earlier milestones."""
    output_dir = Path(config.output_dir)
    artifacts = {"predictions": [], "metrics": []}
    for state in ("base", "tuned"):
        pred_dir = output_dir / "predictions" / state
        metric_dir = output_dir / "metrics" / state
        if pred_dir.exists():
            artifacts["predictions"].extend(str(path) for path in sorted(pred_dir.glob("*.jsonl")))
        if metric_dir.exists():
            artifacts["metrics"].extend(str(path) for path in sorted(metric_dir.glob("*.json")))
    return artifacts


def _write_training_artifacts(
    *,
    config: ASRConfig,
    family: str,
    checkpoint_dir: Path,
    run_result: Mapping[str, Any],
    key: str,
    wandb_summary: Mapping[str, Any],
) -> dict[str, str]:
    """Write training config, metrics, and best-checkpoint pointer files."""
    cfg_path = checkpoint_dir / "training_config.json"
    metrics_path = checkpoint_dir / "training_metrics.json"
    pointer_path = checkpoint_dir / "best_checkpoint_pointer.json"

    write_json(
        cfg_path,
        {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "config": config.to_dict(),
            "config_hash": config_hash(config),
            "model_family": family,
            "training_run_key": key,
        },
    )
    write_json(
        metrics_path,
        {
            "schema_version": TRAINING_SCHEMA_VERSION,
            "model_name": config.model_name,
            "model_family": family,
            "run_name": config.run_name,
            "config_hash": config_hash(config),
            "train_status": run_result.get("train_status"),
            "training_mode": run_result.get("training_mode"),
            "lora_backend": run_result.get("lora_backend"),
            "lora_backend_mode": run_result.get("lora_backend_mode"),
            "lora_backend_info": run_result.get("lora_backend_info"),
            "lora_config": run_result.get("lora_config"),
            "history": run_result.get("history", []),
            "best_epoch": run_result.get("best_epoch"),
            "best_checkpoint_path": run_result.get("best_checkpoint_path"),
            "best_wer": run_result.get("best_wer"),
            "last_metrics": run_result.get("last_metrics"),
            "early_stopping": run_result.get("early_stopping"),
            "wandb": dict(wandb_summary),
        },
    )
    write_json(
        pointer_path,
        {
            "selection_metric": "val_wer",
            "selection_rule": "lowest validation WER",
            "best_epoch": run_result.get("best_epoch"),
            "best_checkpoint_path": run_result.get("best_checkpoint_path"),
            "best_wer": run_result.get("best_wer"),
        },
    )
    return {
        "training_config_path": str(cfg_path),
        "training_metrics_path": str(metrics_path),
        "best_checkpoint_pointer_path": str(pointer_path),
    }


def train(config: ASRConfig, adapter: Any, prepared_data: Any, collator: Any) -> dict[str, Any]:
    """High-level training API.

    The notebook should call this function only.  It validates architecture
    dispatch, creates a checkpoint root, delegates model-family-specific work to
    ``adapter.train(...)``, logs to W&B/no-op W&B, and writes final artifacts.
    """
    family = getattr(adapter, "family", None) or get_model_family(config.model_name)
    resolved_family = get_model_family(config.model_name)
    if family != resolved_family:
        raise ValueError(f"Adapter family {family!r} does not match model_name family {resolved_family!r}.")

    key = training_run_key(config, family=family, prepared_data=prepared_data)
    checkpoint_dir = training_output_dir(config, family=family, key=key)
    if checkpoint_dir.exists():
        # Training is intentionally not cached in milestone 5; overwrite so the
        # smoke notebook verifies checkpoint writing every time.
        shutil.rmtree(checkpoint_dir)
    ensure_dir(checkpoint_dir)

    lora_preview = {
        "lora_config": lora_config_for_family(family),
        "lora_backend_plan": select_lora_backend(family, smoke_mode=config.smoke_mode),
    }
    artifacts = _existing_artifact_paths(config)
    wandb_run = WandbRun(
        mode=config.wandb_mode,
        project="plug_play_asr_pipeline",
        name=f"{config.run_name}_{family}",
        config={
            **config.to_dict(),
            "model_family": family,
            "config_hash": config_hash(config),
            "dataset_paths": {
                "train": config.train_path,
                "val": config.val_path,
                "test": config.test_path,
            },
            "hyperparameters": {
                "learning_rate": config.learning_rate,
                "max_epochs": config.max_epochs,
                "early_stopping_patience": config.early_stopping_patience,
            },
            **lora_preview,
        },
    )

    try:
        run_result = adapter.train(
            prepared_data=prepared_data,
            collator=collator,
            checkpoint_dir=checkpoint_dir,
            smoke_mode=config.smoke_mode,
            local_model_cache_dir=config.local_model_cache_dir,
        )
        for epoch_row in run_result.get("history", []):
            wandb_run.log(
                {
                    "model_name": config.model_name,
                    "model_family": family,
                    "config_hash": config_hash(config),
                    "lora_backend": run_result.get("lora_backend"),
                    "lora_backend_mode": run_result.get("lora_backend_mode"),
                    "epoch": epoch_row.get("epoch"),
                    "train_loss": epoch_row.get("train_loss"),
                    "validation_loss": epoch_row.get("val_loss"),
                    "wer": epoch_row.get("wer"),
                    "cer": epoch_row.get("cer"),
                    "best_epoch": run_result.get("best_epoch"),
                    "best_checkpoint_path": run_result.get("best_checkpoint_path"),
                }
            )
        for path in artifacts["predictions"]:
            wandb_run.log_artifact(path, artifact_type="prediction_jsonl")
        for path in artifacts["metrics"]:
            wandb_run.log_artifact(path, artifact_type="metrics_json")
        wandb_summary = wandb_run.summary()
        artifact_paths = _write_training_artifacts(
            config=config,
            family=family,
            checkpoint_dir=checkpoint_dir,
            run_result=run_result,
            key=key,
            wandb_summary=wandb_summary,
        )
        wandb_run.log_artifact(artifact_paths["training_metrics_path"], artifact_type="training_metrics")
        wandb_run.log_artifact(artifact_paths["best_checkpoint_pointer_path"], artifact_type="best_checkpoint_pointer")
        metadata = {
            "train_status": run_result.get("train_status", "success"),
            "model_name": config.model_name,
            "model_family": family,
            "run_name": config.run_name,
            "config_hash": config_hash(config),
            "training_run_key": key,
            "checkpoint_dir": str(checkpoint_dir),
            "best_checkpoint_path": run_result.get("best_checkpoint_path"),
            "best_epoch": run_result.get("best_epoch"),
            "lora_backend": run_result.get("lora_backend"),
            "lora_backend_mode": run_result.get("lora_backend_mode"),
            "training_mode": run_result.get("training_mode"),
            "used_mock_training": run_result.get("used_mock_training"),
            "wandb": wandb_summary,
            **artifact_paths,
        }
        return {"metadata": metadata, "training": run_result}
    except Exception as exc:
        error_path = checkpoint_dir / "training_error.json"
        write_json(
            error_path,
            {
                "train_status": "failed",
                "model_name": config.model_name,
                "model_family": family,
                "run_name": config.run_name,
                "error": repr(exc),
                "config_hash": config_hash(config),
            },
        )
        LOGGER.exception("Training failed for %s", config.model_name)
        raise
    finally:
        wandb_run.finish()


def clear_training_outputs(config: ASRConfig) -> None:
    """Remove milestone-5 checkpoint runs for deterministic notebook checks."""
    checkpoints_dir = Path(config.output_dir) / "checkpoints"
    if checkpoints_dir.exists():
        for child in checkpoints_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            elif child.name != ".gitkeep":
                child.unlink()
