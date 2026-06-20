"""LoRA training helpers with checkpointing and early stopping."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from Runs.utils.metrics import evaluate_prediction_records

from .config import WhisperLargeRunConfig
from .dataset import DataCollatorSpeechSeq2SeqWithPadding, build_hf_datasets
from .modeling import attach_lora, load_whisper_bundle


def train_with_lora(config: WhisperLargeRunConfig) -> dict[str, Any]:
    """Train Whisper Large LoRA adapters with epoch evaluation and checkpointing."""

    print("[training] Starting LoRA training stage.")
    if config.smoke_mode:
        return _simulate_training(config)

    import torch
    from transformers import EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments

    print("[training] Loading base model and processor.")
    bundle = load_whisper_bundle(config)
    print("[training] Attaching LoRA adapters.")
    model = attach_lora(config, bundle.model)
    print("[training] Building Hugging Face datasets.")
    train_dataset, eval_dataset = build_hf_datasets(config, bundle.processor)
    print(f"[training] Train samples: {len(train_dataset)}")
    print(f"[training] End-of-epoch eval samples: {len(eval_dataset)}")
    data_collator = DataCollatorSpeechSeq2SeqWithPadding(bundle.processor)

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        pred_ids = eval_pred.predictions
        label_ids = eval_pred.label_ids
        label_ids[label_ids == -100] = bundle.processor.tokenizer.pad_token_id
        predictions = bundle.processor.batch_decode(pred_ids, skip_special_tokens=True)
        references = bundle.processor.batch_decode(label_ids, skip_special_tokens=True)
        records = [
            {"reference": reference, "prediction": prediction}
            for reference, prediction in zip(references, predictions)
        ]
        metrics = evaluate_prediction_records(records)
        return {"wer": metrics["wer"], "cer": metrics["cer"]}

    args = Seq2SeqTrainingArguments(
        output_dir=str(config.training_output_dir),
        per_device_train_batch_size=config.per_device_train_batch_size,
        per_device_eval_batch_size=config.per_device_eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        predict_with_generate=True,
        generation_max_length=config.generation_max_new_tokens,
        eval_strategy="epoch",
        save_strategy="epoch",
        logging_strategy="epoch",
        load_best_model_at_end=True,
        metric_for_best_model="cer",
        greater_is_better=False,
        save_total_limit=config.save_total_limit,
        optim=config.optim,
        fp16=torch.cuda.is_available(),
        report_to=["tensorboard"],
    )
    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=bundle.processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=config.patience)],
    )
    print("[training] Trainer configured.")
    print(f"[training] Checkpoints directory: {config.training_output_dir}")
    print(f"[training] Resume checkpoint: {config.resume_from_checkpoint}")
    print("[training] Training now. Hugging Face Trainer will log epoch progress below.")
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint)
    print("[training] Training complete. Saving best model.")
    trainer.save_model(str(config.training_output_dir / "best"))
    summary = {
        "backend": "transformers",
        "best_checkpoint": str(trainer.state.best_model_checkpoint or config.training_output_dir / "best"),
        "best_metric": trainer.state.best_metric,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    _write_training_summary(config, summary)
    print(f"[training] Best checkpoint: {summary['best_checkpoint']}")
    print(f"[training] Best metric: {summary['best_metric']}")
    return summary


def _simulate_training(config: WhisperLargeRunConfig) -> dict[str, Any]:
    """Create deterministic smoke checkpoints without running a real trainer."""

    print("[training] Smoke mode enabled. Creating placeholder checkpoint files.")
    checkpoint_dir = config.training_output_dir / "checkpoint-epoch-1"
    best_dir = config.training_output_dir / "best"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    best_dir.mkdir(parents=True, exist_ok=True)
    marker = {
        "smoke_mode": True,
        "message": "Smoke checkpoint placeholder; set smoke_mode=False for real LoRA training.",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }
    for path in (checkpoint_dir / "adapter_config.json", best_dir / "adapter_config.json"):
        with path.open("w", encoding="utf-8") as handle:
            json.dump(marker, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
    summary = {
        "backend": "smoke",
        "best_checkpoint": str(best_dir),
        "best_metric": 0.0,
        "completed_at": marker["completed_at"],
    }
    _write_training_summary(config, summary)
    print(f"[training] Smoke best checkpoint: {best_dir}")
    return summary


def _write_training_summary(config: WhisperLargeRunConfig, summary: dict[str, Any]) -> None:
    """Persist training metadata for resume and audit."""

    path = config.output_dir / "training_summary.json"
    with path.open("w", encoding="utf-8") as handle:
        json.dump(summary, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(f"[training] Wrote training summary: {path}")
