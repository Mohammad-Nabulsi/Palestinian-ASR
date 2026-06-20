"""Dataset preparation utilities for Whisper fine-tuning."""

from __future__ import annotations

from typing import Any

from Runs.utils.data import resolve_manifest_records

from .config import WhisperLargeRunConfig


def build_hf_datasets(config: WhisperLargeRunConfig, processor: Any) -> tuple[Any, Any]:
    """Build train and epoch-eval datasets for Hugging Face Seq2SeqTrainer."""

    if config.train_manifest is None or config.test_manifest is None:
        raise ValueError("train_manifest and test_manifest must be configured for training.")

    from datasets import Audio, Dataset

    train_rows = [record.to_dict() for record in resolve_manifest_records(config.train_manifest, split="train")]
    eval_rows = [record.to_dict() for record in resolve_manifest_records(config.test_manifest, split="test")]
    train_dataset = Dataset.from_list(train_rows).cast_column("audio_filepath", Audio(sampling_rate=16_000))
    eval_dataset = Dataset.from_list(eval_rows).cast_column("audio_filepath", Audio(sampling_rate=16_000))

    def prepare_batch(batch: dict[str, Any]) -> dict[str, Any]:
        audio = batch["audio_filepath"]
        batch["input_features"] = processor.feature_extractor(
            audio["array"],
            sampling_rate=audio["sampling_rate"],
        ).input_features[0]
        batch["labels"] = processor.tokenizer(batch["text"]).input_ids
        return batch

    train_dataset = train_dataset.map(prepare_batch, remove_columns=train_dataset.column_names)
    eval_dataset = eval_dataset.map(prepare_batch, remove_columns=eval_dataset.column_names)
    return train_dataset, eval_dataset


class DataCollatorSpeechSeq2SeqWithPadding:
    """Pad Whisper input features and token labels for seq2seq training."""

    def __init__(self, processor: Any) -> None:
        """Store the Whisper processor used for padding."""

        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        """Return a padded batch compatible with Whisper training."""

        import torch

        input_features = [{"input_features": feature["input_features"]} for feature in features]
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_features = [{"input_ids": feature["labels"]} for feature in features]
        labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().cpu().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch
