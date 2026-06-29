"""Placeholder ASR collators used by milestone 2 smoke preparation.

These collators are deliberately lightweight. They do not require model
processors, tokenizers, or downloads. Later milestones can replace the minimal
placeholder tensors with processor-specific tensors while keeping the same
architecture dispatch surface.
"""

from __future__ import annotations

from typing import Any, Sequence


def _minimal_batch(batch: Sequence[dict[str, Any]], family: str) -> dict[str, Any]:
    """Return a small framework-agnostic batch dictionary.

    Lists are used instead of torch tensors so this milestone can run in a clean
    Python environment without installing model-training dependencies.
    """
    rows = list(batch)
    return {
        "family": family,
        "batch_size": len(rows),
        "uids": [row.get("uid") for row in rows],
        "audio_paths": [row.get("audio_path") for row in rows],
        "texts": [row.get("text") for row in rows],
        "durations": [row.get("duration") for row in rows],
    }


class WhisperSeq2SeqCollator:
    """Minimal callable collator for Whisper-style seq2seq ASR batches."""

    family = "whisper"

    def __init__(self, processor: Any | None = None, smoke_mode: bool = True) -> None:
        self.processor = processor
        self.smoke_mode = smoke_mode

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        output = _minimal_batch(batch, family=self.family)
        output.update(
            {
                "input_features": [[0.0] for _ in batch],
                "labels": [row.get("labels_text", row.get("text", "")) for row in batch],
                "task": "transcribe",
            }
        )
        return output


class QwenChatASRCollator:
    """Minimal callable collator for Qwen ASR chat-style examples."""

    family = "qwen"

    def __init__(self, processor: Any | None = None, smoke_mode: bool = True) -> None:
        self.processor = processor
        self.smoke_mode = smoke_mode

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        output = _minimal_batch(batch, family=self.family)
        output.update(
            {
                "messages": [row.get("messages", []) for row in batch],
                "prompt_texts": [row.get("prompt_text", "") for row in batch],
                "labels": [row.get("target_text", row.get("text", "")) for row in batch],
            }
        )
        return output


class OmniASRCollator:
    """Minimal callable collator shared by Omni ASR 300M and 1B."""

    family = "omni"

    def __init__(self, processor: Any | None = None, smoke_mode: bool = True) -> None:
        self.processor = processor
        self.smoke_mode = smoke_mode

    def __call__(self, batch: Sequence[dict[str, Any]]) -> dict[str, Any]:
        output = _minimal_batch(batch, family=self.family)
        output.update(
            {
                "input_values": [[0.0] for _ in batch],
                "labels": [row.get("target_text", row.get("text", "")) for row in batch],
                "language": [row.get("language", "ar") for row in batch],
            }
        )
        return output
