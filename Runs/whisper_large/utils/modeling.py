"""Whisper model loading and LoRA attachment helpers."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

from .config import WhisperLargeRunConfig


@dataclass
class WhisperBundle:
    """Container for model, processor, and backend metadata."""

    model: Any
    processor: Any
    backend: str


class SmokeProcessor:
    """Minimal processor placeholder used by the smoke backend."""

    tokenizer = None
    feature_extractor = None


class SmokeWhisperModel:
    """Small deterministic placeholder used when smoke mode is enabled."""

    def predict(self, reference: str) -> str:
        """Return a deterministic prediction for a smoke sample."""

        return reference


def load_whisper_bundle(config: WhisperLargeRunConfig, adapter_path: str | Path | None = None) -> WhisperBundle:
    """Load Whisper Large locally when available, optionally with a PEFT adapter."""

    if config.smoke_mode:
        print("[model] Smoke mode enabled. Using deterministic smoke model; no Hugging Face download.")
        return WhisperBundle(model=SmokeWhisperModel(), processor=SmokeProcessor(), backend="smoke")

    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    local_dir = Path(config.model_cache_dir)
    source = str(local_dir) if (local_dir / "config.json").exists() else config.model_name
    if source == str(local_dir):
        print(f"[model] Loading cached Whisper model from: {local_dir}")
    else:
        print(f"[model] Cached model not found. Downloading from Hugging Face: {config.model_name}")
        print("[model] This can take a while for Whisper Large.")
    print("[model] Loading processor...")
    processor = WhisperProcessor.from_pretrained(source, language=config.language, task=config.task)
    print("[model] Loading model weights...")
    model = WhisperForConditionalGeneration.from_pretrained(source)
    if source == config.model_name:
        local_dir.mkdir(parents=True, exist_ok=True)
        print(f"[model] Saving model and processor locally for future runs: {local_dir}")
        processor.save_pretrained(local_dir)
        model.save_pretrained(local_dir)
    if adapter_path is not None:
        from peft import PeftModel

        print(f"[model] Loading PEFT adapter from: {adapter_path}")
        model = PeftModel.from_pretrained(model, str(adapter_path))
    model.config.forced_decoder_ids = processor.get_decoder_prompt_ids(language=config.language, task=config.task)
    model.config.suppress_tokens = []
    print("[model] Model bundle is ready.")
    return WhisperBundle(model=model, processor=processor, backend="transformers")


def attach_lora(config: WhisperLargeRunConfig, model: Any) -> Any:
    """Attach the requested LoRA adapters using Unsloth when available, else PEFT."""

    if config.smoke_mode:
        print("[lora] Smoke mode enabled. Skipping real LoRA attachment.")
        return model

    try:
        from unsloth import FastModel

        print("[lora] Attaching LoRA adapters with Unsloth FastModel.")
        return FastModel.get_peft_model(
            model,
            r=config.lora_r,
            target_modules=config.lora_target_modules,
            lora_alpha=config.lora_alpha,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
            use_gradient_checkpointing=config.gradient_checkpointing,
            random_state=config.seed,
        )
    except ImportError:
        from peft import LoraConfig, get_peft_model

        print("[lora] Unsloth is not installed. Falling back to PEFT LoRA.")
        peft_config = LoraConfig(
            r=config.lora_r,
            lora_alpha=config.lora_alpha,
            target_modules=config.lora_target_modules,
            lora_dropout=config.lora_dropout,
            bias=config.lora_bias,
        )
        return get_peft_model(model, peft_config)
