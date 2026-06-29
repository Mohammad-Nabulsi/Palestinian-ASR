"""Model registry and architecture dispatch for supported ASR families."""

from __future__ import annotations

from typing import Type

from asr_pipeline.config import ASRConfig
from asr_pipeline.adapters.base import BaseASRAdapter
from asr_pipeline.adapters.whisper import WhisperASRAdapter
from asr_pipeline.adapters.qwen import QwenASRAdapter
from asr_pipeline.adapters.omni import OmniASRAdapter

WHISPER_MODELS = {
    "openai/whisper-medium",
    "openai/whisper-large-v3",
}

QWEN_MODELS = {
    "Qwen/Qwen3-ASR-0.6B",
    "Qwen/Qwen3-ASR-1.7B",
}

OMNI_MODELS = {
    "Omni ASR 300M",
    "Omni ASR 1B",
}

_MODEL_TO_FAMILY = {
    **{model_name: "whisper" for model_name in WHISPER_MODELS},
    **{model_name: "qwen" for model_name in QWEN_MODELS},
    **{model_name: "omni" for model_name in OMNI_MODELS},
}

_FAMILY_TO_ADAPTER: dict[str, Type[BaseASRAdapter]] = {
    "whisper": WhisperASRAdapter,
    "qwen": QwenASRAdapter,
    "omni": OmniASRAdapter,
}


def supported_model_names() -> tuple[str, ...]:
    """Return all supported model names in a stable order."""
    return tuple(sorted(_MODEL_TO_FAMILY))


def get_model_family(model_name: str) -> str:
    """Resolve a model name to an architecture family.

    Returns one of: ``whisper``, ``qwen``, or ``omni``.
    """
    try:
        return _MODEL_TO_FAMILY[model_name]
    except KeyError as exc:
        supported = "\n  - ".join(supported_model_names())
        raise ValueError(
            f"Unsupported model_name: {model_name!r}. Supported values:\n  - {supported}"
        ) from exc


def get_adapter_class(model_name: str) -> Type[BaseASRAdapter]:
    """Return the adapter class for a supported model name."""
    family = get_model_family(model_name)
    return _FAMILY_TO_ADAPTER[family]


def create_adapter(config: ASRConfig) -> BaseASRAdapter:
    """Instantiate the correct adapter for the config's model_name."""
    adapter_cls = get_adapter_class(config.model_name)
    return adapter_cls(config=config)
