"""ASR model-family adapters."""

from asr_pipeline.adapters.base import BaseASRAdapter
from asr_pipeline.adapters.whisper import WhisperASRAdapter
from asr_pipeline.adapters.qwen import QwenASRAdapter
from asr_pipeline.adapters.omni import OmniASRAdapter

__all__ = [
    "BaseASRAdapter",
    "WhisperASRAdapter",
    "QwenASRAdapter",
    "OmniASRAdapter",
]
