"""Plug-and-play ASR pipeline package."""

from asr_pipeline.config import ASRConfig, load_config, save_resolved_config_json
from asr_pipeline.registry import create_adapter, get_adapter_class, get_model_family, supported_model_names
from asr_pipeline.predict import predict
from asr_pipeline.evaluate import evaluate_predictions

__all__ = [
    "ASRConfig",
    "load_config",
    "save_resolved_config_json",
    "create_adapter",
    "get_adapter_class",
    "get_model_family",
    "supported_model_names",
    "predict",
    "evaluate_predictions",
]
