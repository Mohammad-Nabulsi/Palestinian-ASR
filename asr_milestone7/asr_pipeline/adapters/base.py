"""Base adapter interface for ASR model families."""

from __future__ import annotations

from abc import ABC
from typing import Any

from asr_pipeline.config import ASRConfig


class BaseASRAdapter(ABC):
    """Shared interface for all ASR architecture adapters.

    Milestone 1 defines the contract only. Later milestones will implement
    real dataset preparation, collators, model loading, training, and inference.
    """

    family: str = "base"

    def __init__(self, config: ASRConfig) -> None:
        self.config = config

    def prepare_dataset(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def build_collator(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def predict(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def train(self, *args: Any, **kwargs: Any) -> Any:
        raise NotImplementedError

    def summary(self) -> dict[str, Any]:
        """Return a small adapter summary useful for notebooks and logs."""
        return {
            "model_name": self.config.model_name,
            "family": self.family,
            "adapter_class": self.__class__.__name__,
            "smoke_mode": self.config.smoke_mode,
            "run_name": self.config.run_name,
        }
