"""Omni ASR adapter with runnable placeholder prediction and training hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from asr_pipeline.adapters.base import BaseASRAdapter
from asr_pipeline.collators import OmniASRCollator


class OmniASRAdapter(BaseASRAdapter):
    """Adapter for all supported Omni ASR model sizes.

    Supported examples:
    - Omni ASR 300M
    - Omni ASR 1B

    The real Omni backend is still pending, but this adapter now keeps the
    end-to-end pipeline runnable in both smoke and non-smoke configs by using a
    deterministic placeholder path. That lets notebooks exercise preparation,
    prediction, evaluation, checkpoint writing, and tuned-vs-base comparison
    without raising NotImplementedError.
    """

    family = "omni"

    def prepare_dataset(self, dataset: Mapping[str, list[dict[str, Any]]], *args: Any, **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
        """Prepare rows for Omni ASR without loading models."""
        from asr_pipeline.data import prepare_rows_for_family

        return prepare_rows_for_family(dataset, family=self.family, preparation_settings=kwargs.get("preparation_settings"))

    def build_collator(self, *args: Any, **kwargs: Any) -> OmniASRCollator:
        """Build the placeholder Omni ASR collator shared by 300M and 1B."""
        return OmniASRCollator(smoke_mode=kwargs.get("smoke_mode", self.config.smoke_mode))

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        """Return placeholder model-loading metadata for smoke and non-smoke runs."""
        smoke_mode = kwargs.get("smoke_mode", self.config.smoke_mode)
        cache_dir = kwargs.get("local_model_cache_dir", self.config.local_model_cache_dir)
        return {
            "family": self.family,
            "model_name": self.config.model_name,
            "smoke_mode": bool(smoke_mode),
            "local_model_cache_dir": str(cache_dir),
            "placeholder_backend": True,
            "real_model_loaded": False,
            "message": (
                "Using smoke placeholder model loading."
                if smoke_mode
                else "Using non-smoke placeholder Omni backend until the real loader is implemented."
            ),
        }

    def predict(self, rows: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any) -> list[dict[str, str]]:
        """Return deterministic base/tuned predictions for both smoke and non-smoke paths."""
        smoke_mode = kwargs.get("smoke_mode", self.config.smoke_mode)
        tuned_adapter_path = kwargs.get("tuned_adapter_path")
        if tuned_adapter_path is not None and not Path(tuned_adapter_path).exists():
            raise FileNotFoundError(f"Tuned adapter path does not exist: {tuned_adapter_path}")

        self.load_model(
            smoke_mode=smoke_mode,
            local_model_cache_dir=kwargs.get("local_model_cache_dir", self.config.local_model_cache_dir),
        )

        mode_prefix = "[smoke omni prediction]" if smoke_mode else "[omni placeholder prediction]"
        predictions: list[dict[str, str]] = []
        for row in rows:
            reference = str(row.get("reference", row.get("target_text", row.get("labels_text", row.get("text", "")))))
            prediction_text = reference if tuned_adapter_path else mode_prefix
            predictions.append(
                {
                    "uid": str(row["uid"]),
                    "reference": reference,
                    "prediction": prediction_text,
                }
            )
        return predictions

    def train(self, *args: Any, **kwargs: Any) -> Any:
        """Run the placeholder LoRA training path for smoke and non-smoke configs."""
        smoke_mode = kwargs.get("smoke_mode", self.config.smoke_mode)
        loaded_model = self.load_model(
            smoke_mode=smoke_mode,
            local_model_cache_dir=kwargs.get("local_model_cache_dir", self.config.local_model_cache_dir),
        )
        from asr_pipeline.train import run_smoke_lora_training

        result = run_smoke_lora_training(
            config=self.config,
            family=self.family,
            prepared_data=kwargs["prepared_data"],
            collator=kwargs["collator"],
            checkpoint_dir=kwargs["checkpoint_dir"],
        )
        result["model_loading"] = loaded_model
        if smoke_mode:
            result["training_mode"] = "smoke_mock_lora_training"
            result["used_mock_training"] = True
        else:
            result["training_mode"] = "non_smoke_placeholder_lora_training"
            result["used_mock_training"] = True
            result["placeholder_backend"] = True
            result["message"] = (
                "Non-smoke Omni training currently uses the deterministic placeholder backend. "
                "It writes checkpoints and enables end-to-end notebook execution, but it is not real model fine-tuning yet."
            )
        return result
