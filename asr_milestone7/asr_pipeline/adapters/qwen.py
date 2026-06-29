"""Qwen ASR adapter with milestone-5 smoke prediction and training hooks."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

from asr_pipeline.adapters.base import BaseASRAdapter
from asr_pipeline.collators import QwenChatASRCollator


class QwenASRAdapter(BaseASRAdapter):
    """Adapter for all supported Qwen ASR model sizes.

    Supported examples:
    - Qwen/Qwen3-ASR-0.6B
    - Qwen/Qwen3-ASR-1.7B
    """

    family = "qwen"

    def prepare_dataset(self, dataset: Mapping[str, list[dict[str, Any]]], *args: Any, **kwargs: Any) -> dict[str, list[dict[str, Any]]]:
        """Prepare rows for Qwen chat-style ASR without loading models."""
        # TODO: Replace placeholder messages with the exact Qwen processor chat template.
        from asr_pipeline.data import prepare_rows_for_family

        return prepare_rows_for_family(dataset, family=self.family, preparation_settings=kwargs.get("preparation_settings"))

    def build_collator(self, *args: Any, **kwargs: Any) -> QwenChatASRCollator:
        """Build the placeholder Qwen chat ASR collator."""
        # TODO: Inject the real Qwen processor/tokenizer in the training milestone.
        return QwenChatASRCollator(smoke_mode=kwargs.get("smoke_mode", self.config.smoke_mode))

    def load_model(self, *args: Any, **kwargs: Any) -> Any:
        """Placeholder model-loading hook using the configured local cache later."""
        if kwargs.get("smoke_mode", self.config.smoke_mode):
            return {"family": self.family, "model_name": self.config.model_name, "smoke_model": True}
        cache_dir = kwargs.get("local_model_cache_dir", self.config.local_model_cache_dir)
        raise NotImplementedError(
            f"Real Qwen ASR loading is planned for a later milestone. Intended cache dir: {cache_dir}"
        )

    def predict(self, rows: Sequence[Mapping[str, Any]], *args: Any, **kwargs: Any) -> list[dict[str, str]]:
        """Return deterministic smoke predictions or defer real inference."""
        # TODO: Replace smoke output with real Qwen ASR inference + adapter loading.
        smoke_mode = kwargs.get("smoke_mode", self.config.smoke_mode)
        tuned_adapter_path = kwargs.get("tuned_adapter_path")
        if tuned_adapter_path is not None and not Path(tuned_adapter_path).exists():
            raise FileNotFoundError(f"Tuned adapter path does not exist: {tuned_adapter_path}")

        if not smoke_mode:
            cache_dir = kwargs.get("local_model_cache_dir", self.config.local_model_cache_dir)
            raise NotImplementedError(
                "Real Qwen ASR prediction is planned for a later milestone. "
                f"It should load the base model from cache_dir={cache_dir} and "
                f"merge/load tuned adapter checkpoint={tuned_adapter_path!r} when provided."
            )

        # Smoke mode still exercises the base-vs-tuned API branch.  Base output
        # intentionally contains a family prefix; tuned output returns the
        # reference exactly to simulate an improved adapter without downloads.
        self.load_model(smoke_mode=True, local_model_cache_dir=kwargs.get("local_model_cache_dir", self.config.local_model_cache_dir))
        predictions: list[dict[str, str]] = []
        for row in rows:
            reference = str(row.get("reference", row.get("target_text", row.get("labels_text", row.get("text", "")))))
            prediction_text = reference if tuned_adapter_path else f"[smoke qwen prediction] {reference}"
            predictions.append(
                {
                    "uid": str(row["uid"]),
                    "reference": reference,
                    "prediction": prediction_text,
                }
            )
        return predictions

    def train(self, *args: Any, **kwargs: Any) -> Any:
        """Run Qwen ASR training through the milestone-5 smoke-safe LoRA path."""
        # TODO: Replace smoke mock with real Qwen ASR + Unsloth LoRA where available.
        smoke_mode = kwargs.get("smoke_mode", self.config.smoke_mode)
        if not smoke_mode:
            raise NotImplementedError(
                "Real Qwen ASR LoRA training is planned for a later milestone. "
                f"Use local_model_cache_dir={kwargs.get('local_model_cache_dir', self.config.local_model_cache_dir)!r}."
            )
        loaded_model = self.load_model(smoke_mode=True, local_model_cache_dir=kwargs.get("local_model_cache_dir", self.config.local_model_cache_dir))
        from asr_pipeline.train import run_smoke_lora_training

        result = run_smoke_lora_training(
            config=self.config,
            family=self.family,
            prepared_data=kwargs["prepared_data"],
            collator=kwargs["collator"],
            checkpoint_dir=kwargs["checkpoint_dir"],
        )
        result["model_loading"] = loaded_model
        return result
