"""Configuration for the Whisper Large LoRA workflow."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any


@dataclass
class WhisperLargeRunConfig:
    """Run configuration used by the notebook and high-level APIs."""

    train_manifest: Path | None = None
    validation_manifest: Path | None = None
    test_manifest: Path | None = None
    output_dir: Path = Path("Runs/whisper_large/outputs")
    model_cache_dir: Path = Path("Runs/whisper_large/models/openai_whisper-large")
    model_name: str = "openai/whisper-large"
    model_key: str = "whisper_large"
    language: str = "ar"
    task: str = "transcribe"
    smoke_mode: bool = True
    seed: int = 3407
    lora_r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: list[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "fc1", "fc2"])
    lora_bias: str = "none"
    gradient_checkpointing: str = "unsloth"
    optim: str = "adamw_8bit"
    learning_rate: float = 1e-4
    num_train_epochs: int = 10
    per_device_train_batch_size: int = 1
    per_device_eval_batch_size: int = 1
    gradient_accumulation_steps: int = 4
    patience: int = 3
    save_total_limit: int = 3
    resume_from_checkpoint: str | None = None
    generation_max_new_tokens: int = 128

    def resolved(self) -> "WhisperLargeRunConfig":
        """Create output directories and return this config for fluent notebook use."""

        self.output_dir = Path(self.output_dir).resolve()
        self.model_cache_dir = Path(self.model_cache_dir).resolve()
        print(f"[config] Output directory: {self.output_dir}")
        print(f"[config] Model cache directory: {self.model_cache_dir}")
        print(f"[config] Model: {self.model_name}")
        print(f"[config] Smoke mode: {self.smoke_mode}")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        (self.output_dir / "predictions").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "metrics").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (self.output_dir / "smoke_data").mkdir(parents=True, exist_ok=True)
        return self

    @property
    def results_path(self) -> Path:
        """Return the shared results JSON path for this model run."""

        return self.output_dir / "results.json"

    @property
    def training_output_dir(self) -> Path:
        """Return the Hugging Face trainer output directory."""

        return self.output_dir / "checkpoints" / self.model_key

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-friendly configuration snapshot."""

        payload = asdict(self)
        for key, value in list(payload.items()):
            if isinstance(value, Path):
                payload[key] = str(value)
        return payload
