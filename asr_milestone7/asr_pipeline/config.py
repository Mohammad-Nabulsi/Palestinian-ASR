"""Configuration objects and YAML/JSON helpers for the ASR pipeline."""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, fields
from pathlib import Path
from typing import Any, Mapping

from asr_pipeline.utils.io import read_yaml, write_json


@dataclass(frozen=True)
class ASRConfig:
    """Resolved run configuration for the plug-and-play ASR pipeline.

    This config intentionally contains only architecture-independent fields.
    Model-family-specific training fields should be added in later milestones
    under explicit nested sections rather than hard-coding by model size.
    """

    model_name: str
    train_path: str
    val_path: str
    test_path: str
    output_dir: str = "outputs"
    run_name: str = "smoke_run"
    smoke_mode: bool = True
    wandb_mode: str = "disabled"
    learning_rate: float = 5e-5
    max_epochs: int = 50
    early_stopping_patience: int = 5
    seed: int = 42
    local_model_cache_dir: str = "models/cache"

    @classmethod
    def from_mapping(cls, data: Mapping[str, Any]) -> "ASRConfig":
        """Build a config from a mapping and reject unknown keys."""
        valid_keys = {field.name for field in fields(cls)}
        unknown_keys = sorted(set(data) - valid_keys)
        if unknown_keys:
            raise ValueError(
                "Unknown config key(s): " + ", ".join(unknown_keys)
            )

        missing_required = [
            field.name
            for field in fields(cls)
            if field.default is MISSING
            and field.default_factory is MISSING
            and field.name not in data
        ]
        if missing_required:
            raise ValueError(
                "Missing required config key(s): " + ", ".join(missing_required)
            )

        return cls(**dict(data))

    def to_dict(self) -> dict[str, Any]:
        """Return the config as a JSON/YAML-safe dictionary."""
        return asdict(self)


def load_config(path: str | Path) -> ASRConfig:
    """Load an ASRConfig from a YAML file."""
    data = read_yaml(path)
    if data is None:
        data = {}
    if not isinstance(data, dict):
        raise ValueError(f"Config YAML must contain a mapping/object: {path}")
    return ASRConfig.from_mapping(data)


def save_resolved_config_json(config: ASRConfig, path: str | Path) -> None:
    """Save the resolved dataclass config to JSON."""
    write_json(path, config.to_dict())
