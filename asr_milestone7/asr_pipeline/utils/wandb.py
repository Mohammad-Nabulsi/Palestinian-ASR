"""Small optional Weights & Biases helper.

The pipeline must be able to run in clean smoke-test environments where
``wandb`` is not installed and where network access is not desired.  This module
therefore exposes a no-op compatible wrapper that records what would have been
logged and only calls the real SDK when it is explicitly available.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping

from asr_pipeline.utils.logging import get_logger

LOGGER = get_logger(__name__)

VALID_WANDB_MODES = {"online", "offline", "disabled"}


class WandbRun:
    """Optional W&B run wrapper with a no-op fallback."""

    def __init__(self, *, mode: str, project: str, name: str, config: Mapping[str, Any]) -> None:
        mode = str(mode or "disabled").lower()
        if mode not in VALID_WANDB_MODES:
            raise ValueError(f"wandb_mode must be one of {sorted(VALID_WANDB_MODES)}, got {mode!r}.")

        self.mode = mode
        self.project = project
        self.name = name
        self.config = dict(config)
        self.enabled = False
        self.backend = "disabled" if mode == "disabled" else "noop"
        self.logs: list[dict[str, Any]] = []
        self.artifacts: list[str] = []
        self._run: Any | None = None
        self.status_message = "W&B disabled by config."

        if mode == "disabled":
            LOGGER.info("W&B disabled for run %s", name)
            return

        try:
            import wandb  # type: ignore
        except Exception as exc:  # pragma: no cover - depends on local env
            self.status_message = f"W&B SDK unavailable; using no-op logger: {exc!r}"
            LOGGER.warning(self.status_message)
            return

        if mode == "offline":
            os.environ.setdefault("WANDB_MODE", "offline")

        self._run = wandb.init(project=project, name=name, mode=mode, config=dict(config))
        self.enabled = True
        self.backend = "wandb"
        self.status_message = f"W&B initialized in {mode!r} mode."
        LOGGER.info(self.status_message)

    def log(self, data: Mapping[str, Any]) -> None:
        """Log scalar data to W&B or store it locally in no-op mode."""
        payload = dict(data)
        self.logs.append(payload)
        if self._run is not None:  # pragma: no cover - depends on local env
            self._run.log(payload)

    def log_artifact(self, path: str | Path, *, artifact_type: str = "file") -> None:
        """Log an artifact if W&B is enabled; otherwise just remember the path."""
        path = Path(path)
        if not path.exists():
            return
        self.artifacts.append(str(path))
        if self._run is None:  # no-op mode
            return
        try:  # pragma: no cover - depends on local env
            import wandb  # type: ignore

            artifact_name = path.stem.replace("/", "_")
            artifact = wandb.Artifact(artifact_name, type=artifact_type)
            if path.is_dir():
                artifact.add_dir(str(path))
            else:
                artifact.add_file(str(path))
            self._run.log_artifact(artifact)
        except Exception as exc:
            LOGGER.warning("Could not log W&B artifact %s: %r", path, exc)

    def finish(self) -> None:
        """Finish the W&B run if a real run exists."""
        if self._run is not None:  # pragma: no cover - depends on local env
            self._run.finish()

    def summary(self) -> dict[str, Any]:
        """Return logger metadata suitable for training_metrics.json."""
        return {
            "mode": self.mode,
            "backend": self.backend,
            "enabled": self.enabled,
            "status_message": self.status_message,
            "logged_steps": len(self.logs),
            "artifact_count": len(self.artifacts),
            "artifacts": self.artifacts,
        }
