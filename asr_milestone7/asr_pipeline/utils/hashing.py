"""Stable hashing helpers for configs and dataset identities."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping, Sequence


def stable_json_dumps(value: Any) -> str:
    """Dump a Python value to canonical JSON for stable hashing."""
    return json.dumps(value, sort_keys=True, ensure_ascii=False, separators=(",", ":"))


def stable_hash(value: Any, length: int = 12) -> str:
    """Return a short SHA256 hash for any JSON-serializable value."""
    digest = hashlib.sha256(stable_json_dumps(value).encode("utf-8")).hexdigest()
    return digest[:length]


def config_hash(config: Mapping[str, Any] | Any, length: int = 12) -> str:
    """Return a stable hash for a config mapping or dataclass-like object."""
    if hasattr(config, "to_dict"):
        payload = config.to_dict()
    elif hasattr(config, "__dict__"):
        payload = vars(config)
    else:
        payload = config
    return stable_hash(payload, length=length)


def dataset_fingerprint(paths: Sequence[str | Path], length: int = 12) -> str:
    """Hash dataset path identities without reading the full datasets.

    This is intentionally lightweight for milestone 1. Later milestones can add
    content-aware fingerprints over manifests, row counts, or checksums.
    """
    normalized = [str(Path(path).expanduser()) for path in paths]
    return stable_hash(normalized, length=length)
