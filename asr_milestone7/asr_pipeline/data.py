"""Dataset loading, fake smoke data, architecture preparation, and caching."""

from __future__ import annotations

import math
import shutil
import struct
import wave
from pathlib import Path
from typing import Any, Mapping

from asr_pipeline.collators import (
    OmniASRCollator,
    QwenChatASRCollator,
    WhisperSeq2SeqCollator,
)
from asr_pipeline.config import ASRConfig
from asr_pipeline.registry import get_model_family
from asr_pipeline.utils.hashing import config_hash, stable_hash
from asr_pipeline.utils.io import ensure_dir, read_json, read_jsonl, write_json, write_jsonl
from asr_pipeline.utils.logging import get_logger

LOGGER = get_logger(__name__)

REQUIRED_COLUMNS = {
    "uid",
    "audio_path",
    "text",
    "duration",
    "sample_rate",
    "source",
}

SMOKE_REPRESENTATIVE_MODELS = {
    "whisper": "openai/whisper-medium",
    "qwen": "Qwen/Qwen3-ASR-0.6B",
    "omni": "Omni ASR 300M",
}

DEFAULT_PREPARATION_SETTINGS: dict[str, Any] = {
    "schema_version": 1,
    "whisper_max_audio_seconds": 30.0,
    "qwen_system_prompt": "You are an ASR system. Transcribe the provided Arabic speech exactly.",
    "qwen_user_prompt": "Transcribe this audio.",
    "omni_task": "asr",
    "language": "ar",
}


def _write_tiny_wav(path: str | Path, *, sample_rate: int = 16_000, duration: float = 0.25, frequency: float = 440.0) -> None:
    """Create a deterministic tiny mono PCM WAV file."""
    path = Path(path)
    ensure_dir(path.parent)
    n_samples = max(1, int(sample_rate * duration))
    amplitude = 0.15

    with wave.open(str(path), "wb") as wav_file:
        wav_file.setnchannels(1)
        wav_file.setsampwidth(2)
        wav_file.setframerate(sample_rate)
        for i in range(n_samples):
            sample = amplitude * math.sin(2 * math.pi * frequency * (i / sample_rate))
            wav_file.writeframes(struct.pack("<h", int(sample * 32767)))


def generate_fake_smoke_dataset(
    output_dir: str | Path = "data/smoke",
    *,
    sample_rate: int = 16_000,
    duration: float = 0.25,
    overwrite: bool = True,
) -> dict[str, Any]:
    """Generate one train, one validation, and one test sample as JSONL + WAV.

    The JSONL schema is:
    ``uid``, ``audio_path``, ``text``, ``duration``, ``sample_rate``, ``source``.
    Audio paths are stored relative to each JSONL file, making the fake dataset
    portable inside the project folder.
    """
    output_dir = Path(output_dir)
    wav_dir = output_dir / "wav"
    ensure_dir(output_dir)
    ensure_dir(wav_dir)

    split_text = {
        "train": "مرحبا هذا مثال تدريب صغير",
        "val": "مرحبا هذا مثال تحقق صغير",
        "test": "مرحبا هذا مثال اختبار صغير",
    }
    split_frequency = {"train": 440.0, "val": 554.37, "test": 659.25}

    split_paths: dict[str, str] = {}
    rows_by_split: dict[str, list[dict[str, Any]]] = {}

    for split, text in split_text.items():
        wav_path = wav_dir / f"{split}_smoke.wav"
        jsonl_path = output_dir / f"{split}.jsonl"
        if overwrite or not wav_path.exists():
            _write_tiny_wav(
                wav_path,
                sample_rate=sample_rate,
                duration=duration,
                frequency=split_frequency[split],
            )

        row = {
            "uid": f"smoke_{split}_0001",
            "audio_path": str(wav_path.relative_to(output_dir)),
            "text": text,
            "duration": duration,
            "sample_rate": sample_rate,
            "source": "fake_smoke",
        }
        write_jsonl(jsonl_path, [row])
        split_paths[split] = str(jsonl_path)
        rows_by_split[split] = [row]

    return {"split_paths": split_paths, "rows": rows_by_split, "output_dir": str(output_dir)}


def split_paths_from_config(config: ASRConfig) -> dict[str, str]:
    """Return train/validation/test paths from the resolved config."""
    return {
        "train": config.train_path,
        "val": config.val_path,
        "test": config.test_path,
    }


def _resolve_audio_path(audio_path: str | Path, metadata_path: str | Path) -> Path:
    audio_path = Path(audio_path)
    if audio_path.is_absolute():
        return audio_path
    return Path(metadata_path).parent / audio_path


def validate_dataset_rows(rows: list[dict[str, Any]], *, metadata_path: str | Path, split: str) -> list[dict[str, Any]]:
    """Validate required JSONL columns and audio-file existence."""
    if not rows:
        raise ValueError(f"Split {split!r} is empty: {metadata_path}")

    validated: list[dict[str, Any]] = []
    for idx, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {idx} in split {split!r} is not a JSON object: {metadata_path}")

        missing = sorted(REQUIRED_COLUMNS - set(row))
        if missing:
            raise ValueError(
                f"Row {idx} in split {split!r} is missing required column(s): {', '.join(missing)}"
            )

        audio_path = _resolve_audio_path(row["audio_path"], metadata_path)
        if not audio_path.exists():
            raise FileNotFoundError(
                f"Audio file for row {idx} in split {split!r} does not exist: {audio_path}"
            )

        duration = float(row["duration"])
        sample_rate = int(row["sample_rate"])
        if duration <= 0:
            raise ValueError(f"Row {idx} in split {split!r} has non-positive duration: {duration}")
        if sample_rate <= 0:
            raise ValueError(f"Row {idx} in split {split!r} has non-positive sample_rate: {sample_rate}")

        normalized = dict(row)
        normalized["audio_path"] = str(audio_path)
        normalized["duration"] = duration
        normalized["sample_rate"] = sample_rate
        normalized["split"] = split
        validated.append(normalized)

    return validated


def load_dataset_file(path: str | Path, *, split: str) -> list[dict[str, Any]]:
    """Load and validate a supported dataset metadata file."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Dataset metadata file does not exist for split {split!r}: {path}")
    if path.suffix.lower() != ".jsonl":
        raise ValueError(f"Unsupported dataset metadata format for split {split!r}: {path}. JSONL is required for milestone 2.")
    return validate_dataset_rows(read_jsonl(path), metadata_path=path, split=split)


def load_datasets(split_paths: Mapping[str, str | Path]) -> dict[str, list[dict[str, Any]]]:
    """Load train/val/test JSONL metadata into Python lists of dictionaries."""
    expected = {"train", "val", "test"}
    missing = sorted(expected - set(split_paths))
    if missing:
        raise ValueError(f"Missing split path(s): {', '.join(missing)}")
    return {split: load_dataset_file(split_paths[split], split=split) for split in ("train", "val", "test")}


def _prepare_whisper_rows(rows: list[dict[str, Any]], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    max_seconds = float(settings["whisper_max_audio_seconds"])
    prepared: list[dict[str, Any]] = []
    for row in rows:
        if float(row["duration"]) > max_seconds:
            continue
        item = dict(row)
        item.update(
            {
                "architecture": "whisper",
                "input_audio_path": row["audio_path"],
                "labels_text": row["text"],
                "task": "transcribe",
            }
        )
        prepared.append(item)
    return prepared


def _prepare_qwen_rows(rows: list[dict[str, Any]], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    system_prompt = str(settings["qwen_system_prompt"])
    user_prompt = str(settings["qwen_user_prompt"])
    prepared: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        messages = [
            {"role": "system", "content": system_prompt},
            {
                "role": "user",
                "content": [
                    {"type": "audio", "audio_path": row["audio_path"]},
                    {"type": "text", "text": user_prompt},
                ],
            },
            {"role": "assistant", "content": row["text"]},
        ]
        item.update(
            {
                "architecture": "qwen",
                "prompt_text": user_prompt,
                "target_text": row["text"],
                "messages": messages,
            }
        )
        prepared.append(item)
    return prepared


def _prepare_omni_rows(rows: list[dict[str, Any]], settings: Mapping[str, Any]) -> list[dict[str, Any]]:
    prepared: list[dict[str, Any]] = []
    for row in rows:
        item = dict(row)
        item.update(
            {
                "architecture": "omni",
                "task": settings["omni_task"],
                "language": settings["language"],
                "input_audio_path": row["audio_path"],
                "target_text": row["text"],
            }
        )
        prepared.append(item)
    return prepared


def prepare_rows_for_family(
    datasets: Mapping[str, list[dict[str, Any]]],
    *,
    family: str,
    preparation_settings: Mapping[str, Any] | None = None,
) -> dict[str, list[dict[str, Any]]]:
    """Prepare loaded rows with architecture-specific fields."""
    settings = {**DEFAULT_PREPARATION_SETTINGS, **dict(preparation_settings or {})}
    if family == "whisper":
        fn = _prepare_whisper_rows
    elif family == "qwen":
        fn = _prepare_qwen_rows
    elif family == "omni":
        fn = _prepare_omni_rows
    else:
        raise ValueError(f"Unsupported model family for preparation: {family!r}")

    return {split: fn(rows, settings) for split, rows in datasets.items()}


def build_collator_for_family(family: str, *, smoke_mode: bool = True) -> Any:
    """Create the correct placeholder collator for a model family."""
    if family == "whisper":
        return WhisperSeq2SeqCollator(smoke_mode=smoke_mode)
    if family == "qwen":
        return QwenChatASRCollator(smoke_mode=smoke_mode)
    if family == "omni":
        return OmniASRCollator(smoke_mode=smoke_mode)
    raise ValueError(f"Unsupported model family for collator: {family!r}")


def _path_identity(path: str | Path) -> dict[str, Any]:
    path = Path(path)
    resolved = path.resolve()
    stat = resolved.stat() if resolved.exists() else None
    return {
        "path": str(resolved),
        "exists": resolved.exists(),
        "size": stat.st_size if stat else None,
        "mtime_ns": stat.st_mtime_ns if stat else None,
    }


def preparation_cache_key(
    config: ASRConfig,
    *,
    family: str,
    split_paths: Mapping[str, str | Path],
    preparation_settings: Mapping[str, Any] | None = None,
) -> str:
    """Create a cache key covering model, data paths, settings, and config."""
    settings = {**DEFAULT_PREPARATION_SETTINGS, **dict(preparation_settings or {})}
    relevant_config_hash = config_hash(config)
    payload = {
        "model_family": family,
        "model_name": config.model_name,
        "dataset_paths": {split: _path_identity(path) for split, path in sorted(split_paths.items())},
        "preparation_settings": settings,
        "relevant_config_hash": relevant_config_hash,
    }
    return stable_hash(payload, length=16)


def _cache_dir(config: ASRConfig, family: str, key: str) -> Path:
    safe_model = config.model_name.replace("/", "__").replace(" ", "_")
    return Path(config.output_dir) / "prepared" / f"{family}__{safe_model}__{key}"


def save_prepared_cache(cache_dir: str | Path, prepared: Mapping[str, list[dict[str, Any]]], metadata: Mapping[str, Any]) -> None:
    """Persist prepared split JSONL files and metadata."""
    cache_dir = ensure_dir(cache_dir)
    for split, rows in prepared.items():
        write_jsonl(cache_dir / f"{split}.jsonl", rows)
    write_json(cache_dir / "metadata.json", dict(metadata))


def load_prepared_cache(cache_dir: str | Path) -> tuple[dict[str, list[dict[str, Any]]], dict[str, Any]]:
    """Load prepared split JSONL files and metadata from cache."""
    cache_dir = Path(cache_dir)
    prepared = {
        split: read_jsonl(cache_dir / f"{split}.jsonl")
        for split in ("train", "val", "test")
    }
    metadata = read_json(cache_dir / "metadata.json")
    return prepared, metadata


def clear_preparation_cache(config: ASRConfig) -> None:
    """Remove all milestone-2 preparation caches under ``output_dir/prepared``."""
    prepared_dir = Path(config.output_dir) / "prepared"
    if prepared_dir.exists():
        for child in prepared_dir.iterdir():
            if child.is_dir():
                shutil.rmtree(child)


def prepare_data_and_collator(
    config: ASRConfig,
    adapter: Any,
    split_paths: Mapping[str, str | Path] | None = None,
    *,
    preparation_settings: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Load, architecture-prepare, cache, and build collator for train/val/test.

    No model loading, inference, or training happens here.
    """
    family = getattr(adapter, "family", None) or get_model_family(config.model_name)
    resolved_family = get_model_family(config.model_name)
    if family != resolved_family:
        raise ValueError(
            f"Adapter family {family!r} does not match model_name family {resolved_family!r}"
        )

    split_paths = dict(split_paths or split_paths_from_config(config))
    key = preparation_cache_key(
        config,
        family=family,
        split_paths=split_paths,
        preparation_settings=preparation_settings,
    )
    cache_dir = _cache_dir(config, family, key)
    expected_files = [cache_dir / f"{split}.jsonl" for split in ("train", "val", "test")] + [cache_dir / "metadata.json"]

    if all(path.exists() for path in expected_files):
        prepared, metadata = load_prepared_cache(cache_dir)
        metadata = dict(metadata)
        metadata.update({"cache_status": "loaded", "cache_created": False, "cache_loaded": True})
        LOGGER.info("Loaded prepared data cache: %s", cache_dir)
    else:
        datasets = load_datasets(split_paths)
        settings = {**DEFAULT_PREPARATION_SETTINGS, **dict(preparation_settings or {})}
        prepared = prepare_rows_for_family(datasets, family=family, preparation_settings=settings)
        split_counts = {split: len(rows) for split, rows in prepared.items()}
        metadata = {
            "cache_status": "created",
            "cache_created": True,
            "cache_loaded": False,
            "cache_key": key,
            "cache_dir": str(cache_dir),
            "model_family": family,
            "model_name": config.model_name,
            "split_paths": {split: str(path) for split, path in split_paths.items()},
            "split_counts": split_counts,
            "preparation_settings": settings,
            "relevant_config_hash": config_hash(config),
        }
        save_prepared_cache(cache_dir, prepared, metadata)
        LOGGER.info("Created prepared data cache: %s", cache_dir)

    collator = adapter.build_collator(smoke_mode=config.smoke_mode)
    return {
        "prepared": prepared,
        "collator": collator,
        "metadata": metadata,
        "cache_dir": str(cache_dir),
        "cache_key": key,
    }
