"""Model-agnostic dataset manifest loading and smoke-data generation."""

from __future__ import annotations

import csv
import json
import math
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable


AUDIO_COLUMNS = ("audio_filepath", "audio_path", "path", "file", "audio")
TEXT_COLUMNS = ("text", "transcript", "sentence", "normalized_text")


@dataclass(frozen=True)
class ManifestRecord:
    """A normalized ASR manifest row."""

    audio_filepath: Path
    text: str
    split: str | None = None
    sample_id: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Return a JSON-serializable representation of the record."""

        return {
            "audio_filepath": str(self.audio_filepath),
            "text": self.text,
            "split": self.split,
            "sample_id": self.sample_id,
        }


def load_manifest(path: str | Path) -> list[dict[str, Any]]:
    """Load a CSV, JSON, JSONL, or Parquet manifest into row dictionaries."""

    manifest_path = Path(path)
    print(f"[data] Loading manifest: {manifest_path}")
    suffix = manifest_path.suffix.lower()
    if suffix == ".csv":
        with manifest_path.open("r", encoding="utf-8-sig", newline="") as handle:
            rows = list(csv.DictReader(handle))
        print(f"[data] Loaded {len(rows)} rows from CSV.")
        return rows
    if suffix in {".jsonl", ".ndjson"}:
        with manifest_path.open("r", encoding="utf-8") as handle:
            rows = [json.loads(line) for line in handle if line.strip()]
        print(f"[data] Loaded {len(rows)} rows from JSONL.")
        return rows
    if suffix == ".json":
        with manifest_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        if isinstance(payload, list):
            print(f"[data] Loaded {len(payload)} rows from JSON.")
            return payload
        if isinstance(payload, dict):
            for key in ("data", "rows", "samples", "records"):
                if isinstance(payload.get(key), list):
                    print(f"[data] Loaded {len(payload[key])} rows from JSON key '{key}'.")
                    return payload[key]
        raise ValueError(f"JSON manifest must contain a list of rows: {manifest_path}")
    if suffix == ".parquet":
        try:
            import pandas as pd
        except ImportError as exc:
            raise ImportError("Reading Parquet manifests requires pandas and pyarrow.") from exc
        rows = pd.read_parquet(manifest_path).to_dict(orient="records")
        print(f"[data] Loaded {len(rows)} rows from Parquet.")
        return rows
    raise ValueError(f"Unsupported manifest format: {manifest_path}")


def resolve_manifest_records(path: str | Path, split: str | None = None) -> list[ManifestRecord]:
    """Normalize a manifest into ASR records with absolute audio paths."""

    manifest_path = Path(path).resolve()
    rows = load_manifest(manifest_path)
    records: list[ManifestRecord] = []
    for idx, row in enumerate(rows):
        audio_value = _first_present(row, AUDIO_COLUMNS)
        text_value = _first_present(row, TEXT_COLUMNS)
        if audio_value is None or text_value is None:
            raise ValueError(
                f"Manifest {manifest_path} must include an audio column {AUDIO_COLUMNS} "
                f"and a text column {TEXT_COLUMNS}."
            )
        audio_path = Path(str(audio_value))
        if not audio_path.is_absolute():
            audio_path = (manifest_path.parent / audio_path).resolve()
        sample_id = str(row.get("sample_id") or row.get("id") or f"{split or 'sample'}-{idx:06d}")
        records.append(
            ManifestRecord(
                audio_filepath=audio_path,
                text=str(text_value),
                split=split or row.get("split"),
                sample_id=sample_id,
            )
        )
    print(f"[data] Resolved {len(records)} records for split '{split or 'unspecified'}'.")
    return records


def create_smoke_asr_dataset(root: str | Path) -> dict[str, Path]:
    """Create one train, validation, and test WAV sample plus CSV manifests."""

    root_path = Path(root).resolve()
    print(f"[smoke] Preparing smoke ASR dataset under: {root_path}")
    audio_dir = root_path / "audio"
    manifest_dir = root_path / "manifests"
    audio_dir.mkdir(parents=True, exist_ok=True)
    manifest_dir.mkdir(parents=True, exist_ok=True)

    samples = {
        "train": ("train_000.wav", "مرحبا يا عالم", 330.0),
        "validation": ("validation_000.wav", "هذا اختبار قصير", 440.0),
        "test": ("test_000.wav", "صوت فلسطيني واضح", 550.0),
    }
    manifests: dict[str, Path] = {}
    for split, (filename, text, frequency) in samples.items():
        wav_path = audio_dir / filename
        if not wav_path.exists():
            _write_tone_wav(wav_path, frequency_hz=frequency, seconds=1.0)
        manifest_path = manifest_dir / f"{split}.csv"
        _write_manifest_csv(manifest_path, wav_path, text, split)
        manifests[split] = manifest_path
        print(f"[smoke] {split}: audio={wav_path.name}, manifest={manifest_path}")
    return manifests


def audio_duration_seconds(path: str | Path) -> float | None:
    """Return WAV duration in seconds when it can be read cheaply."""

    try:
        with wave.open(str(path), "rb") as handle:
            frames = handle.getnframes()
            rate = handle.getframerate()
        return frames / float(rate) if rate else None
    except (wave.Error, FileNotFoundError, OSError):
        return None


def _first_present(row: dict[str, Any], candidates: Iterable[str]) -> Any | None:
    """Return the first non-empty value from a row for the candidate keys."""

    for key in candidates:
        value = row.get(key)
        if value is not None and str(value).strip() != "":
            return value
    return None


def _write_manifest_csv(path: Path, audio_path: Path, text: str, split: str) -> None:
    """Write a one-row ASR manifest."""

    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["sample_id", "audio_filepath", "text", "split"])
        writer.writeheader()
        writer.writerow(
            {
                "sample_id": f"{split}-000000",
                "audio_filepath": str(audio_path),
                "text": text,
                "split": split,
            }
        )


def _write_tone_wav(path: Path, frequency_hz: float, seconds: float, sample_rate: int = 16_000) -> None:
    """Write a deterministic mono PCM WAV tone for smoke testing."""

    amplitude = 0.2
    total_frames = int(sample_rate * seconds)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(1)
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        for frame_idx in range(total_frames):
            value = int(32767 * amplitude * math.sin(2 * math.pi * frequency_hz * frame_idx / sample_rate))
            handle.writeframesraw(value.to_bytes(2, byteorder="little", signed=True))
