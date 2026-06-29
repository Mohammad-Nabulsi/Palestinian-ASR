#!/usr/bin/env python3
"""Prepare reusable tiny datasets for notebook smoke tests."""

from __future__ import annotations

import json
import subprocess
from pathlib import Path
from typing import Any

from asr_universal_trainer import dump_json, make_tiny_wav, write_jsonl

BUNDLE_DIR = Path(__file__).resolve().parent
DATASETS_DIR = BUNDLE_DIR / "datasets"
REGISTRY_PATH = DATASETS_DIR / "registry.json"


def build_short_dataset() -> dict[str, Any]:
    dataset_dir = DATASETS_DIR / "synthetic_short_1x1x1"
    audio_dir = dataset_dir / "audio"
    rows = [
        {"uid": "short_train", "split": "train", "text": "مرحبا هذا تسجيل تدريب قصير", "seconds": 0.8, "freq": 410.0},
        {"uid": "short_validation", "split": "validation", "text": "مرحبا هذا تسجيل تحقق قصير", "seconds": 0.7, "freq": 510.0},
        {"uid": "short_test", "split": "test", "text": "مرحبا هذا تسجيل اختبار قصير", "seconds": 0.9, "freq": 610.0},
    ]
    materialized = []
    for row in rows:
        wav_path = audio_dir / f"{row["uid"]}.wav"
        make_tiny_wav(wav_path, seconds=float(row["seconds"]), freq=float(row["freq"]))
        materialized.append(
            {
                "uid": row["uid"],
                "audio_path": str(wav_path),
                "text": row["text"],
                "duration": float(row["seconds"]),
                "split": row["split"],
                "language": "ar",
            }
        )
    write_jsonl(dataset_dir / "combined.jsonl", materialized)
    dump_json(
        dataset_dir / "dataset_summary.json",
        {
            "dataset_name": "synthetic_short_1x1x1",
            "rows": len(materialized),
            "splits": {split: sum(1 for row in materialized if row["split"] == split) for split in ["train", "validation", "test"]},
            "combined_path": str(dataset_dir / "combined.jsonl"),
        },
    )
    return {
        "format": "jsonl",
        "path": "synthetic_short_1x1x1/combined.jsonl",
        "columns": {"audio": "audio_path", "text": "text", "duration": "duration", "split": "split", "uid": "uid", "language": "language"},
        "language": "ar",
        "min_seconds": 0.05,
        "long_audio_policy": "drop",
        "summary_path": "synthetic_short_1x1x1/dataset_summary.json",
    }


def build_long_probe_dataset() -> dict[str, Any]:
    dataset_dir = DATASETS_DIR / "synthetic_long_train_probe"
    audio_dir = dataset_dir / "audio"
    rows = [
        {"uid": "probe_train_short", "split": "train", "text": "هذا تدريب قصير صالح", "seconds": 0.8, "freq": 420.0},
        {"uid": "probe_train_long", "split": "train", "text": "هذا تدريب طويل لاختبار ترشيح المقاطع الطويلة", "seconds": 31.0, "freq": 220.0},
        {"uid": "probe_validation", "split": "validation", "text": "هذا تحقق قصير", "seconds": 0.7, "freq": 520.0},
        {"uid": "probe_test", "split": "test", "text": "هذا اختبار قصير", "seconds": 0.9, "freq": 620.0},
    ]
    materialized = []
    for row in rows:
        wav_path = audio_dir / f"{row["uid"]}.wav"
        make_tiny_wav(wav_path, seconds=float(row["seconds"]), freq=float(row["freq"]))
        materialized.append(
            {
                "uid": row["uid"],
                "audio_path": str(wav_path),
                "text": row["text"],
                "duration": float(row["seconds"]),
                "split": row["split"],
                "language": "ar",
            }
        )
    write_jsonl(dataset_dir / "combined.jsonl", materialized)
    dump_json(
        dataset_dir / "dataset_summary.json",
        {
            "dataset_name": "synthetic_long_train_probe",
            "rows": len(materialized),
            "splits": {split: sum(1 for row in materialized if row["split"] == split) for split in ["train", "validation", "test"]},
            "combined_path": str(dataset_dir / "combined.jsonl"),
            "expected_behavior": {
                "whisper_train_rows_after_drop": 1,
                "qwen_train_rows_after_keep": 2,
            },
        },
    )
    return {
        "format": "jsonl",
        "path": "synthetic_long_train_probe/combined.jsonl",
        "columns": {"audio": "audio_path", "text": "text", "duration": "duration", "split": "split", "uid": "uid", "language": "language"},
        "language": "ar",
        "min_seconds": 0.05,
        "long_audio_policy": "drop",
        "summary_path": "synthetic_long_train_probe/dataset_summary.json",
    }


def ensure_levantine_dataset() -> dict[str, Any]:
    dataset_dir = DATASETS_DIR / "whisper_large_v3_levantine_1x1x1"
    combined = dataset_dir / "combined.jsonl"
    if not combined.exists():
        prep_script = BUNDLE_DIR / "prepare_levantine_whisper_large_v3_1x1x1.py"
        subprocess.run(["python3", str(prep_script)], cwd=BUNDLE_DIR, check=True)
    return {
        "format": "jsonl",
        "path": "whisper_large_v3_levantine_1x1x1/combined.jsonl",
        "columns": {"audio": "audio_path", "text": "text", "duration": "duration", "split": "split", "uid": "uid", "speaker_id": "speaker_id", "language": "language"},
        "language": "ar",
        "min_seconds": 0.05,
        "long_audio_policy": "drop",
        "summary_path": "whisper_large_v3_levantine_1x1x1/dataset_summary.json",
    }


def maybe_register_materialized_full_dataset() -> dict[str, Any] | None:
    dataset_dir = DATASETS_DIR / "omnilingual_300m_levantine_full_from_whisper_splits"
    combined = dataset_dir / "combined.jsonl"
    summary = dataset_dir / "dataset_summary.json"
    if not combined.exists() or not summary.exists():
        return None
    return {
        "format": "jsonl",
        "path": "omnilingual_300m_levantine_full_from_whisper_splits/combined.jsonl",
        "columns": {
            "audio": "audio_path",
            "text": "text",
            "duration": "duration",
            "split": "split",
            "uid": "uid",
            "speaker_id": "speaker_id",
            "language": "language",
        },
        "language": "ar",
        "min_seconds": 0.3,
        "long_audio_policy": "drop",
        "summary_path": "omnilingual_300m_levantine_full_from_whisper_splits/dataset_summary.json",
    }


def main() -> None:
    DATASETS_DIR.mkdir(parents=True, exist_ok=True)
    datasets = {
        "synthetic_short_1x1x1": build_short_dataset(),
        "synthetic_long_train_probe": build_long_probe_dataset(),
        "whisper_large_v3_levantine_1x1x1": ensure_levantine_dataset(),
    }
    materialized_full = maybe_register_materialized_full_dataset()
    if materialized_full is not None:
        datasets["omnilingual_300m_levantine_full_from_whisper_splits"] = materialized_full
    registry = {"datasets": datasets}
    REGISTRY_PATH.write_text(json.dumps(registry, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"registry_path": str(REGISTRY_PATH), "datasets": sorted(registry["datasets"].keys())}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
