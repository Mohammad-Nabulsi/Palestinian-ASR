#!/usr/bin/env python3
"""Materialize a 1/1/1 Whisper large-v3 Levantine dataset for the plug-and-play trainer.

This script deliberately reuses the already-prepared Whisper large-v3 Levantine manifests
so we keep the same dataset selection and filtering without rebuilding or overwriting the
original run directory.
"""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

import soundfile as sf

from Runs.whisper_large_v3_levantine_custom_streaming_5minckpt.pipeline import (
    load_audio_for_row,
    make_config,
)


REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = Path(__file__).resolve().parent
SOURCE_RUN_DIR = REPO_ROOT / "Runs" / "whisper_large_v3_levantine_custom_streaming_5minckpt"
SOURCE_MANIFEST_DIR = SOURCE_RUN_DIR / "manifests"
OUTPUT_DATASET_DIR = BUNDLE_DIR / "datasets" / "whisper_large_v3_levantine_1x1x1"
OUTPUT_AUDIO_DIR = OUTPUT_DATASET_DIR / "audio"

SOURCE_MANIFESTS = {
    "train": SOURCE_MANIFEST_DIR / "train" / "manifest_train_custom_levantine_lt30s.jsonl",
    "validation": SOURCE_MANIFEST_DIR / "val" / "manifest_val_custom_levantine_lt30s.jsonl",
    "test": SOURCE_MANIFEST_DIR / "test" / "manifest_test_custom_levantine_lt30s.jsonl",
}


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return cleaned.strip("._") or "sample"


def materialize_split(
    source_rows: list[dict[str, Any]],
    split: str,
    config: Any,
) -> list[dict[str, Any]]:
    rows = source_rows[:1]
    if not rows:
        raise RuntimeError(f"Source manifest for split={split} is empty.")

    output_rows: list[dict[str, Any]] = []
    split_audio_dir = OUTPUT_AUDIO_DIR / split
    split_audio_dir.mkdir(parents=True, exist_ok=True)

    for idx, row in enumerate(rows):
        audio, sample_rate = load_audio_for_row(row, config)
        uid = str(row.get("uid") or f"{split}_{idx:03d}")
        wav_path = split_audio_dir / f"{idx:03d}_{safe_name(uid)}.wav"
        sf.write(wav_path, audio, sample_rate)
        duration = float(row.get("duration") or (len(audio) / float(sample_rate)))
        output_rows.append(
            {
                "uid": uid,
                "audio_path": str(wav_path),
                "text": str(row.get("text") or "").strip(),
                "duration": duration,
                "split": split,
                "language": str(row.get("language") or "ar"),
                "speaker_id": str(row.get("speaker_id") or ""),
                "source": row.get("source"),
                "source_group": row.get("source_group"),
                "metadata": {
                    "materialized_from_manifest": str(SOURCE_MANIFESTS[split]),
                    "materialized_from_uid": uid,
                    "audio_kind": row.get("audio_kind"),
                    "row_idx": row.get("row_idx"),
                    "parquet_file": row.get("parquet_file"),
                    "arrow_file": row.get("arrow_file"),
                },
            }
        )
    return output_rows


def main() -> None:
    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)
    config = make_config(smoke_mode=False, num_train_epochs=1)

    source_rows = {split: read_jsonl(path) for split, path in SOURCE_MANIFESTS.items()}
    output_rows = {
        split: materialize_split(rows, split, config)
        for split, rows in source_rows.items()
    }

    combined_rows: list[dict[str, Any]] = []
    for split in ["train", "validation", "test"]:
        split_path = OUTPUT_DATASET_DIR / f"{split}.jsonl"
        write_jsonl(split_path, output_rows[split])
        combined_rows.extend(output_rows[split])

    write_jsonl(OUTPUT_DATASET_DIR / "combined.jsonl", combined_rows)

    summary = {
        "dataset_dir": str(OUTPUT_DATASET_DIR),
        "source_run_dir": str(SOURCE_RUN_DIR),
        "source_manifests": {split: str(path) for split, path in SOURCE_MANIFESTS.items()},
        "selected_counts": {split: len(rows) for split, rows in output_rows.items()},
        "selected_uids": {split: [row["uid"] for row in rows] for split, rows in output_rows.items()},
        "output_files": {
            split: str(OUTPUT_DATASET_DIR / f"{split}.jsonl")
            for split in ["train", "validation", "test"]
        },
    }
    (OUTPUT_DATASET_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
