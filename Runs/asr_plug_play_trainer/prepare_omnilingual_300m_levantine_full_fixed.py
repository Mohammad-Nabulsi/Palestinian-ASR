#!/usr/bin/env python3
"""Materialize the full non-smoke Omnilingual Levantine dataset for the plug-and-play trainer."""

from __future__ import annotations

import json
import re
import sys
from collections import Counter
from pathlib import Path
from typing import Any

import soundfile as sf


# File location expected:
# /home/MohammadNabulsi/whisper/Runs/asr_plug_play_trainer/prepare_omnilingual_300m_levantine_full.py
#
# parents[0] = /home/MohammadNabulsi/whisper/Runs/asr_plug_play_trainer
# parents[1] = /home/MohammadNabulsi/whisper/Runs
# parents[2] = /home/MohammadNabulsi/whisper
REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE_DIR = Path(__file__).resolve().parent

# Needed so `from Runs...` works when this script is launched from asr_plug_play_trainer.
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Runs.qwen3_asr_0_6b_levantine_custom_streaming_5minckpt.pipeline import (  # noqa: E402
    load_audio_for_row,
    make_config,
)


SOURCE_RUN_DIR = REPO_ROOT / "Runs" / "omnilingual_asr_1b_levantine_custom_streaming_5minckpt"
OUTPUT_DATASET_DIR = BUNDLE_DIR / "datasets" / "omnilingual_300m_levantine_full"
OUTPUT_AUDIO_DIR = OUTPUT_DATASET_DIR / "audio"

SOURCE_MANIFESTS = {
    "train": SOURCE_RUN_DIR / "manifests" / "train" / "manifest_train.jsonl",
    "validation": SOURCE_RUN_DIR / "manifests" / "val" / "manifest_val.jsonl",
    "test": SOURCE_RUN_DIR / "manifests" / "test" / "manifest_test.jsonl",
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
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    output_rows: list[dict[str, Any]] = []
    split_audio_dir = OUTPUT_AUDIO_DIR / split
    split_audio_dir.mkdir(parents=True, exist_ok=True)
    source_group_counts: Counter[str] = Counter()
    source_hours: Counter[str] = Counter()

    for idx, row in enumerate(source_rows):
        audio, sample_rate = load_audio_for_row(row, config)

        uid = str(row.get("uid") or f"{split}_{idx:06d}")
        audio_name = f"{idx:06d}_{safe_name(uid)}.flac"
        audio_path = split_audio_dir / audio_name

        sf.write(audio_path, audio, sample_rate, format="FLAC")

        duration = float(row.get("duration") or (len(audio) / float(sample_rate)))
        source_group = str(row.get("source_group") or row.get("source") or "unknown")
        source_group_counts[source_group] += 1
        source_hours[source_group] += duration / 3600.0

        output_rows.append(
            {
                "uid": uid,
                "audio_path": str(audio_path),
                "text": str(row.get("text") or row.get("transcript") or row.get("transcription") or "").strip(),
                "duration": duration,
                "split": split,
                "language": str(row.get("language") or "ar"),
                "speaker_id": str(row.get("speaker_id") or ""),
                "source": row.get("source"),
                "source_group": source_group,
                "metadata": {
                    "materialized_from_manifest": str(SOURCE_MANIFESTS[split]),
                    "materialized_from_uid": uid,
                    "audio_kind": row.get("audio_kind"),
                    "row_idx": row.get("row_idx"),
                    "parquet_file": row.get("parquet_file"),
                    "arrow_file": row.get("arrow_file"),
                    "source_root": row.get("source_root"),
                    "original_split": row.get("original_split"),
                },
            }
        )

        if (idx + 1) % 100 == 0:
            print(f"[{split}] materialized {idx + 1}/{len(source_rows)}", flush=True)

    summary = {
        "count": len(output_rows),
        "hours": sum(source_hours.values()),
        "source_group_counts": dict(source_group_counts),
        "source_group_hours": {key: round(value, 6) for key, value in source_hours.items()},
    }
    return output_rows, summary


def main() -> None:
    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    missing = [str(path) for path in SOURCE_MANIFESTS.values() if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing source manifests:\n" + "\n".join(missing))

    config = make_config(
        smoke_mode=False,
        num_train_epochs=1,
        run_baseline_before_train=False,
        run_post_train_eval=False,
    )

    source_rows = {split: read_jsonl(path) for split, path in SOURCE_MANIFESTS.items()}
    output_rows: dict[str, list[dict[str, Any]]] = {}
    split_summary: dict[str, dict[str, Any]] = {}

    for split, rows in source_rows.items():
        print(f"Materializing split={split} rows={len(rows)}", flush=True)
        rows_out, summary = materialize_split(rows, split, config)
        output_rows[split] = rows_out
        split_summary[split] = summary

    combined_rows: list[dict[str, Any]] = []
    for split in ["train", "validation", "test"]:
        split_path = OUTPUT_DATASET_DIR / f"{split}.jsonl"
        write_jsonl(split_path, output_rows[split])
        combined_rows.extend(output_rows[split])

    write_jsonl(OUTPUT_DATASET_DIR / "combined.jsonl", combined_rows)

    summary = {
        "dataset_name": "omnilingual_300m_levantine_full",
        "dataset_dir": str(OUTPUT_DATASET_DIR),
        "source_run_dir": str(SOURCE_RUN_DIR),
        "source_manifests": {split: str(path) for split, path in SOURCE_MANIFESTS.items()},
        "selected_counts": {split: len(rows) for split, rows in output_rows.items()},
        "selected_hours": {split: round(split_summary[split]["hours"], 6) for split in split_summary},
        "source_group_breakdown": split_summary,
        "output_files": {
            split: str(OUTPUT_DATASET_DIR / f"{split}.jsonl")
            for split in ["train", "validation", "test"]
        },
        "combined_path": str(OUTPUT_DATASET_DIR / "combined.jsonl"),
        "connected_data": {
            "train": [
                "1514 rows from casablanca_levantine_train",
                "345 rows from omnilingual_apc_north_levantine_train",
            ],
            "validation": [
                "757 rows from casablanca_levantine_eval_pool",
                "86 rows from omnilingual_apc_north_levantine_eval_pool",
            ],
            "test": [
                "757 rows from casablanca_levantine_eval_pool",
                "86 rows from omnilingual_apc_north_levantine_eval_pool",
            ],
        },
        "notes": [
            "This is a non-smoke full Levantine materialization for the OmniLingual 300M notebook copy.",
            "The plug-and-play trainer uses this dataset for prepare/export; actual OmniLingual fine-tuning remains an external recipe flow.",
        ],
    }

    (OUTPUT_DATASET_DIR / "dataset_summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
