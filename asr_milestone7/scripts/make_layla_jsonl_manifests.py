#!/usr/bin/env python3
"""Create milestone-7 JSONL manifests from three Layla source batches.

This converts the Layla normalized JSON batches into the JSONL schema expected
by ``asr_milestone7/asr_pipeline/data.py``:

- uid
- audio_path
- text
- duration
- sample_rate
- source

The current milestone-7 loader accepts JSONL manifests only, not Parquet.
"""

from __future__ import annotations

import argparse
import json
import wave
from pathlib import Path


SCRIPT_PATH = Path(__file__).resolve()
MILESTONE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_LAYLA_ROOT = REPO_ROOT / ".intermediate_data" / "Layla"
DEFAULT_AUDIO_ROOT = DEFAULT_LAYLA_ROOT / "Layla Witheeb Jordanian Arabic Acoustic Dataset"
DEFAULT_OUTPUT_DIR = MILESTONE_ROOT / "data" / "layla_manifests"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--layla-root",
        type=Path,
        default=DEFAULT_LAYLA_ROOT,
        help="Root directory containing the normalized Layla JSON batches.",
    )
    parser.add_argument(
        "--audio-root",
        type=Path,
        default=DEFAULT_AUDIO_ROOT,
        help="Root directory containing the Layla WAV files.",
    )
    parser.add_argument(
        "--train-json",
        default="normalized_output_appended.json",
        help="JSON batch file to use as the train split.",
    )
    parser.add_argument(
        "--val-json",
        default="normalized_layla_batch_130_appended_131.json",
        help="JSON batch file to use as the validation split.",
    )
    parser.add_argument(
        "--test-json",
        default="normalized_pasted_132_133_appended.json",
        help="JSON batch file to use as the test split.",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=DEFAULT_OUTPUT_DIR,
        help="Directory where train/val/test JSONL manifests will be written.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def derive_audio_path(audio_root: Path, source_value: str) -> Path:
    rel = source_value.replace("./", "")
    txt_path = audio_root / rel
    stem = txt_path.name.replace("_Arabic_transcription.txt", "")
    base = txt_path.with_name(stem)
    candidates = (base.with_suffix(".WAV"), base.with_suffix(".wav"))
    for candidate in candidates:
        if candidate.exists():
            return candidate.resolve()
    raise FileNotFoundError(f"No audio file found for source {source_value!r}")


def wav_stats(path: Path) -> tuple[float, int]:
    with wave.open(str(path), "rb") as handle:
        frame_count = handle.getnframes()
        sample_rate = handle.getframerate()
    duration = (frame_count / sample_rate) if sample_rate else 0.0
    return duration, sample_rate


def load_records(path: Path) -> list[dict]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    records = payload.get("records", [])
    if not isinstance(records, list):
        raise ValueError(f"'records' must be a list in {path}")
    return records


def rows_for_split(*, split: str, json_path: Path, audio_root: Path) -> list[dict]:
    rows: list[dict] = []
    for idx, record in enumerate(load_records(json_path)):
        normalized_text = str(record.get("normalized", "")).strip()
        source_value = str(record["source"])
        audio_path = derive_audio_path(audio_root, source_value)
        duration, sample_rate = wav_stats(audio_path)
        rows.append(
            {
                "uid": f"layla_{split}_{json_path.stem}_{idx:05d}",
                "audio_path": str(audio_path),
                "text": normalized_text,
                "duration": duration,
                "sample_rate": sample_rate,
                "source": f"layla:{json_path.name}",
            }
        )
    if not rows:
        raise ValueError(f"No rows produced for split {split!r} from {json_path}")
    return rows


def write_jsonl(path: Path, rows: list[dict]) -> None:
    ensure_dir(path.parent)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False, sort_keys=True))
            handle.write("\n")


def main() -> None:
    args = parse_args()
    layla_root = args.layla_root.resolve()
    audio_root = args.audio_root.resolve()
    output_dir = ensure_dir(args.output_dir.resolve())

    split_to_json = {
        "train": layla_root / args.train_json,
        "val": layla_root / args.val_json,
        "test": layla_root / args.test_json,
    }

    summary: dict[str, dict[str, str | int]] = {}
    for split, json_path in split_to_json.items():
        if not json_path.exists():
            raise FileNotFoundError(f"Split source JSON does not exist: {json_path}")
        rows = rows_for_split(split=split, json_path=json_path, audio_root=audio_root)
        out_path = output_dir / f"{split}.jsonl"
        write_jsonl(out_path, rows)
        summary[split] = {
            "json_source": str(json_path),
            "manifest_path": str(out_path),
            "row_count": len(rows),
        }
        print(f"{split}: wrote {len(rows)} rows -> {out_path}")

    summary_path = output_dir / "summary.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(f"summary: {summary_path}")


if __name__ == "__main__":
    main()
