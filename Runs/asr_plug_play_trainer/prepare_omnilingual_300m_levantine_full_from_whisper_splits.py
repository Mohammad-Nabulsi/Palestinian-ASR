#!/usr/bin/env python3
"""Materialize OmniLingual 300M Levantine full data from the Whisper pipeline split logic.

This script rebuilds the train/validation/test split from the raw Casablanca parquet
and Omnilingual Arrow files used by the Whisper Medium Levantine pipeline.

Important difference from the Whisper script:
- This does NOT drop clips >= 30 seconds.
- It only drops empty-text rows and rows below min_audio_seconds.
"""

from __future__ import annotations

import hashlib
import io
import json
import os
import re
import shutil
import sys
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import numpy as np
import soundfile as sf


REPO_ROOT = Path("/home/MohammadNabulsi/whisper")
BUNDLE_DIR = Path(__file__).resolve().parent

DATASET_NAME = "omnilingual_300m_levantine_full_from_whisper_splits"
OUTPUT_DATASET_DIR = BUNDLE_DIR / "datasets" / DATASET_NAME
OUTPUT_AUDIO_DIR = OUTPUT_DATASET_DIR / "audio"

SAMPLE_RATE = 16_000
MIN_AUDIO_SECONDS = 0.3
SPLIT_SEED = 42
FORCE_REBUILD = os.environ.get("FORCE_REBUILD_DATASET", "").strip().lower() in {"1", "true", "yes", "y"}

EXPECTED_OUTPUT_FILES = (
    OUTPUT_DATASET_DIR / "dataset_summary.json",
    OUTPUT_DATASET_DIR / "train.jsonl",
    OUTPUT_DATASET_DIR / "validation.jsonl",
    OUTPUT_DATASET_DIR / "test.jsonl",
)

TRAIN_PARQUET_FILES = (
    REPO_ROOT / "casablanca" / "levant" / "Palestine" / "validation-00001-of-00002.parquet",
    REPO_ROOT / "casablanca" / "levant" / "Palestine" / "validation-00000-of-00002.parquet",
    REPO_ROOT / "casablanca" / "levant" / "Jordan" / "validation-00000-of-00001.parquet",
)

EVAL_PARQUET_FILES = (
    REPO_ROOT / "casablanca" / "levant" / "Palestine" / "test-00001-of-00002.parquet",
    REPO_ROOT / "casablanca" / "levant" / "Palestine" / "test-00000-of-00002.parquet",
    REPO_ROOT / "casablanca" / "levant" / "Jordan" / "test-00000-of-00001.parquet",
)

TRAIN_ARROW_FILES = (
    REPO_ROOT / "omnilingual_selected" / "apc_north_levantine_all_splits" / "data-00001-of-00003.arrow",
    REPO_ROOT / "omnilingual_selected" / "apc_north_levantine_all_splits" / "data-00000-of-00003.arrow",
)

EVAL_ARROW_FILES = (
    REPO_ROOT / "omnilingual_selected" / "apc_north_levantine_all_splits" / "data-00002-of-00003.arrow",
)


AR_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
CONTROL_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
SPACE_RE = re.compile(r"\s+")
ASR_TAG_RE = re.compile(r"<[^>]+>|\[[^\]]+\]")
AR_PUNCT_SPACING_RE = re.compile(r"\s*([،؛؟,.!?;:])\s*")
REPEATED_PUNCT_RE = re.compile(r"([،؛؟,.!?;:])\1+")


def normalize_arabic_text(text: Any) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = CONTROL_RE.sub(" ", text)
    text = ASR_TAG_RE.sub(" ", text)
    text = text.replace("ـ", "")
    text = AR_DIACRITICS_RE.sub("", text)
    text = AR_PUNCT_SPACING_RE.sub(r" \1 ", text)
    text = REPEATED_PUNCT_RE.sub(r"\1", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def safe_name(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", value)
    return cleaned.strip("._") or "sample"


def stable_hash(text: str, seed: int) -> int:
    payload = f"{seed}|{text}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16)


def row_uid(prefix: str, path: Path, row_idx: int) -> str:
    digest = hashlib.sha1(f"{prefix}:{path}:{row_idx}".encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:16]}"


def stable_row_order(rows: list[dict[str, Any]], *, seed: int, salt: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: stable_hash(f"{salt}|{row.get('uid')}", seed))


def split_rows_half(rows: list[dict[str, Any]], *, seed: int, salt: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = stable_row_order(rows, seed=seed, salt=salt)
    midpoint = len(ordered) // 2
    val_rows = [dict(row, split="validation") for row in ordered[:midpoint]]
    test_rows = [dict(row, split="test") for row in ordered[midpoint:]]
    return val_rows, test_rows


def ensure_audio_tools():
    try:
        import librosa
    except ImportError:
        librosa = None
    try:
        import pyarrow as pa
        import pyarrow.ipc as pa_ipc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for parquet/arrow reading.") from exc
    return librosa, pa, pa_ipc, pq


def to_mono_float32(audio: Any) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[0] < arr.shape[1]:
            return arr.mean(axis=0).astype(np.float32)
        return arr.mean(axis=1).astype(np.float32)
    return arr.reshape(-1).astype(np.float32)


def read_audio_bytes(payload: bytes) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
    return to_mono_float32(audio), int(sample_rate)


def read_audio_path(path: str) -> tuple[np.ndarray, int]:
    audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
    return to_mono_float32(audio), int(sample_rate)


def resample_audio(audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if int(sample_rate) == int(target_sample_rate):
        return to_mono_float32(audio)
    librosa, *_ = ensure_audio_tools()
    if librosa is None:
        raise RuntimeError("librosa is required when an input sample rate is not already 16 kHz.")
    return librosa.resample(
        to_mono_float32(audio),
        orig_sr=int(sample_rate),
        target_sr=int(target_sample_rate),
    ).astype(np.float32)


def read_parquet_row(path: str, row_idx: int) -> dict[str, Any]:
    *_, pq = ensure_audio_tools()
    parquet_file = pq.ParquetFile(path)
    seen = 0
    for batch in parquet_file.iter_batches(batch_size=1024):
        py_rows = batch.to_pylist()
        if seen + len(py_rows) > row_idx:
            return py_rows[row_idx - seen]
        seen += len(py_rows)
    raise IndexError(f"row_idx={row_idx} out of range for {path}")


def read_arrow_row(path: str, row_idx: int, columns: list[str] | None = None) -> dict[str, Any]:
    _, pa, pa_ipc, _ = ensure_audio_tools()
    index = int(row_idx)
    seen = 0
    with pa.memory_map(path, "r") as source:
        reader = pa_ipc.open_stream(source)
        for batch in reader:
            if seen + batch.num_rows <= index:
                seen += batch.num_rows
                continue
            local = index - seen
            names = set(batch.schema.names)
            use_cols = [column for column in (columns or batch.schema.names) if column in names]
            return {
                column: batch.column(batch.schema.get_field_index(column)).to_pylist()[local]
                for column in use_cols
            }
    raise IndexError(f"row_idx={row_idx} out of range for {path}")


def load_parquet_audio(row: dict[str, Any]) -> tuple[np.ndarray, int]:
    example = read_parquet_row(str(row["parquet_file"]), int(row["row_idx"]))
    audio = example.get("audio") or {}
    if isinstance(audio, dict):
        if audio.get("bytes") is not None:
            return read_audio_bytes(audio["bytes"])
        if audio.get("path"):
            candidate = str(audio["path"])
            if not Path(candidate).is_absolute():
                candidate = str(Path(str(row["parquet_file"])).parent / candidate)
            return read_audio_path(candidate)
    raise RuntimeError(f"Unsupported Parquet audio payload for uid={row.get('uid')}")


def load_arrow_audio(row: dict[str, Any]) -> tuple[np.ndarray, int]:
    example = read_arrow_row(str(row["arrow_file"]), int(row["row_idx"]), columns=["audio"])
    audio = example.get("audio")
    if isinstance(audio, dict):
        if audio.get("array") is not None:
            sample_rate = audio.get("sampling_rate") or SAMPLE_RATE
            return to_mono_float32(audio["array"]), int(sample_rate)
        if audio.get("bytes") is not None:
            return read_audio_bytes(audio["bytes"])
        if audio.get("path"):
            candidate = str(audio["path"])
            if not Path(candidate).is_absolute():
                candidate = str(Path(str(row["arrow_file"])).parent / candidate)
            return read_audio_path(candidate)
    raise RuntimeError(f"Unsupported Arrow audio payload for uid={row.get('uid')}")


def load_audio_for_manifest_row(row: dict[str, Any]) -> tuple[np.ndarray, int]:
    if row.get("audio_kind") == "hf_audio":
        audio, sample_rate = load_arrow_audio(row)
    elif row.get("audio_kind") == "bytes":
        audio, sample_rate = load_parquet_audio(row)
    elif row.get("audio_kind") == "path" and row.get("audio_path"):
        audio, sample_rate = read_audio_path(str(row["audio_path"]))
    else:
        raise RuntimeError(f"Unsupported audio_kind={row.get('audio_kind')} uid={row.get('uid')}")
    return resample_audio(audio, int(sample_rate), SAMPLE_RATE), SAMPLE_RATE


def build_parquet_rows(paths: tuple[Path, ...], *, original_split: str, source_group: str, source_label: str) -> list[dict[str, Any]]:
    *_, pq = ensure_audio_tools()
    rows: list[dict[str, Any]] = []
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        row_idx = 0
        for batch in parquet_file.iter_batches(
            batch_size=1024,
            columns=["seg_id", "transcription", "gender", "duration"],
        ):
            for item in batch.to_pylist():
                rows.append(
                    {
                        "uid": row_uid("parquet", path, row_idx),
                        "source": "casablanca",
                        "source_group": source_group,
                        "source_root": str(path.parent),
                        "original_split": original_split,
                        "parquet_file": str(path),
                        "arrow_file": None,
                        "row_idx": row_idx,
                        "audio_kind": "bytes",
                        "text": item.get("transcription", ""),
                        "duration": item.get("duration"),
                        "segment_id": item.get("seg_id"),
                        "speaker_id": None,
                        "gender": item.get("gender"),
                        "language": "apc_Arab",
                        "metadata": {
                            "source_label": source_label,
                            "country": path.parent.name,
                            "filename": path.name,
                        },
                    }
                )
                row_idx += 1
    return rows


def build_arrow_rows(paths: tuple[Path, ...], *, original_split: str, source_group: str) -> list[dict[str, Any]]:
    _, pa, pa_ipc, _ = ensure_audio_tools()
    rows: list[dict[str, Any]] = []
    columns = [
        "language",
        "speaker_id",
        "prompt_id",
        "prompt",
        "segment_id",
        "duration",
        "raw_text",
        "iso_639_3",
        "glottocode",
        "iso_15924",
        "config",
        "original_split",
    ]
    for path in paths:
        row_idx = 0
        with pa.memory_map(str(path), "r") as source:
            reader = pa_ipc.open_stream(source)
            for batch in reader:
                names = set(batch.schema.names)
                use_cols = [column for column in columns if column in names]
                payload = {
                    column: batch.column(batch.schema.get_field_index(column)).to_pylist()
                    for column in use_cols
                }
                for offset in range(batch.num_rows):
                    def value(column: str, default: Any = None) -> Any:
                        return payload.get(column, [default] * batch.num_rows)[offset]

                    metadata = {
                        "config": value("config"),
                        "glottocode": value("glottocode"),
                        "iso_639_3": value("iso_639_3"),
                        "iso_15924": value("iso_15924"),
                        "prompt": value("prompt"),
                        "prompt_id": value("prompt_id"),
                        "filename": path.name,
                    }
                    rows.append(
                        {
                            "uid": row_uid("arrow", path, row_idx),
                            "source": "omnilingual",
                            "source_group": source_group,
                            "source_root": str(path.parent),
                            "original_split": value("original_split", original_split) or original_split,
                            "parquet_file": None,
                            "arrow_file": str(path),
                            "row_idx": row_idx,
                            "audio_kind": "hf_audio",
                            "text": value("raw_text", ""),
                            "duration": value("duration"),
                            "segment_id": value("segment_id"),
                            "speaker_id": value("speaker_id"),
                            "gender": None,
                            "language": value("language", "apc_Arab"),
                            "metadata": metadata,
                        }
                    )
                    row_idx += 1
    return rows


def filter_rows(rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], dict[str, int]]:
    kept: list[dict[str, Any]] = []
    stats = {"input": len(rows), "kept": 0, "dropped_empty": 0, "dropped_too_short": 0, "dropped_bad_duration": 0}
    for row in rows:
        text = normalize_arabic_text(row.get("text", ""))
        if not text:
            stats["dropped_empty"] += 1
            continue
        try:
            duration = float(row.get("duration"))
        except Exception:
            stats["dropped_bad_duration"] += 1
            continue
        if duration < MIN_AUDIO_SECONDS:
            stats["dropped_too_short"] += 1
            continue

        clean = dict(row)
        clean["text"] = text
        clean["duration"] = duration
        clean["split"] = split
        kept.append(clean)

    stats["kept"] = len(kept)
    return kept, stats


def rows_by_source_group(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    for group in sorted({str(row.get("source_group")) for row in rows}):
        group_rows = [row for row in rows if str(row.get("source_group")) == group]
        out[group] = {
            "rows": len(group_rows),
            "hours": sum(float(row.get("duration") or 0.0) for row in group_rows) / 3600.0,
        }
    return out


def materialize_rows(rows: list[dict[str, Any]], split: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    split_audio_dir = OUTPUT_AUDIO_DIR / split
    split_audio_dir.mkdir(parents=True, exist_ok=True)

    output_rows: list[dict[str, Any]] = []
    source_group_counts: Counter[str] = Counter()
    source_hours: Counter[str] = Counter()

    for idx, row in enumerate(rows):
        audio, sample_rate = load_audio_for_manifest_row(row)
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
                "text": str(row.get("text") or "").strip(),
                "duration": duration,
                "split": split,
                "language": str(row.get("language") or "ar"),
                "speaker_id": str(row.get("speaker_id") or ""),
                "source": row.get("source"),
                "source_group": source_group,
                "segment_id": row.get("segment_id"),
                "gender": row.get("gender"),
                "metadata": {
                    **(row.get("metadata") or {}),
                    "materialized_from": {
                        "parquet_file": row.get("parquet_file"),
                        "arrow_file": row.get("arrow_file"),
                        "row_idx": row.get("row_idx"),
                        "source_root": row.get("source_root"),
                        "original_split": row.get("original_split"),
                        "audio_kind": row.get("audio_kind"),
                    },
                },
            }
        )

        if (idx + 1) % 100 == 0:
            print(f"[{split}] materialized {idx + 1}/{len(rows)}", flush=True)

    summary = {
        "count": len(output_rows),
        "hours": sum(source_hours.values()),
        "source_group_counts": dict(source_group_counts),
        "source_group_hours": {key: round(value, 6) for key, value in source_hours.items()},
    }
    return output_rows, summary



def dataset_outputs_ready() -> bool:
    return all(path.exists() for path in EXPECTED_OUTPUT_FILES)


def check_inputs() -> None:
    all_paths = list(TRAIN_PARQUET_FILES) + list(EVAL_PARQUET_FILES) + list(TRAIN_ARROW_FILES) + list(EVAL_ARROW_FILES)
    missing = [str(path) for path in all_paths if not path.exists()]
    if missing:
        raise FileNotFoundError("Missing input files:\n" + "\n".join(missing))


def main() -> None:
    check_inputs()

    outputs_ready = dataset_outputs_ready()
    if outputs_ready and not FORCE_REBUILD:
        print(
            f"Dataset already materialized at {OUTPUT_DATASET_DIR}; skipping rebuild. "
            "Set FORCE_REBUILD_DATASET=1 to rebuild.",
            flush=True,
        )
        return

    if OUTPUT_DATASET_DIR.exists() and (FORCE_REBUILD or not outputs_ready):
        reason = "force rebuild requested" if FORCE_REBUILD else "existing outputs are incomplete"
        print(f"Removing previous output dataset ({reason}): {OUTPUT_DATASET_DIR}", flush=True)
        shutil.rmtree(OUTPUT_DATASET_DIR)

    OUTPUT_DATASET_DIR.mkdir(parents=True, exist_ok=True)

    print("Building raw row references from Whisper split sources...", flush=True)

    train_rows_raw = build_parquet_rows(
        TRAIN_PARQUET_FILES,
        original_split="validation",
        source_group="casablanca_levantine_train",
        source_label="casablanca_levantine_validation_as_train",
    ) + build_arrow_rows(
        TRAIN_ARROW_FILES,
        original_split="train",
        source_group="omnilingual_apc_north_levantine_train",
    )

    eval_parquet_raw = build_parquet_rows(
        EVAL_PARQUET_FILES,
        original_split="test",
        source_group="casablanca_levantine_eval_pool",
        source_label="casablanca_levantine_test_split_50_50",
    )

    eval_arrow_raw = build_arrow_rows(
        EVAL_ARROW_FILES,
        original_split="train",
        source_group="omnilingual_apc_north_levantine_eval_pool",
    )

    filtered_train, train_filter_stats = filter_rows(train_rows_raw, "train")
    filtered_eval_parquet, eval_parquet_filter_stats = filter_rows(eval_parquet_raw, "eval_pool")
    filtered_eval_arrow, eval_arrow_filter_stats = filter_rows(eval_arrow_raw, "eval_pool")

    parquet_val_rows, parquet_test_rows = split_rows_half(
        filtered_eval_parquet,
        seed=SPLIT_SEED,
        salt="custom-casablanca-heldout",
    )
    arrow_val_rows, arrow_test_rows = split_rows_half(
        filtered_eval_arrow,
        seed=SPLIT_SEED,
        salt="custom-omnilingual-heldout",
    )

    train_rows = [dict(row, split="train") for row in stable_row_order(filtered_train, seed=SPLIT_SEED, salt="custom-train")]
    validation_rows = stable_row_order(parquet_val_rows + arrow_val_rows, seed=SPLIT_SEED, salt="custom-val")
    test_rows = stable_row_order(parquet_test_rows + arrow_test_rows, seed=SPLIT_SEED, salt="custom-test")

    print(f"Selected rows: train={len(train_rows)} validation={len(validation_rows)} test={len(test_rows)}", flush=True)
    print("Materializing audio to FLAC...", flush=True)

    output_rows: dict[str, list[dict[str, Any]]] = {}
    split_summary: dict[str, dict[str, Any]] = {}

    for split, rows in [("train", train_rows), ("validation", validation_rows), ("test", test_rows)]:
        rows_out, summary = materialize_rows(rows, split)
        output_rows[split] = rows_out
        split_summary[split] = summary
        jsonl_write(OUTPUT_DATASET_DIR / f"{split}.jsonl", rows_out)

    combined_rows = output_rows["train"] + output_rows["validation"] + output_rows["test"]
    jsonl_write(OUTPUT_DATASET_DIR / "combined.jsonl", combined_rows)

    selection_summary = {
        "dataset_name": DATASET_NAME,
        "dataset_dir": str(OUTPUT_DATASET_DIR),
        "sample_rate": SAMPLE_RATE,
        "min_audio_seconds": MIN_AUDIO_SECONDS,
        "drop_audio_at_or_above_seconds": None,
        "note_on_long_audio": "No >=30s rule is applied here. Long clips are kept during materialization.",
        "raw_counts": {
            "train_rows_raw": len(train_rows_raw),
            "eval_parquet_raw": len(eval_parquet_raw),
            "eval_arrow_raw": len(eval_arrow_raw),
        },
        "filter_stats": {
            "train": train_filter_stats,
            "eval_parquet_pool": eval_parquet_filter_stats,
            "eval_arrow_pool": eval_arrow_filter_stats,
        },
        "selected_counts": {split: len(rows) for split, rows in output_rows.items()},
        "selected_hours": {
            split: round(split_summary[split]["hours"], 6)
            for split in ["train", "validation", "test"]
        },
        "source_group_breakdown": split_summary,
        "train_by_source_group": rows_by_source_group(train_rows),
        "validation_by_source_group": rows_by_source_group(validation_rows),
        "test_by_source_group": rows_by_source_group(test_rows),
        "data_sources": {
            "train_parquet_files": [str(path) for path in TRAIN_PARQUET_FILES],
            "eval_parquet_files": [str(path) for path in EVAL_PARQUET_FILES],
            "train_arrow_files": [str(path) for path in TRAIN_ARROW_FILES],
            "eval_arrow_files": [str(path) for path in EVAL_ARROW_FILES],
        },
        "output_files": {
            "train": str(OUTPUT_DATASET_DIR / "train.jsonl"),
            "validation": str(OUTPUT_DATASET_DIR / "validation.jsonl"),
            "test": str(OUTPUT_DATASET_DIR / "test.jsonl"),
            "combined": str(OUTPUT_DATASET_DIR / "combined.jsonl"),
        },
        "split_logic": {
            "train": [
                "Casablanca validation Palestine/Jordan parquet files are used as train.",
                "Omnilingual APC Arrow shards data-00001 and data-00000 are used as train.",
            ],
            "validation": [
                "Half of Casablanca test Palestine/Jordan parquet pool using stable hash split.",
                "Half of Omnilingual APC Arrow data-00002 using stable hash split.",
            ],
            "test": [
                "Other half of Casablanca test Palestine/Jordan parquet pool using stable hash split.",
                "Other half of Omnilingual APC Arrow data-00002 using stable hash split.",
            ],
        },
    }

    summary_path = OUTPUT_DATASET_DIR / "dataset_summary.json"
    summary_path.write_text(json.dumps(selection_summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(json.dumps(selection_summary, ensure_ascii=False, indent=2), flush=True)


if __name__ == "__main__":
    main()
