#!/usr/bin/env python3
"""Materialize an OmniLingual recipe dataset from the curated binary Levant set.

This script builds a real OmniLingual mixture-parquet dataset from
`data_curated_levant_binary_v1`, excluding every `qasr` subtree by default.

Outputs:
- run_root/dataset/version=0/...           OmniLingual parquet partitions
- run_root/language_distribution_0.tsv     corpus-language weighting summary
- run_root/prepare_summary.json            materialization summary
- run_root/configs/omni_300m_train.yaml    generated recipe config
- Omni dataset asset card under the local omnilingual-asr repo
"""

from __future__ import annotations

import argparse
import csv
import io
import json
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import librosa
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import yaml


SCRIPT_PATH = Path(__file__).resolve()
MILESTONE_ROOT = SCRIPT_PATH.parents[1]
REPO_ROOT = SCRIPT_PATH.parents[2]
DEFAULT_DATASET_ROOT = REPO_ROOT / "data_curated_levant_binary_v1"
DEFAULT_RUN_ROOT = MILESTONE_ROOT / "runs" / "omni_300m_levant_binary_no_qasr"
DEFAULT_OMNI_REPO = REPO_ROOT / ".omni_lingual_guide" / "omnilingual-asr"
DEFAULT_SAMPLE_RATE = 16_000
LANG_LEVANTINE = "apc_Arab"
LANG_NON_LEV = "arb_Arab"
TEXT_KEYS = (
    "manual_normalized_transcript",
    "transcription",
    "raw_text",
    "text",
)
WHITESPACE_RE = re.compile(r"\s+")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--run-root", type=Path, default=DEFAULT_RUN_ROOT)
    parser.add_argument("--omni-repo", type=Path, default=DEFAULT_OMNI_REPO)
    parser.add_argument("--dataset-card-name", default="levant_binary_no_qasr_omni_300m_dataset")
    parser.add_argument("--config-name", default="omni_300m_levant_binary_no_qasr")
    parser.add_argument("--sample-rate", type=int, default=DEFAULT_SAMPLE_RATE)
    parser.add_argument("--model-name", default="omniASR_LLM_300M_v2")
    parser.add_argument("--tokenizer-name", default="omniASR_tokenizer_written_v2")
    parser.add_argument("--train-steps", type=int, default=5_000)
    parser.add_argument("--validate-every", type=int, default=250)
    parser.add_argument("--checkpoint-every", type=int, default=250)
    parser.add_argument("--publish-metrics-every", type=int, default=25)
    parser.add_argument("--learning-rate", type=float, default=5e-5)
    parser.add_argument("--max-audio-seconds", type=float, default=75.0)
    parser.add_argument("--min-audio-seconds", type=float, default=0.05)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--max-num-elements", type=int, default=9_600_000)
    parser.add_argument("--num-seqs-multiple-of", type=int, default=1)
    parser.add_argument("--grad-accumulation", type=int, default=1)
    parser.add_argument("--beta-corpus", type=float, default=0.5)
    parser.add_argument("--beta-language", type=float, default=0.5)
    parser.add_argument("--example-shuffle-window", type=int, default=10_000)
    parser.add_argument("--batch-shuffle-window", type=int, default=64)
    parser.add_argument("--max-rows-per-split", type=int, default=None)
    parser.add_argument("--force-rebuild", action="store_true")
    return parser.parse_args()


def ensure_dir(path: Path) -> Path:
    path.mkdir(parents=True, exist_ok=True)
    return path


def normalize_text(value: Any) -> str:
    text = str(value or "").strip()
    return WHITESPACE_RE.sub(" ", text)


def choose_text(row: dict[str, Any]) -> str:
    for key in TEXT_KEYS:
        text = normalize_text(row.get(key))
        if text:
            return text
    return ""


def source_descriptor(parquet_path: Path, dataset_root: Path) -> tuple[str, str]:
    rel = parquet_path.relative_to(dataset_root)
    split = rel.parts[0]
    if rel.parts[1] == "layla":
        return split, "layla"
    if rel.parts[1] == "omni":
        return split, "omni"
    if rel.parts[1] == "casa":
        return split, f"casa_{rel.parts[2]}"
    if rel.parts[1] == "masc":
        return split, f"masc_{rel.parts[2]}"
    raise ValueError(f"Unsupported source path layout: {parquet_path}")


def language_for_source(parquet_path: Path) -> str:
    rel = parquet_path.parts
    if "masc" in rel and "non_lev" in rel:
        return LANG_NON_LEV
    return LANG_LEVANTINE


def included_parquet_paths(dataset_root: Path) -> dict[str, list[Path]]:
    result: dict[str, list[Path]] = {}
    for split in ("train", "val", "test"):
        split_root = dataset_root / split
        paths = [
            path
            for path in sorted(split_root.rglob("*.parquet"))
            if "qasr" not in {part.lower() for part in path.parts}
        ]
        result[split] = paths
    return result


def to_mono_float32(audio: Any) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        axis = 1 if arr.shape[0] > arr.shape[1] else 0
        return arr.mean(axis=axis).astype(np.float32)
    return arr.reshape(-1).astype(np.float32)


def resolve_audio_path(raw_path: str, parquet_path: Path) -> Path:
    path = Path(raw_path)
    if path.is_absolute():
        return path
    candidate = (parquet_path.parent / path).resolve()
    if candidate.exists():
        return candidate
    candidate = (REPO_ROOT / path).resolve()
    if candidate.exists():
        return candidate
    raise FileNotFoundError(f"Could not resolve audio path {raw_path!r} from {parquet_path}")


def load_audio_payload(audio_obj: Any, parquet_path: Path, target_sample_rate: int) -> tuple[np.ndarray, int]:
    if not isinstance(audio_obj, dict):
        raise TypeError(f"Expected dict audio payload in {parquet_path}, got {type(audio_obj).__name__}")

    if audio_obj.get("array") is not None:
        audio = to_mono_float32(audio_obj["array"])
        sample_rate = int(audio_obj.get("sampling_rate") or target_sample_rate)
    elif audio_obj.get("bytes") is not None:
        audio, sample_rate = sf.read(io.BytesIO(audio_obj["bytes"]), dtype="float32", always_2d=False)
        audio = to_mono_float32(audio)
        sample_rate = int(sample_rate)
    elif audio_obj.get("path"):
        audio_path = resolve_audio_path(str(audio_obj["path"]), parquet_path)
        audio, sample_rate = sf.read(str(audio_path), dtype="float32", always_2d=False)
        audio = to_mono_float32(audio)
        sample_rate = int(sample_rate)
    else:
        raise RuntimeError(f"Unsupported audio payload in {parquet_path}: keys={sorted(audio_obj.keys())}")

    if sample_rate != int(target_sample_rate):
        audio = librosa.resample(audio, orig_sr=sample_rate, target_sr=int(target_sample_rate)).astype(np.float32)
        sample_rate = int(target_sample_rate)

    return to_mono_float32(audio), sample_rate


def encode_flac_bytes(audio: np.ndarray, sample_rate: int) -> bytes:
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="FLAC")
    return buffer.getvalue()


def write_output_part(
    output_path: Path,
    rows: list[dict[str, Any]],
) -> None:
    table = pa.Table.from_pydict(
        {
            "uid": [row["uid"] for row in rows],
            "audio_bytes": [row["audio_bytes"] for row in rows],
            "audio_size": [row["audio_size"] for row in rows],
            "text": [row["text"] for row in rows],
        }
    )
    pq.write_table(table, output_path, row_group_size=100)


def build_dataset(args: argparse.Namespace) -> dict[str, Any]:
    dataset_root = args.dataset_root.resolve()
    run_root = args.run_root.resolve()
    omni_repo = args.omni_repo.resolve()
    dataset_dir = run_root / "dataset" / "version=0"
    summary_tsv = run_root / "language_distribution_0.tsv"
    config_dir = ensure_dir(run_root / "configs")
    cards_dataset_dir = ensure_dir(omni_repo / "src" / "omnilingual_asr" / "cards" / "datasets")
    config_path = config_dir / "omni_300m_train.yaml"
    summary_path = run_root / "prepare_summary.json"

    if args.force_rebuild and run_root.exists():
        import shutil
        shutil.rmtree(run_root)
        ensure_dir(config_dir)

    if summary_path.exists() and config_path.exists() and not args.force_rebuild:
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        payload["skipped_rebuild"] = True
        return payload

    ensure_dir(dataset_dir)
    parquet_paths = included_parquet_paths(dataset_root)

    part_counters: Counter[tuple[str, str, str]] = Counter()
    counts_by_split: Counter[str] = Counter()
    counts_by_corpus: Counter[str] = Counter()
    counts_by_language: Counter[str] = Counter()
    train_hours_by_partition: defaultdict[tuple[str, str], float] = defaultdict(float)
    skip_counts: Counter[str] = Counter()
    included_sources: list[str] = []

    for split in ("train", "val", "test"):
        split_rows_written = 0
        for parquet_path in parquet_paths[split]:
            included_sources.append(str(parquet_path))
            _, corpus = source_descriptor(parquet_path, dataset_root)
            language = language_for_source(parquet_path)
            part_index = part_counters[(split, corpus, language)]
            out_dir = ensure_dir(dataset_dir / f"corpus={corpus}" / f"split={split}" / f"language={language}")
            out_path = out_dir / f"data-{part_index:05d}.parquet"
            part_counters[(split, corpus, language)] += 1

            parquet_file = pq.ParquetFile(parquet_path)
            output_rows: list[dict[str, Any]] = []

            for row_group_idx in range(parquet_file.num_row_groups):
                table = parquet_file.read_row_group(row_group_idx)
                for row_idx, row in enumerate(table.to_pylist()):
                    if args.max_rows_per_split is not None and split_rows_written >= args.max_rows_per_split:
                        break
                    text = choose_text(row)
                    if not text:
                        skip_counts["empty_text"] += 1
                        continue
                    if row.get("flag_audio_too_short"):
                        skip_counts["flag_audio_too_short"] += 1
                        continue
                    if row.get("flag_missing_duration"):
                        skip_counts["flag_missing_duration"] += 1
                        continue

                    audio, sample_rate = load_audio_payload(row.get("audio"), parquet_path, args.sample_rate)
                    duration_seconds = len(audio) / float(sample_rate or args.sample_rate)
                    if duration_seconds < float(args.min_audio_seconds):
                        skip_counts["below_min_audio_seconds"] += 1
                        continue

                    output_rows.append(
                        {
                            "uid": f"{corpus}_{split}_{part_index:05d}_{row_group_idx:04d}_{row_idx:04d}",
                            "audio_bytes": encode_flac_bytes(audio, sample_rate),
                            "audio_size": int(len(audio)),
                            "text": text,
                        }
                    )
                    split_rows_written += 1
                    counts_by_split[split] += 1
                    counts_by_corpus[corpus] += 1
                    counts_by_language[language] += 1
                    if split == "train":
                        train_hours_by_partition[(corpus, language)] += duration_seconds / 3600.0

                if args.max_rows_per_split is not None and split_rows_written >= args.max_rows_per_split:
                    break

            if output_rows:
                write_output_part(out_path, output_rows)
            else:
                part_counters[(split, corpus, language)] -= 1
            if args.max_rows_per_split is not None and split_rows_written >= args.max_rows_per_split:
                break

    with summary_tsv.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["corpus", "language", "hours"], delimiter="\t")
        writer.writeheader()
        for (corpus, language), hours in sorted(train_hours_by_partition.items()):
            writer.writerow({"corpus": corpus, "language": language, "hours": hours})

    dataset_card_path = cards_dataset_dir / f"{args.dataset_card_name}.yaml"
    dataset_card_path.write_text(
        "\n".join(
            [
                f"name: {args.dataset_card_name}",
                "dataset_family: mixture_parquet_asr_dataset",
                "dataset_config:",
                f"  data: {dataset_dir}",
                f"tokenizer_ref: {args.tokenizer_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    config_payload = {
        "model": {"name": args.model_name},
        "dataset": {
            "name": args.dataset_card_name,
            "train_split": "train",
            "valid_split": "val",
            "storage_mode": "MIXTURE_PARQUET",
            "task_mode": "ASR",
            "mixture_parquet_storage_config": {
                "dataset_summary_path": str(summary_tsv),
                "beta_corpus": float(args.beta_corpus),
                "beta_language": float(args.beta_language),
                "fragment_loading": {"cache": True},
            },
            "asr_task_config": {
                "min_audio_len": int(float(args.min_audio_seconds) * args.sample_rate),
                "max_audio_len": int(float(args.max_audio_seconds) * args.sample_rate),
                "batching_strategy": "LENGTH",
                "batch_size": int(args.batch_size),
                "num_seqs_multiple_of": int(args.num_seqs_multiple_of),
                "max_num_elements": int(args.max_num_elements),
                "batch_shuffle_window": int(args.batch_shuffle_window),
                "example_shuffle_window": int(args.example_shuffle_window),
                "normalize_audio": True,
            },
        },
        "tokenizer": {"name": args.tokenizer_name},
        "optimizer": {"config": {"lr": float(args.learning_rate)}},
        "trainer": {
            "data_parallelism": "fsdp",
            "fsdp": {
                "granularity": "stack",
                "version": "v1",
                "fp32_reduce": False,
            },
            "freeze_encoder_for_n_steps": 0,
            "mixed_precision": {"dtype": "torch.bfloat16"},
            "grad_accumulation": {"num_batches": int(args.grad_accumulation)},
        },
        "regime": {
            "num_steps": int(args.train_steps),
            "validate_after_n_steps": int(args.validate_every),
            "validate_every_n_steps": int(args.validate_every),
            "checkpoint_every_n_steps": int(args.checkpoint_every),
            "publish_metrics_every_n_steps": int(args.publish_metrics_every),
        },
    }
    config_path.write_text(yaml.safe_dump(config_payload, sort_keys=False, allow_unicode=True), encoding="utf-8")

    payload = {
        "dataset_root": str(dataset_root),
        "run_root": str(run_root),
        "dataset_dir": str(dataset_dir),
        "dataset_summary_tsv": str(summary_tsv),
        "dataset_card_path": str(dataset_card_path),
        "dataset_card_name": args.dataset_card_name,
        "config_path": str(config_path),
        "model_name": args.model_name,
        "tokenizer_name": args.tokenizer_name,
        "counts_by_split": dict(counts_by_split),
        "counts_by_corpus": dict(counts_by_corpus),
        "counts_by_language": dict(counts_by_language),
        "train_hours_by_partition": {
            f"{corpus}|{language}": hours
            for (corpus, language), hours in sorted(train_hours_by_partition.items())
        },
        "skip_counts": dict(skip_counts),
        "included_parquet_files": included_sources,
        "excluded_source_prefixes": ["qasr"],
        "max_rows_per_split": args.max_rows_per_split,
        "max_audio_seconds": args.max_audio_seconds,
        "max_num_elements": args.max_num_elements,
        "batch_size": args.batch_size,
        "grad_accumulation": args.grad_accumulation,
    }
    summary_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return payload


def main() -> None:
    args = parse_args()
    payload = build_dataset(args)
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
