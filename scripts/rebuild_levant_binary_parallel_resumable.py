#!/usr/bin/env python3
"""Parallel, crash-resumable rebuild of the combined Levant/non-Levant binary
split. Same labeling/splitting rules as rebuild_levant_binary_native_test_splits.py:

- `masc`, `casa/pal`, `casa/jor`: native `test-*` shards are kept as the final
  `test` split verbatim; every other row for those leaves is pooled and
  re-split 80/20 into train/val.
- `qasr`, `omni`, `layla`: unchanged global random train/val/test ratio split
  (default 0.70/0.15/0.15).

Unlike that script, this one is designed to run as six independent OS
processes (one per `--worker`), each touching only its own exclusive output
subdirectories (no two workers ever write into the same directory, so there
is no concurrent-write corruption risk). Each worker checkpoints how many of
its (fixed-order) source files it has fully committed, and only closes its
parquet shard writers at a checkpoint boundary -- so a crash can only ever
cost re-processing the files since the last checkpoint, never full restart.

Run `--worker reduce` once all six finish to merge their summaries into one
reports/summary.json, matching the schema of the non-parallel script.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
import re
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

DEFAULT_DATA_ROOT = Path(".logs/full_combined_data_root")
DEFAULT_TEXT_ROW_PROBS = Path(".logs/combined_row_probs/text_row_probabilities.jsonl")
DEFAULT_AUDIO_ROW_PROBS = Path(".logs/combined_row_probs/audio_row_probabilities.jsonl")
DEFAULT_OUTPUT_ROOT = Path("/workspace/asr/Palestinian-ASR/data_curated_levant_binary_v1")

BINARY_SOURCES = {"masc_c", "qasr"}
LEAF_FILE_GLOBS = {
    "masc": ["masc_c_only*.parquet"],
    "qasr": ["processed_qasr_segments*.parquet"],
    "omni": ["omnilingual_apc*.parquet"],
    "layla": ["layla__*.parquet"],
    "casa/pal": ["casablanca_palestinian*.parquet"],
    "casa/jor": ["casablanca_jordanian*.parquet"],
}
SPLITS = ("train", "val", "test")
NATIVE_TEST_LEAVES = {"masc", "casa/pal", "casa/jor"}
REMAINDER_TRAIN_RATIO = 0.8
REMAINDER_VAL_RATIO = 0.2
WORKERS = ("qasr", "masc", "omni", "layla", "casa_pal", "casa_jor")
CHECKPOINT_EVERY_FILES = 20

_NATIVE_TEST_SEGMENT = re.compile(r"^test-\d+-of-\d+\.parquet$")


def is_native_test_file(path: Path) -> bool:
    return any(_NATIVE_TEST_SEGMENT.match(segment) for segment in path.name.split("__"))


def split_native_test_files(files: list[Path]) -> tuple[list[Path], list[Path]]:
    test_files = [f for f in files if is_native_test_file(f)]
    remainder_files = [f for f in files if not is_native_test_file(f)]
    return test_files, remainder_files


def canonical_path(path_like: str | Path) -> str:
    return str(Path(path_like).resolve())


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def score_from_record(record: dict, primary_key: str, fallback_key: str) -> float:
    if record.get(primary_key) is not None:
        return float(record[primary_key])
    label_scores = record.get("label_scores") or {}
    if fallback_key in label_scores:
        return float(label_scores[fallback_key])
    return 0.0


def text_score_from_audio_record(record: dict) -> float:
    for key in ("text_target_label_score", "text_lev_score"):
        if record.get(key) is not None:
            return float(record[key])
    text_label_scores = record.get("text_label_scores") or {}
    if "LEV" in text_label_scores:
        return float(text_label_scores["LEV"])
    return 0.0


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def collect_text_summary(path: Path, threshold: float) -> dict:
    summary = {
        "threshold": threshold,
        "rows_scanned": 0,
        "source_counts": defaultdict(int),
        "source_threshold_counts": defaultdict(int),
    }
    for record in iter_jsonl(path):
        source = record.get("source")
        if source not in BINARY_SOURCES:
            continue
        summary["rows_scanned"] += 1
        summary["source_counts"][source] += 1
        score = score_from_record(record, "target_label_score", "LEV")
        if score >= threshold:
            summary["source_threshold_counts"][source] += 1
    summary["source_counts"] = dict(summary["source_counts"])
    summary["source_threshold_counts"] = dict(summary["source_threshold_counts"])
    return summary


def collect_levant_row_indices(path: Path, threshold: float) -> tuple[dict[str, set[int]], dict]:
    rows_by_file: dict[str, set[int]] = defaultdict(set)
    summary = {
        "threshold": threshold,
        "rows_scanned": 0,
        "rows_status_ok": 0,
        "source_counts": defaultdict(int),
        "source_accepted_counts": defaultdict(int),
    }
    for record in iter_jsonl(path):
        source = record.get("source")
        if source not in BINARY_SOURCES:
            continue
        summary["rows_scanned"] += 1
        summary["source_counts"][source] += 1
        if record.get("status", "ok") == "ok":
            summary["rows_status_ok"] += 1
        audio_score = score_from_record(record, "target_label_score", "Levantine")
        text_score = text_score_from_audio_record(record)
        if record.get("status", "ok") == "ok" and text_score >= threshold and audio_score >= threshold:
            source_file = record["source_file"]
            row_idx = int(record["row_idx"])
            rows_by_file[canonical_path(source_file)].add(row_idx)
            summary["source_accepted_counts"][source] += 1
    summary["source_counts"] = dict(summary["source_counts"])
    summary["source_accepted_counts"] = dict(summary["source_accepted_counts"])
    return dict(rows_by_file), summary


def gather_files(data_root: Path, globs: list[str]) -> list[Path]:
    files: list[Path] = []
    for pattern in globs:
        files.extend(sorted(data_root.glob(pattern)))
    return sorted(set(files))


def count_rows(path: Path) -> int:
    return pq.ParquetFile(path).metadata.num_rows


def unify_leaf_schema(files: list[Path]) -> pa.Schema | None:
    schemas = [pq.ParquetFile(p).schema_arrow for p in files]
    if not schemas:
        return None
    return pa.unify_schemas(schemas)


def align_table_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    arrays = []
    for field in schema:
        if table.schema.get_field_index(field.name) != -1:
            column = table.column(field.name)
            if not column.type.equals(field.type):
                column = column.cast(field.type)
        else:
            column = pa.nulls(table.num_rows, type=field.type)
        arrays.append(column)
    return pa.Table.from_arrays(arrays, schema=schema)


def split_counts(total: int, train_ratio: float, val_ratio: float) -> dict[str, int]:
    train_count = int(math.floor(total * train_ratio))
    val_count = int(math.floor(total * val_ratio))
    test_count = total - train_count - val_count
    return {"train": train_count, "val": val_count, "test": test_count}


def make_position_set(total: int, train_ratio: float, val_ratio: float, seed_key: str) -> tuple[dict[str, set[int]], dict[str, int]]:
    counts = split_counts(total, train_ratio, val_ratio)
    holdout_count = counts["val"] + counts["test"]
    if holdout_count == 0:
        return {"val": set(), "test": set()}, counts
    rng = random.Random(seed_key)
    holdout_positions = rng.sample(range(total), holdout_count)
    val_positions = set(holdout_positions[: counts["val"]])
    test_positions = set(holdout_positions[counts["val"] :])
    return {"val": val_positions, "test": test_positions}, counts


def make_two_way_position_set(total: int, val_ratio: float, seed_key: str) -> tuple[dict[str, set[int]], dict[str, int]]:
    """Train/val only split with no third bucket -- every row goes to
    exactly one of train or val, no flooring leftover is ever discarded
    (unlike make_position_set, which always carves a `test` bucket out of
    the flooring remainder; using that for a true two-way split silently
    drops up to 1 row)."""
    val_count = round(total * val_ratio)
    train_count = total - val_count
    counts = {"train": train_count, "val": val_count, "test": 0}
    if val_count == 0:
        return {"val": set(), "test": set()}, counts
    rng = random.Random(seed_key)
    val_positions = set(rng.sample(range(total), val_count))
    return {"val": val_positions, "test": set()}, counts


def split_indices_for_ordinals(ordinals: np.ndarray, leaf_sets: dict[str, set[int]]) -> dict[str, np.ndarray]:
    val_positions = leaf_sets["val"]
    test_positions = leaf_sets["test"]
    val_mask = np.fromiter((int(x in val_positions) for x in ordinals), dtype=np.int8, count=len(ordinals)).astype(bool)
    remaining_mask = ~val_mask
    test_candidates = ordinals[remaining_mask]
    test_mask_remaining = np.fromiter(
        (int(x in test_positions) for x in test_candidates), dtype=np.int8, count=len(test_candidates)
    ).astype(bool)
    all_indices = np.arange(len(ordinals), dtype=np.int32)
    val_indices = all_indices[val_mask]
    remaining_indices = all_indices[remaining_mask]
    test_indices = remaining_indices[test_mask_remaining]
    train_indices = remaining_indices[~test_mask_remaining]
    return {"train": train_indices, "val": val_indices, "test": test_indices}


@dataclass
class ShardWriter:
    out_dir: Path
    compression: str
    schema: pa.Schema | None = None

    def __post_init__(self) -> None:
        ensure_dir(self.out_dir)
        self.writer: pq.ParquetWriter | None = None
        self.total_rows = 0
        self.shard_index = self._recover_shard_index()

    def _recover_shard_index(self) -> int:
        existing = sorted(self.out_dir.glob("data-*.parquet"))
        if not existing:
            return 0
        last = existing[-1]
        try:
            pq.ParquetFile(last)
            return len(existing)
        except Exception:
            last.unlink()
            return len(existing) - 1

    def append_table(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        target_schema = self.schema or table.schema
        table = align_table_to_schema(table, target_schema)
        if self.writer is None:
            path = self.out_dir / f"data-{self.shard_index:05d}.parquet"
            self.writer = pq.ParquetWriter(path, schema=target_schema, compression=self.compression)
            self.shard_index += 1
        self.writer.write_table(table)
        self.total_rows += table.num_rows

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None


def load_checkpoint(checkpoint_path: Path) -> int:
    if not checkpoint_path.exists():
        return 0
    try:
        return int(json.loads(checkpoint_path.read_text())["files_committed"])
    except Exception:
        return 0


def save_checkpoint(checkpoint_path: Path, files_committed: int) -> None:
    ensure_dir(checkpoint_path.parent)
    tmp = checkpoint_path.with_suffix(".tmp")
    tmp.write_text(json.dumps({"files_committed": files_committed}), encoding="utf-8")
    os.replace(tmp, checkpoint_path)


@dataclass
class ProgressTracker:
    total_rows: int
    progress_path: Path
    report_interval_sec: int = 30

    def __post_init__(self) -> None:
        self.processed_rows = 0
        self.last_report_time = time.monotonic()
        self.last_phase = "starting"
        ensure_dir(self.progress_path.parent)

    def advance(self, count: int, phase: str) -> None:
        self.processed_rows += count
        self.last_phase = phase
        now = time.monotonic()
        if now - self.last_report_time >= self.report_interval_sec:
            self.emit(force=False)
            self.last_report_time = now

    def emit(self, force: bool) -> None:
        percent = 100.0 if self.total_rows == 0 else (self.processed_rows / self.total_rows) * 100.0
        payload = {
            "processed_rows": self.processed_rows,
            "total_rows": self.total_rows,
            "percent_complete": round(percent, 4),
            "phase": self.last_phase,
            "timestamp_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        }
        self.progress_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if force or percent < 100.0:
            print(json.dumps(payload, ensure_ascii=False), flush=True)


def replay_offset_binary(files_all: list[Path], committed: int, lev_rows_by_file: dict) -> tuple[int, int]:
    lev_offset = 0
    non_lev_offset = 0
    for path in files_all[:committed]:
        total = count_rows(path)
        lev_rows = lev_rows_by_file.get(canonical_path(path), set())
        lev_count = sum(1 for idx in lev_rows if 0 <= idx < total)
        lev_offset += lev_count
        non_lev_offset += total - lev_count
    return lev_offset, non_lev_offset


def replay_offset_single(files_all: list[Path], committed: int) -> int:
    return sum(count_rows(p) for p in files_all[:committed])


def route_binary_resumable(
    files_all: list[Path],
    lev_rows_by_file: dict,
    leaf_prefix: str,
    position_sets: dict,
    writers: dict,
    output_counts: dict,
    batch_size: int,
    progress: ProgressTracker,
    checkpoint_path: Path,
    allowed_splits: tuple[str, ...] = SPLITS,
) -> None:
    committed = load_checkpoint(checkpoint_path)
    lev_offset, non_lev_offset = replay_offset_binary(files_all, committed, lev_rows_by_file)
    leaves = (f"{leaf_prefix}/lev", f"{leaf_prefix}/non_lev")
    ordinal_counters = {leaves[0]: lev_offset, leaves[1]: non_lev_offset}

    remaining = files_all[committed:]
    for i, path in enumerate(remaining):
        parquet_file = pq.ParquetFile(path)
        lev_rows = lev_rows_by_file.get(canonical_path(path), set())
        file_row_idx = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            row_numbers = np.arange(file_row_idx, file_row_idx + table.num_rows, dtype=np.int64)
            lev_mask = np.fromiter(
                (int(r in lev_rows) for r in row_numbers), dtype=np.int8, count=table.num_rows
            ).astype(bool)
            batch_indices = np.arange(table.num_rows, dtype=np.int32)
            for is_lev, leaf in ((True, leaves[0]), (False, leaves[1])):
                selected_indices = batch_indices[lev_mask] if is_lev else batch_indices[~lev_mask]
                if selected_indices.size == 0:
                    continue
                start_ordinal = ordinal_counters[leaf]
                ordinals = np.arange(start_ordinal, start_ordinal + selected_indices.size, dtype=np.int64)
                ordinal_counters[leaf] += selected_indices.size
                split_indices = split_indices_for_ordinals(ordinals, position_sets[leaf])
                selected_table = table.take(pa.array(selected_indices, type=pa.int32()))
                for split, local_idx in split_indices.items():
                    if split not in allowed_splits or local_idx.size == 0:
                        continue
                    final_table = selected_table.take(pa.array(local_idx, type=pa.int32()))
                    writers[(split, leaf)].append_table(final_table)
                    output_counts[leaf][split] += final_table.num_rows
            file_row_idx += table.num_rows
            progress.advance(table.num_rows, f"{leaf_prefix} rewrite")

        files_done_now = committed + i + 1
        if (i + 1) % CHECKPOINT_EVERY_FILES == 0 or (i + 1) == len(remaining):
            for split in SPLITS:
                for leaf in leaves:
                    writers[(split, leaf)].close()
            save_checkpoint(checkpoint_path, files_done_now)


def route_single_resumable(
    files_all: list[Path],
    leaf: str,
    position_sets: dict,
    writers: dict,
    output_counts: dict,
    batch_size: int,
    progress: ProgressTracker,
    checkpoint_path: Path,
) -> None:
    committed = load_checkpoint(checkpoint_path)
    ordinal = replay_offset_single(files_all, committed)
    remaining = files_all[committed:]
    for i, path in enumerate(remaining):
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            ordinals = np.arange(ordinal, ordinal + table.num_rows, dtype=np.int64)
            split_indices = split_indices_for_ordinals(ordinals, position_sets[leaf])
            ordinal += table.num_rows
            for split, indices in split_indices.items():
                if indices.size == 0:
                    continue
                selected = table.take(pa.array(indices, type=pa.int32()))
                writers[(split, leaf)].append_table(selected)
                output_counts[leaf][split] += selected.num_rows
            progress.advance(table.num_rows, f"{leaf} rewrite")

        files_done_now = committed + i + 1
        if (i + 1) % CHECKPOINT_EVERY_FILES == 0 or (i + 1) == len(remaining):
            for split in SPLITS:
                writers[(split, leaf)].close()
            save_checkpoint(checkpoint_path, files_done_now)


def route_all_to_test_resumable(
    files_all: list[Path],
    writers: dict,
    output_counts: dict,
    leaf: str,
    batch_size: int,
    progress: ProgressTracker,
    checkpoint_path: Path,
    lev_rows_by_file: dict | None = None,
    leaf_prefix: str | None = None,
) -> None:
    committed = load_checkpoint(checkpoint_path)
    remaining = files_all[committed:]
    sub_leaves = [f"{leaf_prefix}/lev", f"{leaf_prefix}/non_lev"] if leaf_prefix else [leaf]

    for i, path in enumerate(remaining):
        parquet_file = pq.ParquetFile(path)
        lev_rows = (lev_rows_by_file or {}).get(canonical_path(path), set())
        file_row_idx = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            if leaf_prefix is not None:
                row_numbers = np.arange(file_row_idx, file_row_idx + table.num_rows, dtype=np.int64)
                lev_mask = np.fromiter(
                    (int(r in lev_rows) for r in row_numbers), dtype=np.int8, count=table.num_rows
                ).astype(bool)
                batch_indices = np.arange(table.num_rows, dtype=np.int32)
                for is_lev, sub_leaf in ((True, sub_leaves[0]), (False, sub_leaves[1])):
                    selected_indices = batch_indices[lev_mask] if is_lev else batch_indices[~lev_mask]
                    if selected_indices.size == 0:
                        continue
                    selected_table = table.take(pa.array(selected_indices, type=pa.int32()))
                    writers[("test", sub_leaf)].append_table(selected_table)
                    output_counts[sub_leaf]["test"] += selected_table.num_rows
            else:
                writers[("test", leaf)].append_table(table)
                output_counts[leaf]["test"] += table.num_rows
            file_row_idx += table.num_rows
            progress.advance(table.num_rows, f"{leaf} native-test passthrough")

        files_done_now = committed + i + 1
        if (i + 1) % CHECKPOINT_EVERY_FILES == 0 or (i + 1) == len(remaining):
            for sl in sub_leaves:
                writers[("test", sl)].close()
            save_checkpoint(checkpoint_path, files_done_now)


def route_remainder_resumable(
    files_all: list[Path],
    writers: dict,
    output_counts: dict,
    leaf: str,
    batch_size: int,
    progress: ProgressTracker,
    checkpoint_path: Path,
    seed: int,
    lev_rows_by_file: dict | None = None,
    leaf_prefix: str | None = None,
) -> dict[str, dict[str, int]]:
    sub_leaves = [f"{leaf_prefix}/lev", f"{leaf_prefix}/non_lev"] if leaf_prefix else [leaf]

    totals: dict[str, int] = {sl: 0 for sl in sub_leaves}
    for path in files_all:
        total = count_rows(path)
        if leaf_prefix is not None:
            lev_rows = (lev_rows_by_file or {}).get(canonical_path(path), set())
            lev_count = sum(1 for idx in lev_rows if 0 <= idx < total)
            totals[f"{leaf_prefix}/lev"] += lev_count
            totals[f"{leaf_prefix}/non_lev"] += total - lev_count
        else:
            totals[leaf] += total

    position_sets = {}
    planned = {}
    for sl in sub_leaves:
        pos_set, counts = make_two_way_position_set(totals[sl], REMAINDER_VAL_RATIO, f"{seed}:{sl}:remainder")
        position_sets[sl] = pos_set
        planned[sl] = counts

    committed = load_checkpoint(checkpoint_path)
    if leaf_prefix is not None:
        ordinal_counters = {}
        lev_off, non_lev_off = replay_offset_binary(files_all, committed, lev_rows_by_file or {})
        ordinal_counters[sub_leaves[0]] = lev_off
        ordinal_counters[sub_leaves[1]] = non_lev_off
    else:
        ordinal_counters = {leaf: replay_offset_single(files_all, committed)}

    remaining = files_all[committed:]
    for i, path in enumerate(remaining):
        parquet_file = pq.ParquetFile(path)
        lev_rows = (lev_rows_by_file or {}).get(canonical_path(path), set())
        file_row_idx = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            batch_indices = np.arange(table.num_rows, dtype=np.int32)
            if leaf_prefix is not None:
                row_numbers = np.arange(file_row_idx, file_row_idx + table.num_rows, dtype=np.int64)
                lev_mask = np.fromiter(
                    (int(r in lev_rows) for r in row_numbers), dtype=np.int8, count=table.num_rows
                ).astype(bool)
                groups = ((True, sub_leaves[0]), (False, sub_leaves[1]))
            else:
                lev_mask = np.ones(table.num_rows, dtype=bool)
                groups = ((True, leaf),)

            for is_lev, sl in groups:
                selected_indices = batch_indices[lev_mask] if is_lev else batch_indices[~lev_mask]
                if selected_indices.size == 0:
                    continue
                start_ordinal = ordinal_counters[sl]
                ordinals = np.arange(start_ordinal, start_ordinal + selected_indices.size, dtype=np.int64)
                ordinal_counters[sl] += selected_indices.size
                split_indices = split_indices_for_ordinals(ordinals, position_sets[sl])
                selected_table = table.take(pa.array(selected_indices, type=pa.int32()))
                for split, local_idx in split_indices.items():
                    if split == "test" or local_idx.size == 0:
                        continue
                    final_table = selected_table.take(pa.array(local_idx, type=pa.int32()))
                    writers[(split, sl)].append_table(final_table)
                    output_counts[sl][split] += final_table.num_rows
            file_row_idx += table.num_rows
            progress.advance(table.num_rows, f"{leaf} remainder 80/20 split")

        files_done_now = committed + i + 1
        if (i + 1) % CHECKPOINT_EVERY_FILES == 0 or (i + 1) == len(remaining):
            for sl in sub_leaves:
                writers[("train", sl)].close()
                writers[("val", sl)].close()
            save_checkpoint(checkpoint_path, files_done_now)

    return planned


def build_writer_subset(output_root: Path, compression: str, keys: list[tuple[str, str]], schemas: dict) -> dict:
    writers = {}
    for split, leaf in keys:
        writers[(split, leaf)] = ShardWriter(
            out_dir=output_root / split / Path(leaf),
            compression=compression,
            schema=schemas.get(leaf),
        )
    return writers


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--worker", required=True, choices=(*WORKERS, "reduce"))
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--text-row-probs", type=Path, default=DEFAULT_TEXT_ROW_PROBS)
    parser.add_argument("--audio-row-probs", type=Path, default=DEFAULT_AUDIO_ROW_PROBS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--compression", default="snappy")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    reports_dir = args.output_root / "reports"
    checkpoints_dir = reports_dir / "checkpoints"
    worker_summaries_dir = reports_dir / "worker_summaries"
    ensure_dir(reports_dir)
    ensure_dir(checkpoints_dir)
    ensure_dir(worker_summaries_dir)

    if args.worker == "reduce":
        run_reduce(args, worker_summaries_dir, reports_dir)
        return

    file_groups = {leaf: gather_files(args.data_root, globs) for leaf, globs in LEAF_FILE_GLOBS.items()}
    levant_rows_by_file, audio_summary = collect_levant_row_indices(args.audio_row_probs, args.threshold)
    progress = ProgressTracker(total_rows=1, progress_path=reports_dir / f"progress_{args.worker}.json")

    if args.worker == "qasr":
        files = file_groups["qasr"]
        totals = {"qasr/lev": 0, "qasr/non_lev": 0}
        for path in files:
            total = count_rows(path)
            lev_rows = levant_rows_by_file.get(canonical_path(path), set())
            lev_count = sum(1 for idx in lev_rows if 0 <= idx < total)
            totals["qasr/lev"] += lev_count
            totals["qasr/non_lev"] += total - lev_count
        progress.total_rows = sum(totals.values())
        position_sets = {}
        planned = {}
        for leaf in ("qasr/lev", "qasr/non_lev"):
            pos_set, counts = make_position_set(totals[leaf], args.train_ratio, args.val_ratio, f"{args.seed}:{leaf}")
            position_sets[leaf] = pos_set
            planned[leaf] = counts
        schema = unify_leaf_schema(files)
        writers = build_writer_subset(
            args.output_root, args.compression,
            [(s, leaf) for s in SPLITS for leaf in ("qasr/lev", "qasr/non_lev")],
            {"qasr/lev": schema, "qasr/non_lev": schema},
        )
        output_counts = {leaf: {"train": 0, "val": 0, "test": 0} for leaf in ("qasr/lev", "qasr/non_lev")}
        try:
            route_binary_resumable(
                files, levant_rows_by_file, "qasr", position_sets, writers, output_counts,
                args.batch_size, progress, checkpoints_dir / "qasr.json",
            )
        finally:
            for w in writers.values():
                w.close()
            progress.emit(force=True)
        write_worker_summary(worker_summaries_dir, "qasr", totals, planned, output_counts)

    elif args.worker == "masc":
        test_files, remainder_files = split_native_test_files(file_groups["masc"])
        total_all = sum(count_rows(p) for p in file_groups["masc"])
        progress.total_rows = total_all
        schema = unify_leaf_schema(file_groups["masc"])
        writers = build_writer_subset(
            args.output_root, args.compression,
            [(s, leaf) for s in SPLITS for leaf in ("masc/lev", "masc/non_lev")],
            {"masc/lev": schema, "masc/non_lev": schema},
        )
        output_counts = {leaf: {"train": 0, "val": 0, "test": 0} for leaf in ("masc/lev", "masc/non_lev")}
        try:
            route_all_to_test_resumable(
                test_files, writers, output_counts, "masc", args.batch_size, progress,
                checkpoints_dir / "masc_test.json", lev_rows_by_file=levant_rows_by_file, leaf_prefix="masc",
            )
            planned = route_remainder_resumable(
                remainder_files, writers, output_counts, "masc", args.batch_size, progress,
                checkpoints_dir / "masc_remainder.json", args.seed,
                lev_rows_by_file=levant_rows_by_file, leaf_prefix="masc",
            )
        finally:
            for w in writers.values():
                w.close()
            progress.emit(force=True)
        totals = {
            "masc/lev": output_counts["masc/lev"]["train"] + output_counts["masc/lev"]["val"] + output_counts["masc/lev"]["test"],
            "masc/non_lev": output_counts["masc/non_lev"]["train"] + output_counts["masc/non_lev"]["val"] + output_counts["masc/non_lev"]["test"],
        }
        write_worker_summary(worker_summaries_dir, "masc", totals, planned, output_counts)

    elif args.worker in ("omni", "layla"):
        files = file_groups[args.worker]
        total = sum(count_rows(p) for p in files)
        progress.total_rows = total
        pos_set, counts = make_position_set(total, args.train_ratio, args.val_ratio, f"{args.seed}:{args.worker}")
        schema = unify_leaf_schema(files)
        writers = build_writer_subset(
            args.output_root, args.compression, [(s, args.worker) for s in SPLITS], {args.worker: schema}
        )
        output_counts = {args.worker: {"train": 0, "val": 0, "test": 0}}
        try:
            route_single_resumable(
                files, args.worker, {args.worker: pos_set}, writers, output_counts,
                args.batch_size, progress, checkpoints_dir / f"{args.worker}.json",
            )
        finally:
            for w in writers.values():
                w.close()
            progress.emit(force=True)
        write_worker_summary(worker_summaries_dir, args.worker, {args.worker: total}, {args.worker: counts}, output_counts)

    elif args.worker in ("casa_pal", "casa_jor"):
        leaf = "casa/pal" if args.worker == "casa_pal" else "casa/jor"
        test_files, remainder_files = split_native_test_files(file_groups[leaf])
        total_all = sum(count_rows(p) for p in file_groups[leaf])
        progress.total_rows = total_all
        schema = unify_leaf_schema(file_groups[leaf])
        writers = build_writer_subset(args.output_root, args.compression, [(s, leaf) for s in SPLITS], {leaf: schema})
        output_counts = {leaf: {"train": 0, "val": 0, "test": 0}}
        try:
            route_all_to_test_resumable(
                test_files, writers, output_counts, leaf, args.batch_size, progress,
                checkpoints_dir / f"{args.worker}_test.json",
            )
            planned = route_remainder_resumable(
                remainder_files, writers, output_counts, leaf, args.batch_size, progress,
                checkpoints_dir / f"{args.worker}_remainder.json", args.seed,
            )
        finally:
            for w in writers.values():
                w.close()
            progress.emit(force=True)
        write_worker_summary(worker_summaries_dir, args.worker, {leaf: total_all}, planned, output_counts)


def write_worker_summary(worker_summaries_dir: Path, worker: str, totals: dict, planned: dict, actual: dict) -> None:
    payload = {"worker": worker, "totals": totals, "planned_split_counts": planned, "actual_split_counts": actual}
    (worker_summaries_dir / f"{worker}.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def run_reduce(args: argparse.Namespace, worker_summaries_dir: Path, reports_dir: Path) -> None:
    missing = [w for w in WORKERS if not (worker_summaries_dir / f"{w}.json").exists()]
    if missing:
        raise SystemExit(f"Cannot reduce: missing worker summaries for {missing}")

    text_summary = collect_text_summary(args.text_row_probs, args.threshold)
    _, audio_summary = collect_levant_row_indices(args.audio_row_probs, args.threshold)

    totals_by_leaf: dict[str, int] = {}
    planned_split_counts: dict[str, dict[str, int]] = {}
    actual_split_counts: dict[str, dict[str, int]] = {}
    for w in WORKERS:
        payload = json.loads((worker_summaries_dir / f"{w}.json").read_text())
        totals_by_leaf.update(payload["totals"])
        planned_split_counts.update(payload["planned_split_counts"])
        actual_split_counts.update(payload["actual_split_counts"])

    summary = {
        "output_root": str(args.output_root),
        "data_root": str(args.data_root),
        "text_row_probabilities": str(args.text_row_probs),
        "audio_row_probabilities": str(args.audio_row_probs),
        "threshold": args.threshold,
        "seed": args.seed,
        "unchanged_leaves_ratios": {"train": args.train_ratio, "val": args.val_ratio, "test": args.test_ratio},
        "native_test_leaves": sorted(NATIVE_TEST_LEAVES),
        "native_test_leaves_remainder_ratios": {"train": REMAINDER_TRAIN_RATIO, "val": REMAINDER_VAL_RATIO},
        "binary_label_rule": {
            "masc_c_and_qasr_lev": "text LEV >= threshold and audio Levantine >= threshold",
            "masc_c_and_qasr_non_lev": "all remaining rows",
        },
        "text_scan_summary": text_summary,
        "audio_scan_summary": audio_summary,
        "totals_by_leaf": totals_by_leaf,
        "planned_split_counts": planned_split_counts,
        "actual_split_counts": actual_split_counts,
    }
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps(actual_split_counts, indent=2, ensure_ascii=False))
    print(f"Wrote summary to {reports_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
