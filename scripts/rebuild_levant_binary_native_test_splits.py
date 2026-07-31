#!/usr/bin/env python3
"""Rebuild the combined Levant/non-Levant binary split, derived from
create_levant_non_levant_splits.py, with one behavioral change:

For `masc`, `casa/pal`, and `casa/jor`, rows that originated from the source
dataset's own native `test-NNNNN-of-NNNNN.parquet` shards are kept as the
final `test` split verbatim (never resampled into train/val). Every other
row for those three leaves (originally `train-*` and/or `validation-*`
shards) is pooled together and re-split 80/20 into train/val.

`qasr`, `omni`, and `layla` are untouched: same global random
train/val/test ratio split as the original script, no native-test carving.
"""
from __future__ import annotations

import argparse
import json
import math
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
NON_BINARY_LEAVES = ("omni", "layla", "casa/pal", "casa/jor")
SPLITS = ("train", "val", "test")

# Leaves where the source dataset's own `test-*` shards are preserved as the
# final test split, with the remaining rows re-split 80/20 train/val.
NATIVE_TEST_LEAVES = {"masc", "casa/pal", "casa/jor"}
REMAINDER_TRAIN_RATIO = 0.8
REMAINDER_VAL_RATIO = 0.2

_NATIVE_TEST_SEGMENT = re.compile(r"^test-\d+-of-\d+\.parquet$")


def is_native_test_file(path: Path) -> bool:
    return any(_NATIVE_TEST_SEGMENT.match(segment) for segment in path.name.split("__"))


def split_native_test_files(files: list[Path]) -> tuple[list[Path], list[Path]]:
    test_files = [f for f in files if is_native_test_file(f)]
    remainder_files = [f for f in files if not is_native_test_file(f)]
    return test_files, remainder_files


@dataclass
class ProgressTracker:
    total_rows: int
    output_root: Path
    report_interval_sec: int = 60

    def __post_init__(self) -> None:
        self.processed_rows = 0
        self.last_report_time = time.monotonic()
        self.last_phase = "starting"
        ensure_dir(self.output_root / "reports")

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
        progress_path = self.output_root / "reports" / "progress.json"
        progress_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        if force or percent < 100.0:
            print(json.dumps(payload, ensure_ascii=False), flush=True)


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

        if (
            record.get("status", "ok") == "ok"
            and text_score >= threshold
            and audio_score >= threshold
        ):
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
    schemas = []
    for path in files:
        parquet_file = pq.ParquetFile(path)
        schemas.append(parquet_file.schema_arrow)
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


def split_indices_for_ordinals(ordinals: np.ndarray, leaf_sets: dict[str, set[int]]) -> dict[str, np.ndarray]:
    val_positions = leaf_sets["val"]
    test_positions = leaf_sets["test"]

    val_mask = np.fromiter((int(x in val_positions) for x in ordinals), dtype=np.int8, count=len(ordinals)).astype(bool)
    remaining_mask = ~val_mask
    test_candidates = ordinals[remaining_mask]
    test_mask_remaining = np.fromiter(
        (int(x in test_positions) for x in test_candidates),
        dtype=np.int8,
        count=len(test_candidates),
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
    rows_per_shard: int
    compression: str
    schema: pa.Schema | None = None

    def __post_init__(self) -> None:
        ensure_dir(self.out_dir)
        self.writer: pq.ParquetWriter | None = None
        self.rows_in_current = 0
        self.shard_index = 0
        self.shard_paths: list[str] = []
        self.total_rows = 0

    def _start_new_shard(self, schema: pa.Schema) -> None:
        path = self.out_dir / f"data-{self.shard_index:05d}.parquet"
        self.writer = pq.ParquetWriter(path, schema=schema, compression=self.compression)
        self.shard_paths.append(str(path))
        self.rows_in_current = 0
        self.shard_index += 1

    def append_table(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        target_schema = self.schema or table.schema
        table = align_table_to_schema(table, target_schema)
        offset = 0
        while offset < table.num_rows:
            if self.writer is None:
                self._start_new_shard(target_schema)
            assert self.writer is not None
            remaining = self.rows_per_shard - self.rows_in_current
            chunk = table.slice(offset, remaining)
            self.writer.write_table(chunk)
            written = chunk.num_rows
            self.rows_in_current += written
            self.total_rows += written
            offset += written
            if self.rows_in_current >= self.rows_per_shard:
                self.close()

    def close(self) -> None:
        if self.writer is not None:
            self.writer.close()
            self.writer = None
            self.rows_in_current = 0


def writer_key(split: str, leaf: str) -> tuple[str, str]:
    return split, leaf


ALL_LEAVES = (
    "masc/lev",
    "masc/non_lev",
    "qasr/lev",
    "qasr/non_lev",
    "omni",
    "layla",
    "casa/pal",
    "casa/jor",
)


def build_writers(
    output_root: Path,
    rows_per_shard: int,
    compression: str,
    leaf_schemas: dict[str, pa.Schema | None],
) -> dict[tuple[str, str], ShardWriter]:
    writers = {}
    for split in SPLITS:
        for leaf in ALL_LEAVES:
            writers[writer_key(split, leaf)] = ShardWriter(
                out_dir=output_root / split / Path(leaf),
                rows_per_shard=rows_per_shard,
                compression=compression,
                schema=leaf_schemas.get(leaf),
            )
    return writers


def route_rows_all_to_test(
    files: list[Path],
    writers: dict[tuple[str, str], ShardWriter],
    output_counts: dict[str, dict[str, int]],
    leaf: str,
    batch_size: int,
    progress: ProgressTracker,
    lev_rows_by_file: dict[str, set[int]] | None = None,
    leaf_prefix: str | None = None,
) -> None:
    """Every row from `files` goes straight to the `test` split. If
    lev_rows_by_file/leaf_prefix are given, rows are further routed to
    `{leaf_prefix}/lev` or `{leaf_prefix}/non_lev` first."""
    for path in files:
        parquet_file = pq.ParquetFile(path)
        lev_rows = (lev_rows_by_file or {}).get(canonical_path(path), set())
        file_row_idx = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            if leaf_prefix is not None:
                row_numbers = np.arange(file_row_idx, file_row_idx + table.num_rows, dtype=np.int64)
                lev_mask = np.fromiter(
                    (int(row_idx in lev_rows) for row_idx in row_numbers),
                    dtype=np.int8,
                    count=table.num_rows,
                ).astype(bool)
                batch_indices = np.arange(table.num_rows, dtype=np.int32)
                for is_lev, sub_leaf in ((True, f"{leaf_prefix}/lev"), (False, f"{leaf_prefix}/non_lev")):
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


def route_rows_remainder_split(
    files: list[Path],
    writers: dict[tuple[str, str], ShardWriter],
    output_counts: dict[str, dict[str, int]],
    leaf: str,
    batch_size: int,
    progress: ProgressTracker,
    seed: int,
    lev_rows_by_file: dict[str, set[int]] | None = None,
    leaf_prefix: str | None = None,
) -> dict[str, dict[str, int]]:
    """80/20 train/val split of `files`' rows (no test). If
    lev_rows_by_file/leaf_prefix are given, rows are first routed to
    `{leaf_prefix}/lev` or `{leaf_prefix}/non_lev`, each split 80/20
    independently."""
    sub_leaves = [f"{leaf_prefix}/lev", f"{leaf_prefix}/non_lev"] if leaf_prefix else [leaf]

    totals: dict[str, int] = {sl: 0 for sl in sub_leaves}
    for path in files:
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
        pos_set, counts = make_position_set(totals[sl], REMAINDER_TRAIN_RATIO, REMAINDER_VAL_RATIO, f"{seed}:{sl}:remainder")
        position_sets[sl] = pos_set
        planned[sl] = counts

    ordinal_counters = {sl: 0 for sl in sub_leaves}
    for path in files:
        parquet_file = pq.ParquetFile(path)
        lev_rows = (lev_rows_by_file or {}).get(canonical_path(path), set())
        file_row_idx = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            batch_indices = np.arange(table.num_rows, dtype=np.int32)

            if leaf_prefix is not None:
                row_numbers = np.arange(file_row_idx, file_row_idx + table.num_rows, dtype=np.int64)
                lev_mask = np.fromiter(
                    (int(row_idx in lev_rows) for row_idx in row_numbers),
                    dtype=np.int8,
                    count=table.num_rows,
                ).astype(bool)
                groups = ((True, f"{leaf_prefix}/lev"), (False, f"{leaf_prefix}/non_lev"))
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
                for split, split_local_indices in split_indices.items():
                    if split_local_indices.size == 0:
                        continue
                    if split == "test":
                        continue  # remainder pool never produces test rows
                    final_table = selected_table.take(pa.array(split_local_indices, type=pa.int32()))
                    writers[(split, sl)].append_table(final_table)
                    output_counts[sl][split] += final_table.num_rows

            file_row_idx += table.num_rows
            progress.advance(table.num_rows, f"{leaf} remainder 80/20 split")

    return planned


def route_binary_source_existing_logic(
    files: list[Path],
    lev_rows_by_file: dict[str, set[int]],
    leaf_prefix: str,
    position_sets: dict[str, dict[str, set[int]]],
    writers: dict[tuple[str, str], ShardWriter],
    output_counts: dict[str, dict[str, int]],
    batch_size: int,
    progress: ProgressTracker,
) -> None:
    """Unchanged global random-ratio logic, used for `qasr` only."""
    ordinal_counters = {f"{leaf_prefix}/lev": 0, f"{leaf_prefix}/non_lev": 0}

    for path in files:
        parquet_file = pq.ParquetFile(path)
        lev_rows = lev_rows_by_file.get(canonical_path(path), set())
        file_row_idx = 0

        for batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            row_numbers = np.arange(file_row_idx, file_row_idx + table.num_rows, dtype=np.int64)
            lev_mask = np.fromiter(
                (int(row_idx in lev_rows) for row_idx in row_numbers),
                dtype=np.int8,
                count=table.num_rows,
            ).astype(bool)
            batch_indices = np.arange(table.num_rows, dtype=np.int32)

            for is_lev, leaf in ((True, f"{leaf_prefix}/lev"), (False, f"{leaf_prefix}/non_lev")):
                selected_indices = batch_indices[lev_mask] if is_lev else batch_indices[~lev_mask]
                if selected_indices.size == 0:
                    continue
                start_ordinal = ordinal_counters[leaf]
                ordinals = np.arange(start_ordinal, start_ordinal + selected_indices.size, dtype=np.int64)
                ordinal_counters[leaf] += selected_indices.size
                split_indices = split_indices_for_ordinals(ordinals, position_sets[leaf])

                selected_table = table.take(pa.array(selected_indices, type=pa.int32()))
                for split, split_local_indices in split_indices.items():
                    if split_local_indices.size == 0:
                        continue
                    final_table = selected_table.take(pa.array(split_local_indices, type=pa.int32()))
                    writers[(split, leaf)].append_table(final_table)
                    output_counts[leaf][split] += final_table.num_rows

            file_row_idx += table.num_rows
            progress.advance(table.num_rows, f"{leaf_prefix} source rewrite")


def route_single_leaf_existing_logic(
    files: list[Path],
    leaf: str,
    position_sets: dict[str, dict[str, set[int]]],
    writers: dict[tuple[str, str], ShardWriter],
    output_counts: dict[str, dict[str, int]],
    batch_size: int,
    progress: ProgressTracker,
) -> None:
    """Unchanged global random-ratio logic, used for `omni`/`layla` only."""
    ordinal = 0
    for path in files:
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
            progress.advance(table.num_rows, f"{leaf} source rewrite")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--text-row-probs", type=Path, default=DEFAULT_TEXT_ROW_PROBS)
    parser.add_argument("--audio-row-probs", type=Path, default=DEFAULT_AUDIO_ROW_PROBS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--train-ratio", type=float, default=0.7, help="qasr/omni/layla only")
    parser.add_argument("--val-ratio", type=float, default=0.15, help="qasr/omni/layla only")
    parser.add_argument("--test-ratio", type=float, default=0.15, help="qasr/omni/layla only")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows-per-shard", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise SystemExit(f"Split ratios must sum to 1.0, got {ratio_sum}")
    if args.output_root.exists():
        if not args.overwrite:
            raise SystemExit(f"Output root already exists: {args.output_root}. Use --overwrite to replace it.")
        shutil.rmtree(args.output_root)


def main() -> None:
    args = parse_args()
    validate_args(args)

    text_summary = collect_text_summary(args.text_row_probs, args.threshold)
    levant_rows_by_file, audio_summary = collect_levant_row_indices(args.audio_row_probs, args.threshold)

    file_groups = {leaf: gather_files(args.data_root, globs) for leaf, globs in LEAF_FILE_GLOBS.items()}

    masc_test_files, masc_remainder_files = split_native_test_files(file_groups["masc"])
    casa_pal_test_files, casa_pal_remainder_files = split_native_test_files(file_groups["casa/pal"])
    casa_jor_test_files, casa_jor_remainder_files = split_native_test_files(file_groups["casa/jor"])

    print(
        f"masc: {len(masc_test_files)} native-test shards, {len(masc_remainder_files)} remainder shards",
        flush=True,
    )
    print(
        f"casa/pal: {len(casa_pal_test_files)} native-test shards, {len(casa_pal_remainder_files)} remainder shards",
        flush=True,
    )
    print(
        f"casa/jor: {len(casa_jor_test_files)} native-test shards, {len(casa_jor_remainder_files)} remainder shards",
        flush=True,
    )

    def totals_for(files: list[Path]) -> int:
        return sum(count_rows(p) for p in files)

    totals_by_leaf: dict[str, int] = {}
    for path in file_groups["masc"]:
        total = count_rows(path)
        lev_rows = levant_rows_by_file.get(canonical_path(path), set())
        lev_count = sum(1 for idx in lev_rows if 0 <= idx < total)
        totals_by_leaf["masc/lev"] = totals_by_leaf.get("masc/lev", 0) + lev_count
        totals_by_leaf["masc/non_lev"] = totals_by_leaf.get("masc/non_lev", 0) + (total - lev_count)
    for path in file_groups["qasr"]:
        total = count_rows(path)
        lev_rows = levant_rows_by_file.get(canonical_path(path), set())
        lev_count = sum(1 for idx in lev_rows if 0 <= idx < total)
        totals_by_leaf["qasr/lev"] = totals_by_leaf.get("qasr/lev", 0) + lev_count
        totals_by_leaf["qasr/non_lev"] = totals_by_leaf.get("qasr/non_lev", 0) + (total - lev_count)
    for leaf in NON_BINARY_LEAVES:
        totals_by_leaf[leaf] = totals_for(file_groups[leaf])

    # position sets for the leaves that keep the original global-ratio logic
    unchanged_position_sets: dict[str, dict[str, set[int]]] = {}
    unchanged_planned: dict[str, dict[str, int]] = {}
    for leaf in ("qasr/lev", "qasr/non_lev", "omni", "layla"):
        pos_set, counts = make_position_set(totals_by_leaf[leaf], args.train_ratio, args.val_ratio, f"{args.seed}:{leaf}")
        unchanged_position_sets[leaf] = pos_set
        unchanged_planned[leaf] = counts

    leaf_schemas = {
        "masc/lev": unify_leaf_schema(file_groups["masc"]),
        "masc/non_lev": unify_leaf_schema(file_groups["masc"]),
        "qasr/lev": unify_leaf_schema(file_groups["qasr"]),
        "qasr/non_lev": unify_leaf_schema(file_groups["qasr"]),
        "omni": unify_leaf_schema(file_groups["omni"]),
        "layla": unify_leaf_schema(file_groups["layla"]),
        "casa/pal": unify_leaf_schema(file_groups["casa/pal"]),
        "casa/jor": unify_leaf_schema(file_groups["casa/jor"]),
    }

    writers = build_writers(args.output_root, args.rows_per_shard, args.compression, leaf_schemas)
    progress = ProgressTracker(total_rows=sum(totals_by_leaf.values()), output_root=args.output_root)
    actual_counts = {leaf: {"train": 0, "val": 0, "test": 0} for leaf in ALL_LEAVES}

    remainder_planned: dict[str, dict[str, int]] = {}

    try:
        # masc: native test -> test, remainder -> 80/20 train/val, both lev/non_lev aware
        route_rows_all_to_test(
            files=masc_test_files,
            writers=writers,
            output_counts=actual_counts,
            leaf="masc",
            batch_size=args.batch_size,
            progress=progress,
            lev_rows_by_file=levant_rows_by_file,
            leaf_prefix="masc",
        )
        remainder_planned.update(
            route_rows_remainder_split(
                files=masc_remainder_files,
                writers=writers,
                output_counts=actual_counts,
                leaf="masc",
                batch_size=args.batch_size,
                progress=progress,
                seed=args.seed,
                lev_rows_by_file=levant_rows_by_file,
                leaf_prefix="masc",
            )
        )

        # qasr: unchanged existing logic
        route_binary_source_existing_logic(
            files=file_groups["qasr"],
            lev_rows_by_file=levant_rows_by_file,
            leaf_prefix="qasr",
            position_sets=unchanged_position_sets,
            writers=writers,
            output_counts=actual_counts,
            batch_size=args.batch_size,
            progress=progress,
        )

        # omni, layla: unchanged existing logic
        for leaf in ("omni", "layla"):
            route_single_leaf_existing_logic(
                files=file_groups[leaf],
                leaf=leaf,
                position_sets=unchanged_position_sets,
                writers=writers,
                output_counts=actual_counts,
                batch_size=args.batch_size,
                progress=progress,
            )

        # casa/pal, casa/jor: native test -> test, remainder -> 80/20 train/val, no lev/non_lev
        for leaf, test_files, remainder_files in (
            ("casa/pal", casa_pal_test_files, casa_pal_remainder_files),
            ("casa/jor", casa_jor_test_files, casa_jor_remainder_files),
        ):
            route_rows_all_to_test(
                files=test_files,
                writers=writers,
                output_counts=actual_counts,
                leaf=leaf,
                batch_size=args.batch_size,
                progress=progress,
            )
            remainder_planned.update(
                route_rows_remainder_split(
                    files=remainder_files,
                    writers=writers,
                    output_counts=actual_counts,
                    leaf=leaf,
                    batch_size=args.batch_size,
                    progress=progress,
                    seed=args.seed,
                )
            )
    finally:
        for writer in writers.values():
            writer.close()
        progress.emit(force=True)

    planned_counts: dict[str, dict[str, int]] = {}
    for leaf in ("qasr/lev", "qasr/non_lev", "omni", "layla"):
        planned_counts[leaf] = unchanged_planned[leaf]
    for leaf in ("masc/lev", "masc/non_lev", "casa/pal", "casa/jor"):
        planned_counts[leaf] = remainder_planned.get(leaf, {"train": 0, "val": 0, "test": 0})

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
        "native_test_shard_counts": {
            "masc": {"test_shards": len(masc_test_files), "remainder_shards": len(masc_remainder_files)},
            "casa/pal": {"test_shards": len(casa_pal_test_files), "remainder_shards": len(casa_pal_remainder_files)},
            "casa/jor": {"test_shards": len(casa_jor_test_files), "remainder_shards": len(casa_jor_remainder_files)},
        },
        "text_scan_summary": text_summary,
        "audio_scan_summary": audio_summary,
        "totals_by_leaf": totals_by_leaf,
        "planned_split_counts": planned_counts,
        "actual_split_counts": actual_counts,
    }

    reports_dir = args.output_root / "reports"
    ensure_dir(reports_dir)
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary["actual_split_counts"], indent=2, ensure_ascii=False))
    print(f"Wrote summary to {reports_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
