#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/home/MohammadNabulsi/whisper")
DATA_ROOT = ROOT / "data"
DEFAULT_TEXT_ROW_PROBS = (
    ROOT
    / "Runs"
    / "text_dialect_scan_marbertv2_written_clean_masc_c_qasr"
    / "row_probabilities.jsonl"
)
DEFAULT_AUDIO_ROW_PROBS = (
    ROOT
    / "Runs"
    / "dialect_scan_badrex_mms300m_lev08_text_candidates_masc_c_qasr"
    / "row_probabilities.jsonl"
)
DEFAULT_OUTPUT_ROOT = ROOT / "data_curated_levant_binary_v1"

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


@dataclass
class ProgressTracker:
    total_rows: int
    output_root: Path
    report_interval_sec: int = 300

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


def shard_row_key(source_file: str | Path, row_idx: int) -> str:
    return f"{canonical_path(source_file)}::{row_idx}"


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


def compute_totals(
    data_root: Path,
    levant_rows_by_file: dict[str, set[int]],
) -> tuple[dict[str, list[Path]], dict[str, int], list[str]]:
    file_groups = {
        leaf: gather_files(data_root, globs)
        for leaf, globs in LEAF_FILE_GLOBS.items()
    }
    totals = defaultdict(int)
    warnings: list[str] = []

    for path in file_groups["masc"]:
        total = count_rows(path)
        lev_rows = levant_rows_by_file.get(canonical_path(path), set())
        invalid = [idx for idx in lev_rows if idx < 0 or idx >= total]
        if invalid:
            warnings.append(f"{path.name}: ignored {len(invalid)} out-of-range MASC lev row indices")
        lev_count = sum(1 for idx in lev_rows if 0 <= idx < total)
        totals["masc/lev"] += lev_count
        totals["masc/non_lev"] += total - lev_count

    for path in file_groups["qasr"]:
        total = count_rows(path)
        lev_rows = levant_rows_by_file.get(canonical_path(path), set())
        invalid = [idx for idx in lev_rows if idx < 0 or idx >= total]
        if invalid:
            warnings.append(f"{path.name}: ignored {len(invalid)} out-of-range QASR lev row indices")
        lev_count = sum(1 for idx in lev_rows if 0 <= idx < total)
        totals["qasr/lev"] += lev_count
        totals["qasr/non_lev"] += total - lev_count

    for leaf in NON_BINARY_LEAVES:
        totals[leaf] = sum(count_rows(path) for path in file_groups[leaf])

    return file_groups, dict(totals), warnings


def split_counts(total: int, train_ratio: float, val_ratio: float) -> dict[str, int]:
    train_count = int(math.floor(total * train_ratio))
    val_count = int(math.floor(total * val_ratio))
    test_count = total - train_count - val_count
    return {"train": train_count, "val": val_count, "test": test_count}


def make_position_sets(
    totals_by_leaf: dict[str, int],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> tuple[dict[str, dict[str, set[int]]], dict[str, dict[str, int]]]:
    position_sets: dict[str, dict[str, set[int]]] = {}
    counts_by_leaf: dict[str, dict[str, int]] = {}

    for leaf, total in totals_by_leaf.items():
        counts = split_counts(total, train_ratio, val_ratio)
        counts_by_leaf[leaf] = counts
        holdout_count = counts["val"] + counts["test"]
        if holdout_count == 0:
            position_sets[leaf] = {"val": set(), "test": set()}
            continue
        leaf_seed = f"{seed}:{leaf}"
        rng = random.Random(leaf_seed)
        holdout_positions = rng.sample(range(total), holdout_count)
        val_positions = set(holdout_positions[: counts["val"]])
        test_positions = set(holdout_positions[counts["val"] :])
        position_sets[leaf] = {"val": val_positions, "test": test_positions}

    return position_sets, counts_by_leaf


def resolve_split(leaf: str, ordinal: int, position_sets: dict[str, dict[str, set[int]]]) -> str:
    leaf_sets = position_sets[leaf]
    if ordinal in leaf_sets["val"]:
        return "val"
    if ordinal in leaf_sets["test"]:
        return "test"
    return "train"


def split_indices_for_ordinals(
    ordinals: np.ndarray,
    leaf: str,
    position_sets: dict[str, dict[str, set[int]]],
) -> dict[str, np.ndarray]:
    leaf_sets = position_sets[leaf]
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

    return {
        "train": train_indices,
        "val": val_indices,
        "test": test_indices,
    }


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


def build_writers(
    output_root: Path,
    rows_per_shard: int,
    compression: str,
    leaf_schemas: dict[str, pa.Schema | None],
) -> dict[tuple[str, str], ShardWriter]:
    writers = {}
    for split in SPLITS:
        for leaf in (
            "masc/lev",
            "masc/non_lev",
            "qasr/lev",
            "qasr/non_lev",
            "omni",
            "layla",
            "casa/pal",
            "casa/jor",
        ):
            writers[writer_key(split, leaf)] = ShardWriter(
                out_dir=output_root / split / Path(leaf),
                rows_per_shard=rows_per_shard,
                compression=compression,
                schema=leaf_schemas.get(leaf),
            )
    return writers


def route_binary_source(
    files: list[Path],
    lev_rows_by_file: dict[str, set[int]],
    leaf_prefix: str,
    position_sets: dict[str, dict[str, set[int]]],
    writers: dict[tuple[str, str], ShardWriter],
    output_counts: dict[str, dict[str, int]],
    batch_size: int,
    progress: ProgressTracker,
) -> None:
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

            for is_lev, leaf in (
                (True, f"{leaf_prefix}/lev"),
                (False, f"{leaf_prefix}/non_lev"),
            ):
                selected_indices = batch_indices[lev_mask] if is_lev else batch_indices[~lev_mask]
                if selected_indices.size == 0:
                    continue
                start_ordinal = ordinal_counters[leaf]
                ordinals = np.arange(start_ordinal, start_ordinal + selected_indices.size, dtype=np.int64)
                ordinal_counters[leaf] += selected_indices.size
                split_indices = split_indices_for_ordinals(ordinals, leaf, position_sets)

                selected_table = table.take(pa.array(selected_indices, type=pa.int32()))
                for split, split_local_indices in split_indices.items():
                    if split_local_indices.size == 0:
                        continue
                    final_table = selected_table.take(pa.array(split_local_indices, type=pa.int32()))
                    writers[(split, leaf)].append_table(final_table)
                    output_counts[leaf][split] += final_table.num_rows

            file_row_idx += table.num_rows
            progress.advance(table.num_rows, f"{leaf_prefix} source rewrite")


def route_single_leaf(
    files: list[Path],
    leaf: str,
    position_sets: dict[str, dict[str, set[int]]],
    writers: dict[tuple[str, str], ShardWriter],
    output_counts: dict[str, dict[str, int]],
    batch_size: int,
    progress: ProgressTracker,
) -> None:
    ordinal = 0
    for path in files:
        parquet_file = pq.ParquetFile(path)
        for batch in parquet_file.iter_batches(batch_size=batch_size):
            table = pa.Table.from_batches([batch])
            ordinals = np.arange(ordinal, ordinal + table.num_rows, dtype=np.int64)
            split_indices = split_indices_for_ordinals(ordinals, leaf, position_sets)
            ordinal += table.num_rows
            for split, indices in split_indices.items():
                if indices.size == 0:
                    continue
                selected = table.take(pa.array(indices, type=pa.int32()))
                writers[(split, leaf)].append_table(selected)
                output_counts[leaf][split] += selected.num_rows
            progress.advance(table.num_rows, f"{leaf} source rewrite")


def build_summary(
    output_root: Path,
    text_summary: dict,
    audio_summary: dict,
    totals_by_leaf: dict[str, int],
    planned_counts: dict[str, dict[str, int]],
    actual_counts: dict[str, dict[str, int]],
    warnings: list[str],
    args: argparse.Namespace,
) -> dict:
    return {
        "output_root": str(output_root),
        "data_root": str(args.data_root),
        "text_row_probabilities": str(args.text_row_probs),
        "audio_row_probabilities": str(args.audio_row_probs),
        "threshold": args.threshold,
        "seed": args.seed,
        "ratios": {
            "train": args.train_ratio,
            "val": args.val_ratio,
            "test": args.test_ratio,
        },
        "binary_label_rule": {
            "masc_c_and_qasr_lev": "text LEV >= threshold and audio Levantine >= threshold",
            "masc_c_and_qasr_non_lev": "all remaining rows",
        },
        "provenance": {
            "text_stage": "Text dialect identification was applied to the full cleaned MASC-C and QASR shards.",
            "audio_stage": "Audio dialect identification was then applied to the text-stage LEV >= threshold candidates.",
        },
        "text_scan_summary": text_summary,
        "audio_scan_summary": audio_summary,
        "totals_by_leaf": totals_by_leaf,
        "planned_split_counts": planned_counts,
        "actual_split_counts": actual_counts,
        "warnings": warnings,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Create fresh train/val/test shards with MASC and QASR split into "
            "lev vs non_lev using the audio dialect identification row probabilities."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DATA_ROOT)
    parser.add_argument("--text-row-probs", type=Path, default=DEFAULT_TEXT_ROW_PROBS)
    parser.add_argument("--audio-row-probs", type=Path, default=DEFAULT_AUDIO_ROW_PROBS)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--train-ratio", type=float, default=0.7)
    parser.add_argument("--val-ratio", type=float, default=0.15)
    parser.add_argument("--test-ratio", type=float, default=0.15)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--rows-per-shard", type=int, default=50000)
    parser.add_argument("--batch-size", type=int, default=16384)
    parser.add_argument("--compression", default="snappy")
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Delete the output root first if it already exists.",
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    ratio_sum = args.train_ratio + args.val_ratio + args.test_ratio
    if not math.isclose(ratio_sum, 1.0, rel_tol=0.0, abs_tol=1e-9):
        raise SystemExit(f"Split ratios must sum to 1.0, got {ratio_sum}")
    if args.output_root.exists():
        if not args.overwrite:
            raise SystemExit(
                f"Output root already exists: {args.output_root}. Use --overwrite to replace it."
            )
        shutil.rmtree(args.output_root)


def main() -> None:
    args = parse_args()
    validate_args(args)

    text_summary = collect_text_summary(args.text_row_probs, args.threshold)
    levant_rows_by_file, audio_summary = collect_levant_row_indices(
        args.audio_row_probs,
        args.threshold,
    )
    file_groups, totals_by_leaf, warnings = compute_totals(args.data_root, levant_rows_by_file)
    position_sets, planned_counts = make_position_sets(
        totals_by_leaf,
        args.train_ratio,
        args.val_ratio,
        args.seed,
    )

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
    actual_counts = {
        leaf: {"train": 0, "val": 0, "test": 0}
        for leaf in totals_by_leaf
    }

    try:
        route_binary_source(
            files=file_groups["masc"],
            lev_rows_by_file=levant_rows_by_file,
            leaf_prefix="masc",
            position_sets=position_sets,
            writers=writers,
            output_counts=actual_counts,
            batch_size=args.batch_size,
            progress=progress,
        )
        route_binary_source(
            files=file_groups["qasr"],
            lev_rows_by_file=levant_rows_by_file,
            leaf_prefix="qasr",
            position_sets=position_sets,
            writers=writers,
            output_counts=actual_counts,
            batch_size=args.batch_size,
            progress=progress,
        )

        for leaf in NON_BINARY_LEAVES:
            route_single_leaf(
                files=file_groups[leaf],
                leaf=leaf,
                position_sets=position_sets,
                writers=writers,
                output_counts=actual_counts,
                batch_size=args.batch_size,
                progress=progress,
            )
    finally:
        for writer in writers.values():
            writer.close()
        progress.emit(force=True)

    summary = build_summary(
        output_root=args.output_root,
        text_summary=text_summary,
        audio_summary=audio_summary,
        totals_by_leaf=totals_by_leaf,
        planned_counts=planned_counts,
        actual_counts=actual_counts,
        warnings=warnings,
        args=args,
    )

    reports_dir = args.output_root / "reports"
    ensure_dir(reports_dir)
    (reports_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary["actual_split_counts"], indent=2, ensure_ascii=False))
    print(f"Wrote summary to {reports_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
