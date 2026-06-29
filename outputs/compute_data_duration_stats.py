from __future__ import annotations

import argparse
import json
import os
import wave
from pathlib import Path

import numpy as np
import pyarrow.dataset as ds
from datasets import load_from_disk


def summarize(values: list[float]) -> dict[str, float | int | None]:
    durations = np.asarray(values, dtype=np.float64)
    if durations.size == 0:
        return {
            "count": 0,
            "mean_seconds": None,
            "median_seconds": None,
            "p95_seconds": None,
        }

    return {
        "count": int(durations.size),
        "mean_seconds": float(durations.mean()),
        "median_seconds": float(np.median(durations)),
        "p95_seconds": float(np.percentile(durations, 95)),
    }


def collect_layla_durations(source_root: Path) -> list[float]:
    durations: list[float] = []
    for root, _, files in os.walk(source_root):
        for name in files:
            if not name.lower().endswith(".wav"):
                continue
            audio_path = Path(root) / name
            with wave.open(str(audio_path), "rb") as wav_file:
                duration = wav_file.getnframes() / float(wav_file.getframerate())
            durations.append(duration)
    return durations


def collect_parquet_durations(source_root: Path) -> list[float]:
    dataset = ds.dataset(str(source_root), format="parquet")
    durations: list[float] = []
    for batch in dataset.to_batches(columns=["duration"]):
        durations.extend(float(value) for value in batch.column(0).to_pylist() if value is not None)
    return durations


def collect_omnilingual_durations(source_root: Path) -> list[float]:
    dataset = load_from_disk(str(source_root))
    return [float(value) for value in dataset["duration"] if value is not None]


def collect_qasr_durations(source_root: Path) -> list[float]:
    index_path = source_root / "index" / "segment_index.jsonl"
    durations: list[float] = []
    with index_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            row = json.loads(line)
            durations.append(float(row["duration_seconds"]))
    return durations


def build_source_map(data_root: Path) -> dict[str, tuple[Path, callable]]:
    return {
        "casablanca_jordanian": (data_root / "casablanca_jordanian", collect_parquet_durations),
        "casablanca_palestinian": (data_root / "casablanca_palestinian", collect_parquet_durations),
        "layla": (data_root / "layla", collect_layla_durations),
        "masc_c_only": (data_root / "masc_c_only" / "data", collect_parquet_durations),
        "omnilingual_apc": (data_root / "omnilingual_apc", collect_omnilingual_durations),
        "processed_qasr_segments": (data_root / "processed_qasr_segments", collect_qasr_durations),
    }


def format_seconds(value: float | None) -> str:
    if value is None:
        return "n/a"
    return f"{value:.2f}"


def print_table(results: dict[str, dict[str, float | int | None]]) -> None:
    header = (
        f"{'source':<24} {'count':>10} {'mean_s':>10} {'median_s':>10} {'p95_s':>10}"
    )
    print(header)
    print("-" * len(header))
    for source, stats in results.items():
        print(
            f"{source:<24} "
            f"{stats['count']:>10} "
            f"{format_seconds(stats['mean_seconds']):>10} "
            f"{format_seconds(stats['median_seconds']):>10} "
            f"{format_seconds(stats['p95_seconds']):>10}"
        )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compute average, median, and 95th percentile audio duration for each source under data/."
    )
    parser.add_argument(
        "--data-root",
        type=Path,
        default=Path("data"),
        help="Root data directory that contains the source folders and symlinks.",
    )
    parser.add_argument(
        "--json-output",
        type=Path,
        default=None,
        help="Optional path to write the full results as JSON.",
    )
    parser.add_argument(
        "--only",
        nargs="*",
        default=None,
        help="Optional subset of source names to compute.",
    )
    args = parser.parse_args()

    source_map = build_source_map(args.data_root.resolve())
    selected_sources = args.only or list(source_map.keys())

    results: dict[str, dict[str, float | int | None]] = {}
    for source_name in selected_sources:
        if source_name not in source_map:
            raise SystemExit(f"Unknown source: {source_name}")

        source_root, collector = source_map[source_name]
        if not source_root.exists():
            raise SystemExit(f"Missing source path for {source_name}: {source_root}")

        print(f"Computing {source_name} from {source_root}...")
        durations = collector(source_root)
        results[source_name] = summarize(durations)

    print()
    print_table(results)

    if args.json_output is not None:
        args.json_output.parent.mkdir(parents=True, exist_ok=True)
        args.json_output.write_text(json.dumps(results, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        print()
        print(f"Wrote JSON results to {args.json_output}")


if __name__ == "__main__":
    main()
