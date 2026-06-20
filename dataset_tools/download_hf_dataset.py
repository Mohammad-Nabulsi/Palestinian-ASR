#!/usr/bin/env python3
"""
Config-aware Hugging Face dataset downloader for large ASR corpora.

Why this script:
- Lets you change only `--dataset` for most datasets.
- Uses a storage strategy that is safer for large audio datasets:
  - non-stream save: Hugging Face Arrow (`save_to_disk`) for full fidelity.
  - optional parquet export (metadata-only by default) for analytics / fast scans.
- Supports both non-stream and stream modes.
- Creates a `levant/` subdirectory for selected Arabic dialect rows
  (Palestinian, Jordanian, Lebanese, Syrian) for Omnilingual/Casablanca-like datasets.
"""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator, List, Optional, Tuple

from datasets import (
    Audio,
    Dataset,
    DatasetDict,
    IterableDataset,
    get_dataset_config_names,
    get_dataset_split_names,
    load_dataset,
)


@dataclass(frozen=True)
class DatasetPreset:
    dataset_id: str
    dialect_filter_hint: bool = False


PRESETS = {
    "facebook/omnilingual-asr-corpus": DatasetPreset(
        dataset_id="facebook/omnilingual-asr-corpus",
        dialect_filter_hint=True,
    ),
    # Placeholder: when you share the exact Casablanca dataset ID, add it here.
}


# We check multiple fields because dataset cards vary in naming conventions.
LEVANT_PATTERNS = [
    re.compile(r"\bpalestin\w*\b", re.IGNORECASE),
    re.compile(r"\bjordan\w*\b", re.IGNORECASE),
    re.compile(r"\bleban\w*\b", re.IGNORECASE),
    re.compile(r"\bsyri\w*\b", re.IGNORECASE),
    # Common Levantine Arabic codes in some corpora.
    re.compile(r"\b(apc|ajp)\b", re.IGNORECASE),
]

LEVANT_FIELDS = [
    "language",
    "language_code",
    "language_name",
    "dialect",
    "dialect_name",
    "locale",
    "iso_639_3",
    "raw_text",
]


def slugify_dataset_id(dataset_id: str) -> str:
    return dataset_id.replace("/", "__")


def ensure_audio_not_decoded(ds: Dataset) -> Dataset:
    if "audio" in ds.column_names:
        return ds.cast_column("audio", Audio(decode=False))
    return ds


def drop_audio_column(ds: Dataset) -> Dataset:
    if "audio" in ds.column_names:
        return ds.remove_columns(["audio"])
    return ds


def write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def row_matches_levant(row: dict) -> bool:
    values: List[str] = []
    for field in LEVANT_FIELDS:
        if field in row and row[field] is not None:
            values.append(str(row[field]))
    text = " | ".join(values)
    return any(p.search(text) for p in LEVANT_PATTERNS)


def parquet_path(base: Path, split: str, part_idx: Optional[int] = None) -> Path:
    if part_idx is None:
        return base / f"{split}.parquet"
    return base / f"{split}.part-{part_idx:05d}.parquet"


def save_dataset_to_disk_and_parquet(
    ds: Dataset,
    out_root: Path,
    split: str,
    export_parquet: bool,
    parquet_include_audio: bool,
) -> None:
    split_root = out_root / "non_stream" / split
    split_root.mkdir(parents=True, exist_ok=True)

    # Best fidelity for large audio data.
    ds.save_to_disk(str(split_root / "hf_arrow"))

    if export_parquet:
        pq_dir = split_root / "parquet"
        pq_dir.mkdir(parents=True, exist_ok=True)
        ds_for_parquet = ds if parquet_include_audio else drop_audio_column(ds)
        ds_for_parquet.to_parquet(str(parquet_path(pq_dir, split)))


def save_levant_subset_non_stream(
    ds: Dataset,
    out_root: Path,
    split: str,
    parquet_include_audio: bool,
) -> None:
    if not ds.column_names:
        return
    subset = ds.filter(row_matches_levant)
    if subset.num_rows == 0:
        return

    split_root = out_root / "levant" / "non_stream" / split
    split_root.mkdir(parents=True, exist_ok=True)
    subset.save_to_disk(str(split_root / "hf_arrow"))
    subset_for_parquet = subset if parquet_include_audio else drop_audio_column(subset)
    subset_for_parquet.to_parquet(str(parquet_path(split_root, split)))


def stream_batches(
    stream_ds: IterableDataset,
    batch_size: int,
) -> Iterator[List[dict]]:
    batch: List[dict] = []
    for row in stream_ds:
        batch.append(row)
        if len(batch) >= batch_size:
            yield batch
            batch = []
    if batch:
        yield batch


def write_stream_parquet(
    stream_ds: IterableDataset,
    out_dir: Path,
    split: str,
    batch_size: int,
    max_rows: Optional[int],
    stream_include_audio: bool,
    levant_only: bool = False,
) -> Tuple[int, int]:
    out_dir.mkdir(parents=True, exist_ok=True)

    rows_written = 0
    parts_written = 0
    emitted = 0

    for batch in stream_batches(stream_ds, batch_size=batch_size):
        if levant_only:
            batch = [row for row in batch if row_matches_levant(row)]
            if not batch:
                continue

        if max_rows is not None:
            remaining = max_rows - emitted
            if remaining <= 0:
                break
            batch = batch[:remaining]

        if not batch:
            continue

        if not stream_include_audio:
            for row in batch:
                row.pop("audio", None)

        part_ds = Dataset.from_list(batch)
        part_path = parquet_path(out_dir, split=split, part_idx=parts_written)
        part_ds.to_parquet(str(part_path))

        rows_written += len(batch)
        parts_written += 1
        emitted += len(batch)

        if max_rows is not None and emitted >= max_rows:
            break

    return rows_written, parts_written


def download_non_stream(
    dataset_id: str,
    config: Optional[str],
    split: Optional[str],
    out_root: Path,
    token: Optional[str],
    export_parquet: bool,
    parquet_include_audio: bool,
    levant_hint: bool,
) -> dict:
    ds_obj = load_dataset(dataset_id, name=config, split=split, token=token, streaming=False)
    stats = {
        "mode": "non_stream",
        "split_rows": {},
    }

    if isinstance(ds_obj, DatasetDict):
        items = list(ds_obj.items())
    else:
        resolved_split = split or "train"
        items = [(resolved_split, ds_obj)]

    for split_name, ds in items:
        ds = ensure_audio_not_decoded(ds)
        save_dataset_to_disk_and_parquet(
            ds=ds,
            out_root=out_root,
            split=split_name,
            export_parquet=export_parquet,
            parquet_include_audio=parquet_include_audio,
        )
        stats["split_rows"][split_name] = ds.num_rows

        if levant_hint:
            save_levant_subset_non_stream(
                ds,
                out_root=out_root,
                split=split_name,
                parquet_include_audio=parquet_include_audio,
            )

    return stats


def download_stream(
    dataset_id: str,
    config: Optional[str],
    split: str,
    out_root: Path,
    token: Optional[str],
    batch_size: int,
    max_stream_rows: Optional[int],
    stream_include_audio: bool,
    levant_hint: bool,
) -> dict:
    stream_ds = load_dataset(dataset_id, name=config, split=split, token=token, streaming=True)

    parquet_dir = out_root / "stream" / split / "parquet"
    rows, parts = write_stream_parquet(
        stream_ds=stream_ds,
        out_dir=parquet_dir,
        split=split,
        batch_size=batch_size,
        max_rows=max_stream_rows,
        stream_include_audio=stream_include_audio,
        levant_only=False,
    )

    stats = {
        "mode": "stream",
        "split": split,
        "rows_written": rows,
        "parts_written": parts,
    }

    if levant_hint:
        levant_ds = load_dataset(dataset_id, name=config, split=split, token=token, streaming=True)
        levant_dir = out_root / "levant" / "stream" / split / "parquet"
        lev_rows, lev_parts = write_stream_parquet(
            stream_ds=levant_ds,
            out_dir=levant_dir,
            split=split,
            batch_size=batch_size,
            max_rows=max_stream_rows,
            stream_include_audio=stream_include_audio,
            levant_only=True,
        )
        stats["levant_rows_written"] = lev_rows
        stats["levant_parts_written"] = lev_parts

    return stats


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Download HF datasets with robust storage for large ASR corpora."
    )
    parser.add_argument("--dataset", required=True, help="Dataset id. Example: facebook/omnilingual-asr-corpus")
    parser.add_argument("--config", default=None, help="Optional config/subset name.")
    parser.add_argument(
        "--split",
        default=None,
        help="Optional split. If omitted in non-stream mode, all splits are downloaded when available.",
    )
    parser.add_argument(
        "--mode",
        choices=["non_stream", "stream", "both"],
        default="both",
        help="Download mode.",
    )
    parser.add_argument(
        "--output_dir",
        default="./datasets_storage",
        help="Root output directory.",
    )
    parser.add_argument(
        "--export_parquet",
        action="store_true",
        help="Also export non-stream splits to parquet (in addition to HF Arrow).",
    )
    parser.add_argument(
        "--parquet_include_audio",
        action="store_true",
        help="Include `audio` column in non-stream parquet export (larger files).",
    )
    parser.add_argument(
        "--stream_include_audio",
        action="store_true",
        help="Include `audio` column in streaming parquet shards (larger files).",
    )
    parser.add_argument(
        "--stream_batch_size",
        type=int,
        default=2000,
        help="Rows per parquet shard in streaming mode.",
    )
    parser.add_argument(
        "--max_stream_rows",
        type=int,
        default=None,
        help="Optional cap for streaming rows per split (for sampling/debug).",
    )
    parser.add_argument(
        "--token",
        default=None,
        help="HF token for gated/private datasets (or set HF_TOKEN env var externally).",
    )
    parser.add_argument(
        "--inspect_only",
        action="store_true",
        help="Only print available configs and split names (no download).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.inspect_only:
        config_names = get_dataset_config_names(args.dataset, token=args.token)
        if not config_names:
            config_names = [None]

        split_map = {}
        for cfg in config_names:
            cfg_key = cfg if cfg is not None else "default"
            split_map[cfg_key] = get_dataset_split_names(args.dataset, cfg, token=args.token)

        payload = {
            "dataset": args.dataset,
            "configs": [c if c is not None else "default" for c in config_names],
            "splits_by_config": split_map,
        }
        print(json.dumps(payload, indent=2, ensure_ascii=False))
        return

    preset = PRESETS.get(args.dataset)

    dataset_slug = slugify_dataset_id(args.dataset)
    root = Path(args.output_dir) / dataset_slug
    root.mkdir(parents=True, exist_ok=True)

    # We only auto-enable Levant routing for known presets (Omnilingual now).
    levant_hint = bool(preset and preset.dialect_filter_hint)

    metadata = {
        "dataset": args.dataset,
        "config": args.config,
        "requested_split": args.split,
        "mode": args.mode,
        "output_root": str(root.resolve()),
        "export_parquet": args.export_parquet,
        "parquet_include_audio": args.parquet_include_audio,
        "stream_include_audio": args.stream_include_audio,
        "stream_batch_size": args.stream_batch_size,
        "max_stream_rows": args.max_stream_rows,
        "levant_filter_enabled": levant_hint,
        "runs": [],
    }

    if args.mode in ("non_stream", "both"):
        non_stream_stats = download_non_stream(
            dataset_id=args.dataset,
            config=args.config,
            split=args.split,
            out_root=root,
            token=args.token,
            export_parquet=args.export_parquet,
            parquet_include_audio=args.parquet_include_audio,
            levant_hint=levant_hint,
        )
        metadata["runs"].append(non_stream_stats)

    if args.mode in ("stream", "both"):
        split_for_stream = args.split or "train"
        stream_stats = download_stream(
            dataset_id=args.dataset,
            config=args.config,
            split=split_for_stream,
            out_root=root,
            token=args.token,
            batch_size=args.stream_batch_size,
            max_stream_rows=args.max_stream_rows,
            stream_include_audio=args.stream_include_audio,
            levant_hint=levant_hint,
        )
        metadata["runs"].append(stream_stats)

    write_json(root / "download_report.json", metadata)
    print(json.dumps(metadata, indent=2, ensure_ascii=False))
    print(f"\nDone. Data saved under: {root}")


if __name__ == "__main__":
    main()
