#!/usr/bin/env python3
"""Group rows by speaker/recording and classify lev-by-text vs lev-by-audio vs both.

Companion to `pipeline/stages/speaker_aggregate.py`. That stage expects a real
`speaker_id`/`video_id` *column* plus a separate text-dialect
`row_probabilities.jsonl` (the shape a fresh pipeline run produces). This
script instead works directly against `data_lev_custom_split_v1`'s parquet
files, which already carry per-row `text_lev`/`audio_lev` scores inline and
embed the group key inside `uid` (see `r2_speaker_group_disjointness.py` for
the two uid formats this parses).

For each (source, group_key) -- a qasr recording_id or a masc_c video_id --
this averages `text_lev` and `audio_lev` across all of that group's rows,
then buckets the group as one of:

    both        mean(text_lev) >= threshold AND mean(audio_lev) >= threshold
    text_only   mean(text_lev) >= threshold, audio is not
    audio_only  mean(audio_lev) >= threshold, text is not
    neither     neither clears the threshold

Usage:
    export R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_ENDPOINT=...
    python scripts/r2_speaker_group_dialect_classify.py \\
        --bucket-paths \\
            backup/transfer/data/data_lev_custom_split_v1/train.parquet \\
            backup/transfer/data/data_lev_custom_split_v1/val.parquet \\
            backup/transfer/data/data_lev_custom_split_v1/test.parquet \\
        --threshold 0.80 --out speaker_scores.jsonl
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq

from r2_speaker_group_disjointness import extract_group_key


def open_s3() -> "pyarrow.fs.S3FileSystem":
    import pyarrow.fs as fs

    return fs.S3FileSystem(
        access_key=os.environ["R2_ACCESS_KEY_ID"],
        secret_key=os.environ["R2_SECRET_ACCESS_KEY"],
        endpoint_override=os.environ["R2_ENDPOINT"],
        scheme="https",
    )


def read_rows(path: str, local: bool):
    columns = ["uid", "source", "text_lev", "audio_lev", "duration"]
    if local:
        return pq.read_table(path, columns=columns)
    s3 = open_s3()
    with s3.open_input_file(path) as fh:
        return pq.ParquetFile(fh).read(columns=columns)


def load_groups(paths: Iterable[str], local: bool) -> dict[tuple[str, str], list[tuple[float, float, float]]]:
    """(source, group_key) -> list of (text_lev, audio_lev, duration_sec)."""
    groups: dict[tuple[str, str], list[tuple[float, float, float]]] = defaultdict(list)
    for path in paths:
        table = read_rows(path, local)
        d = table.to_pydict()
        n_unresolved = 0
        for uid, source, text_lev, audio_lev, duration in zip(
            d["uid"], d["source"], d["text_lev"], d["audio_lev"], d["duration"]
        ):
            key = extract_group_key(source, uid)
            if key is None:
                n_unresolved += 1
                continue
            groups[(source, key)].append((float(text_lev), float(audio_lev), float(duration)))
        print(f"{path}: {len(d['uid'])} rows, {n_unresolved} unresolved group key(s)", file=sys.stderr)
    return groups


def classify_groups(
    groups: dict[tuple[str, str], list[tuple[float, float, float]]], threshold: float
) -> list[dict]:
    rows_out = []
    for (source, key), items in groups.items():
        n = len(items)
        mean_text = sum(t for t, _, _ in items) / n
        mean_audio = sum(a for _, a, _ in items) / n
        hours = sum(d for _, _, d in items) / 3600.0
        is_text = mean_text >= threshold
        is_audio = mean_audio >= threshold
        label = "both" if is_text and is_audio else "text_only" if is_text else "audio_only" if is_audio else "neither"
        rows_out.append({
            "source": source,
            "group_key": key,
            "row_count": n,
            "mean_text_lev": mean_text,
            "mean_audio_lev": mean_audio,
            "hours": hours,
            "label": label,
        })
    return rows_out


def summarize(rows: list[dict]) -> dict:
    by_label: dict[str, dict] = defaultdict(lambda: {"groups": 0, "hours": 0.0, "by_source": defaultdict(lambda: {"groups": 0, "hours": 0.0})})
    for r in rows:
        b = by_label[r["label"]]
        b["groups"] += 1
        b["hours"] += r["hours"]
        sb = b["by_source"][r["source"]]
        sb["groups"] += 1
        sb["hours"] += r["hours"]
    return {
        label: {
            "groups": v["groups"],
            "hours": round(v["hours"], 2),
            "by_source": {s: {"groups": sv["groups"], "hours": round(sv["hours"], 2)} for s, sv in v["by_source"].items()},
        }
        for label, v in by_label.items()
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket-paths", nargs="+", required=True)
    parser.add_argument("--local", action="store_true", help="paths are local files, not R2 keys")
    parser.add_argument("--threshold", type=float, default=0.80)
    parser.add_argument("--out", type=Path, default=None, help="write per-group JSONL here")
    args = parser.parse_args()

    groups = load_groups(args.bucket_paths, args.local)
    rows = classify_groups(groups, args.threshold)
    summary = summarize(rows)

    print(json.dumps(summary, indent=2))

    if args.out:
        with args.out.open("w", encoding="utf-8") as fh:
            for r in rows:
                fh.write(json.dumps(r, ensure_ascii=False) + "\n")
        print(f"wrote {len(rows)} group row(s) to {args.out}", file=sys.stderr)


if __name__ == "__main__":
    main()
