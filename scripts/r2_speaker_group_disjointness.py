#!/usr/bin/env python3
"""Check whether a set of parquet splits are speaker/recording-disjoint.

Written during the omni_tune data-audit session to answer: does the same
qasr recording or masc_c video have rows scattered across more than one of
train/val/test? Neither source carries a `speaker_id`/`video_id` *column* in
`data_lev_custom_split_v1` -- both are embedded inside the `uid` string in one
of two formats that coexist in that dataset:

    qasr:  "<shard>:uid=qasr:<recording_id>:<segment_id>"   (verbose)
           "qasr:<recording_id>:<segment_id>"                (short)
    masc:  "<shard>:video_id=<video_id>"                     (verbose)
           "masc_c:<video_id>:<row_idx>"                     (short)

This grabs `uid`/`source` (and optionally `duration`) via pyarrow's S3
filesystem with column projection -- no full-file downloads needed for plain
parquet. `curated_corpus`'s `*.parquet.zst` shards are NOT readable this way
(they're a single zstd stream wrapping the whole file, not per-page parquet
compression) -- download + `unzstd` those first, then point this script at
the decompressed `.parquet` files with `--local`.

Usage:
    export R2_ACCESS_KEY_ID=... R2_SECRET_ACCESS_KEY=... R2_ENDPOINT=...
    python scripts/r2_speaker_group_disjointness.py \\
        --bucket-paths \\
            backup/transfer/data/data_lev_custom_split_v1/train.parquet \\
            backup/transfer/data/data_lev_custom_split_v1/val.parquet \\
            backup/transfer/data/data_lev_custom_split_v1/test.parquet \\
        --labels train val test
"""
from __future__ import annotations

import argparse
import json
import os
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

import pyarrow.parquet as pq

QASR_LONG_RE = re.compile(r"uid=qasr:([0-9A-Fa-f-]+):")
QASR_SHORT_RE = re.compile(r"^qasr:([0-9A-Fa-f-]+):")
MASC_LONG_RE = re.compile(r"video_id=([^:]+)$")
MASC_SHORT_RE = re.compile(r"^masc_c:([^:]+):")


def extract_group_key(source: str, uid: str) -> str | None:
    """recording_id for qasr, video_id for masc_c -- the closest available
    speaker proxy for each (qasr has no real speaker_id in this dataset;
    masc_c never has one at all, see R2_BUCKET_USAGE.md)."""
    if source == "qasr":
        m = QASR_LONG_RE.search(uid) or QASR_SHORT_RE.search(uid)
        return m.group(1) if m else None
    if source == "masc_c":
        m = MASC_LONG_RE.search(uid) or MASC_SHORT_RE.search(uid)
        return m.group(1) if m else None
    return None


def open_s3() -> "pyarrow.fs.S3FileSystem":
    import pyarrow.fs as fs

    return fs.S3FileSystem(
        access_key=os.environ["R2_ACCESS_KEY_ID"],
        secret_key=os.environ["R2_SECRET_ACCESS_KEY"],
        endpoint_override=os.environ["R2_ENDPOINT"],
        scheme="https",
    )


def read_columns(path: str, columns: list[str], local: bool):
    if local:
        return pq.read_table(path, columns=columns)
    s3 = open_s3()
    with s3.open_input_file(path) as fh:
        return pq.ParquetFile(fh).read(columns=columns)


def load_groups(paths: Iterable[str], labels: Iterable[str], local: bool) -> dict[tuple[str, str], set[str]]:
    """(source, group_key) -> set of split labels its rows appear in."""
    groups: dict[tuple[str, str], set[str]] = defaultdict(set)
    for path, label in zip(paths, labels):
        table = read_columns(path, ["uid", "source"], local)
        d = table.to_pydict()
        n_unresolved = 0
        for uid, source in zip(d["uid"], d["source"]):
            key = extract_group_key(source, uid)
            if key is None:
                n_unresolved += 1
                continue
            groups[(source, key)].add(label)
        print(f"{label}: {len(d['uid'])} rows, {n_unresolved} unresolved group key(s)", file=sys.stderr)
    return groups


def report_overlap(groups: dict[tuple[str, str], set[str]]) -> dict:
    by_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    straddling: dict[str, int] = defaultdict(int)
    for (source, _key), labels in groups.items():
        by_source[source]["total_groups"] += 1
        if len(labels) > 1:
            straddling[source] += 1
    summary = {
        "total_groups": len(groups),
        "by_source": {s: dict(v) for s, v in by_source.items()},
        "groups_straddling_multiple_splits": dict(straddling),
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--bucket-paths", nargs="+", required=True, help="parquet paths, one per split")
    parser.add_argument("--labels", nargs="+", required=True, help="split label per path, same order/length")
    parser.add_argument("--local", action="store_true", help="paths are local files, not R2 keys")
    parser.add_argument("--out", type=Path, default=None, help="write JSON summary here (default: stdout)")
    args = parser.parse_args()

    if len(args.bucket_paths) != len(args.labels):
        parser.error("--bucket-paths and --labels must have the same length")

    groups = load_groups(args.bucket_paths, args.labels, args.local)
    summary = report_overlap(groups)
    text = json.dumps(summary, indent=2)
    if args.out:
        args.out.write_text(text + "\n")
    else:
        print(text)


if __name__ == "__main__":
    main()
