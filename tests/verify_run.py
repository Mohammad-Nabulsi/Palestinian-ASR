#!/usr/bin/env python3
"""Assert end-to-end invariants over a completed pipeline run.

Checks the things a green log does *not* prove:

- every stage in the manifest reported ``ok``
- the clean stage conserved rows (total == kept + sum of drops)
- the curated root and the split tree hold the same number of rows
- normalization did not blank out transcripts (the failure mode that a corrupted
  Arabic character class produces -- see tests/test_textnorm.py)
- each drop reason that fired actually has shards on disk
- the binary split produced both ``lev`` and ``non_lev`` routing decisions

Usage:  python tests/verify_run.py --run-root .sample_run --work .sample_run/work
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

FAILURES: list[str] = []
CHECKS = 0


def check(condition: bool, label: str, detail: str = "") -> None:
    global CHECKS
    CHECKS += 1
    if condition:
        print(f"PASS {label}")
    else:
        FAILURES.append(f"{label}: {detail}")
        print(f"FAIL {label}: {detail}")


def rows_in(patterns: list[str]) -> int:
    total = 0
    for pattern in patterns:
        for path in glob.glob(pattern, recursive=True):
            total += pq.ParquetFile(path).metadata.num_rows
    return total


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-root", type=Path, default=Path(".sample_run"))
    parser.add_argument("--work", type=Path, default=None)
    parser.add_argument("--split-root", type=Path, default=None)
    args = parser.parse_args()

    run_root = args.run_root.resolve()
    work = (args.work or run_root / "work").resolve()
    split_root = (args.split_root or run_root / "data_curated_levant_binary").resolve()

    manifest_path = run_root / "reports" / "pipeline_manifest.json"
    check(manifest_path.exists(), "manifest exists", str(manifest_path))
    if not manifest_path.exists():
        return 1
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))

    # 1. every stage succeeded
    bad = [s["stage_id"] for s in manifest["stages"] if s["status"] != "ok"]
    check(not bad, "all stages ok", f"failed: {bad}")
    check(len(manifest["stages"]) >= 5, "ran >=5 stages", f"{len(manifest['stages'])}")

    # 2. clean stage conserved rows
    report_path = run_root / "reports" / "clean" / "cleaning_report.json"
    check(report_path.exists(), "clean report exists", str(report_path))
    if report_path.exists():
        totals = json.loads(report_path.read_text(encoding="utf-8"))["totals"]
        dropped = sum(v for k, v in totals.items() if k.startswith("dropped_"))
        check(
            totals["total"] == totals["kept"] + dropped,
            "clean conserves rows",
            f"total={totals['total']} kept={totals['kept']} dropped={dropped}",
        )
        check(totals["kept"] > 0, "clean kept some rows", f"kept={totals['kept']}")

        # 3. each reason that fired has shards on disk
        for key, count in totals.items():
            if not key.startswith("dropped_") or count == 0:
                continue
            reason = key[len("dropped_") :]
            found = glob.glob(str(work / "cleaned" / "dropped" / reason / "*.parquet"))
            check(bool(found), f"dropped/{reason} materialized", f"count={count}, files=0")

    # 4. curated rows == split rows
    curated = rows_in([str(work / "curated" / "*.parquet")])
    split_rows = rows_in([str(split_root / "**" / "*.parquet")])
    check(curated > 0, "curated root non-empty", f"rows={curated}")
    check(
        curated == split_rows,
        "curated rows == split rows",
        f"curated={curated} split={split_rows}",
    )

    # 5. normalization did not blank transcripts
    blanked = 0
    scanned = 0
    for path in glob.glob(str(work / "curated" / "*.parquet")):
        table = pq.read_table(path)
        col = next(
            (c for c in table.column_names if c.startswith("manual_normalized_transcript")),
            None,
        )
        if col is None:
            continue
        values = table.column(col).to_pylist()
        scanned += len(values)
        blanked += sum(1 for v in values if not v or not str(v).strip())
    check(scanned > 0, "found normalized transcripts", f"scanned={scanned}")
    check(
        blanked == 0,
        "no blank normalized transcripts",
        f"{blanked}/{scanned} blank -- normalizer may be corrupted",
    )

    # 6. binary split routed both ways
    lev = glob.glob(str(split_root / "*" / "*" / "lev" / "*.parquet"))
    non_lev = glob.glob(str(split_root / "*" / "*" / "non_lev" / "*.parquet"))
    check(bool(lev), "binary split produced lev shards", "none found")
    check(bool(non_lev), "binary split produced non_lev shards", "none found")

    print()
    if FAILURES:
        print(f"{len(FAILURES)}/{CHECKS} check(s) FAILED")
        for failure in FAILURES:
            print(f"  - {failure}")
        return 1
    print(f"all {CHECKS} checks passed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
