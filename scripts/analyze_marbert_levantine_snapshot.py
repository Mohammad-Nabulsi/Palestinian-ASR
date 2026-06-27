#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
import random
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class SnapshotStats:
    scanned_rows: int = 0
    qualifying_rows: int = 0
    malformed_lines: int = 0


def safe_jsonl_iter(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line_no, line in enumerate(f, start=1):
            line = line.strip()
            if not line:
                continue
            try:
                yield line_no, json.loads(line)
            except Exception:
                yield line_no, None


def reservoir_add(reservoir: List[Dict[str, Any]], item: Dict[str, Any], seen: int, sample_size: int, rng: random.Random) -> None:
    if sample_size <= 0:
        return
    if len(reservoir) < sample_size:
        reservoir.append(item)
        return
    idx = rng.randint(0, seen - 1)
    if idx < sample_size:
        reservoir[idx] = item


def load_optional_json(path: Path) -> Optional[Dict[str, Any]]:
    if not path.exists():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def build_sample_record(rec: Dict[str, Any], threshold: float) -> Dict[str, Any]:
    label_scores = rec.get("label_scores") or {}
    text = rec.get("text") or ""
    return {
        "uid": rec.get("uid"),
        "source_file": rec.get("source_file"),
        "row_idx": rec.get("row_idx"),
        "text_field": rec.get("text_field"),
        "top_label": rec.get("top_label"),
        "top_score": rec.get("top_score"),
        "final_label": rec.get("final_label"),
        "lev_score": label_scores.get("LEV"),
        "meets_lev_threshold": (label_scores.get("LEV") or 0.0) >= threshold,
        "text": text,
        "text_preview": text[:220],
    }


def main() -> int:
    ap = argparse.ArgumentParser(description="Snapshot LEV-threshold stats from MARBERT probability logs")
    ap.add_argument("--input", required=True, help="Path to row_probabilities.jsonl")
    ap.add_argument("--progress-json", default=None, help="Optional path to progress_latest.json")
    ap.add_argument("--threshold", type=float, default=0.85)
    ap.add_argument("--sample-count", type=int, default=10)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--output-dir", required=True, help="Directory to write snapshot artifacts into")
    args = ap.parse_args()

    input_path = Path(args.input)
    progress_path = Path(args.progress_json) if args.progress_json else None
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rng = random.Random(args.seed)
    stats = SnapshotStats()
    qualifying_samples: List[Dict[str, Any]] = []

    for line_no, rec in safe_jsonl_iter(input_path):
        if rec is None:
            stats.malformed_lines += 1
            continue
        stats.scanned_rows += 1
        lev_score = (rec.get("label_scores") or {}).get("LEV")
        if lev_score is None:
            continue
        if float(lev_score) >= args.threshold:
            stats.qualifying_rows += 1
            sample = build_sample_record(rec, args.threshold)
            sample["line_no"] = line_no
            reservoir_add(qualifying_samples, sample, stats.qualifying_rows, args.sample_count, rng)

    qualifying_percentage = 0.0 if stats.scanned_rows == 0 else round(100.0 * stats.qualifying_rows / stats.scanned_rows, 4)
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    progress_snapshot = load_optional_json(progress_path) if progress_path else None

    summary = {
        "snapshot_timestamp_utc": timestamp,
        "input_path": str(input_path.resolve()),
        "input_size_bytes": input_path.stat().st_size if input_path.exists() else None,
        "threshold_label": "LEV",
        "threshold": args.threshold,
        "scanned_rows": stats.scanned_rows,
        "qualifying_rows": stats.qualifying_rows,
        "qualifying_percentage": qualifying_percentage,
        "sample_count_requested": args.sample_count,
        "sample_count_saved": len(qualifying_samples),
        "random_seed": args.seed,
        "malformed_lines_skipped": stats.malformed_lines,
        "progress_snapshot": progress_snapshot,
    }

    summary_path = output_dir / "summary.json"
    jsonl_path = output_dir / "random_samples.jsonl"
    csv_path = output_dir / "random_samples.csv"

    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    with jsonl_path.open("w", encoding="utf-8") as f:
        for rec in qualifying_samples:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    with csv_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "uid",
                "lev_score",
                "top_label",
                "top_score",
                "final_label",
                "source_file",
                "row_idx",
                "text_field",
                "line_no",
                "text_preview",
            ],
        )
        writer.writeheader()
        for rec in qualifying_samples:
            writer.writerow({k: rec.get(k) for k in writer.fieldnames})

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    print(f"summary_path={summary_path}")
    print(f"random_samples_jsonl={jsonl_path}")
    print(f"random_samples_csv={csv_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
