#!/usr/bin/env python3
"""Build the final reports/summary.json for data_curated_levant_binary_v1 by
scanning the actual output parquet files on disk (ground truth), instead of
trusting each worker's in-memory row counters -- those are known to
undercount whenever a worker was killed and resumed mid-run (the resumed
process's own counters start at zero and have no memory of rows a prior,
now-exited process already committed safely to disk).
"""
import glob
import json
import sys
from pathlib import Path

import pyarrow.parquet as pq

sys.path.insert(0, "/root/Palestinian-ASR/scripts")
from rebuild_levant_binary_parallel_resumable import (  # noqa: E402
    collect_text_summary,
    collect_levant_row_indices,
)

OUT = Path("/workspace/asr/Palestinian-ASR/data_curated_levant_binary_v1")
SPLITS = ("train", "val", "test")
LEAVES = ("masc/lev", "masc/non_lev", "qasr/lev", "qasr/non_lev", "omni", "layla", "casa/pal", "casa/jor")

TEXT_ROW_PROBS = Path(".logs/combined_row_probs/text_row_probabilities.jsonl")
AUDIO_ROW_PROBS = Path(".logs/combined_row_probs/audio_row_probabilities.jsonl")


def count_rows_in_dir(d: Path) -> int:
    files = sorted(glob.glob(str(d / "data-*.parquet")))
    return sum(pq.ParquetFile(f).metadata.num_rows for f in files)


def main() -> None:
    actual_split_counts = {}
    totals_by_leaf = {}
    for leaf in LEAVES:
        counts = {}
        for split in SPLITS:
            counts[split] = count_rows_in_dir(OUT / split / Path(leaf))
        actual_split_counts[leaf] = counts
        totals_by_leaf[leaf] = sum(counts.values())

    text_summary = collect_text_summary(TEXT_ROW_PROBS, 0.8)
    _, audio_summary = collect_levant_row_indices(AUDIO_ROW_PROBS, 0.8)

    summary = {
        "output_root": str(OUT),
        "data_root": ".logs/full_combined_data_root",
        "text_row_probabilities": str(TEXT_ROW_PROBS),
        "audio_row_probabilities": str(AUDIO_ROW_PROBS),
        "threshold": 0.8,
        "seed": 42,
        "unchanged_leaves_ratios": {"train": 0.7, "val": 0.15, "test": 0.15},
        "native_test_leaves": ["casa/jor", "casa/pal", "masc"],
        "native_test_leaves_remainder_ratios": {"train": 0.8, "val": 0.2},
        "binary_label_rule": {
            "masc_c_and_qasr_lev": "text LEV >= threshold and audio Levantine >= threshold",
            "masc_c_and_qasr_non_lev": "all remaining rows",
        },
        "provenance_note": (
            "actual_split_counts computed by scanning real output parquet files on disk "
            "(ground truth), not from in-process worker counters, because a resumed "
            "worker's counters cannot see rows a prior process already committed."
        ),
        "text_scan_summary": text_summary,
        "audio_scan_summary": audio_summary,
        "totals_by_leaf": totals_by_leaf,
        "actual_split_counts": actual_split_counts,
    }

    reports_dir = OUT / "reports"
    (reports_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(json.dumps({"totals_by_leaf": totals_by_leaf, "actual_split_counts": actual_split_counts}, indent=2, ensure_ascii=False))
    print(f"Wrote ground-truth summary to {reports_dir / 'summary.json'}")


if __name__ == "__main__":
    main()
