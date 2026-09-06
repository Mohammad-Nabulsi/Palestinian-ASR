#!/usr/bin/env python3
"""Bottom-up duration/row statistics for data_curated_levant_binary_v1.

Reports at increasing levels of aggregation:
  1. leaf-within-split  (smallest unit on disk, e.g. train/masc/lev)
  2. leaf-across-splits (e.g. masc/lev: train+val+test)
  3. source-across-splits (e.g. masc: lev+non_lev combined, all splits)
  4. split-across-sources (e.g. train: every leaf combined)
  5. grand total

Only reads the `duration` column (no audio decode), so it's fast even at
scale.
"""
import glob
import json
from collections import defaultdict
from pathlib import Path

import pyarrow.parquet as pq

OUT = Path("/workspace/asr/Palestinian-ASR/data_curated_levant_binary_v1")
SPLITS = ("train", "val", "test")
LEAVES = ("masc/lev", "masc/non_lev", "qasr/lev", "qasr/non_lev", "omni", "layla", "casa/pal", "casa/jor")
# leaf -> (source group, sub-label or None)
LEAF_TO_SOURCE = {
    "masc/lev": ("masc", "lev"),
    "masc/non_lev": ("masc", "non_lev"),
    "qasr/lev": ("qasr", "lev"),
    "qasr/non_lev": ("qasr", "non_lev"),
    "omni": ("omni", None),
    "layla": ("layla", None),
    "casa/pal": ("casa", "pal"),
    "casa/jor": ("casa", "jor"),
}


def leaf_stats(split: str, leaf: str) -> dict:
    files = sorted(glob.glob(str(OUT / split / leaf / "data-*.parquet")))
    rows = 0
    total_seconds = 0.0
    for f in files:
        table = pq.read_table(f, columns=["duration"])
        durations = table.column("duration").to_pylist()
        rows += len(durations)
        total_seconds += sum(d for d in durations if d is not None)
    return {"shards": len(files), "rows": rows, "hours": total_seconds / 3600.0}


def merge(*stats_list: dict) -> dict:
    return {
        "shards": sum(s["shards"] for s in stats_list),
        "rows": sum(s["rows"] for s in stats_list),
        "hours": sum(s["hours"] for s in stats_list),
    }


def fmt(s: dict) -> str:
    return f"{s['rows']:>10,} rows | {s['hours']:>9.2f} h | {s['shards']:>4} shards"


def main() -> None:
    level1 = {}  # (split, leaf) -> stats
    for split in SPLITS:
        for leaf in LEAVES:
            level1[(split, leaf)] = leaf_stats(split, leaf)

    print("=" * 70)
    print("LEVEL 1 — leaf within split (smallest unit)")
    print("=" * 70)
    for split in SPLITS:
        for leaf in LEAVES:
            print(f"{split}/{leaf:<14} {fmt(level1[(split, leaf)])}")

    level2 = {}  # leaf -> stats (across splits)
    for leaf in LEAVES:
        level2[leaf] = merge(*(level1[(s, leaf)] for s in SPLITS))

    print()
    print("=" * 70)
    print("LEVEL 2 — leaf across all splits")
    print("=" * 70)
    for leaf in LEAVES:
        print(f"{leaf:<18} {fmt(level2[leaf])}")

    level3 = defaultdict(list)  # source -> [leaf stats]
    for leaf, (source, _) in LEAF_TO_SOURCE.items():
        level3[source].append(level2[leaf])
    level3_merged = {source: merge(*stats_list) for source, stats_list in level3.items()}

    print()
    print("=" * 70)
    print("LEVEL 3 — source across all splits (lev+non_lev / pal+jor combined)")
    print("=" * 70)
    for source, stats in level3_merged.items():
        print(f"{source:<18} {fmt(stats)}")

    level4 = {}  # split -> stats (across all leaves)
    for split in SPLITS:
        level4[split] = merge(*(level1[(split, leaf)] for leaf in LEAVES))

    print()
    print("=" * 70)
    print("LEVEL 4 — split across all sources")
    print("=" * 70)
    for split in SPLITS:
        print(f"{split:<18} {fmt(level4[split])}")

    grand = merge(*level4.values())
    print()
    print("=" * 70)
    print("LEVEL 5 — grand total")
    print("=" * 70)
    print(f"{'data_curated_levant_binary_v1':<18} {fmt(grand)}")

    report = {
        "level1_leaf_within_split": {f"{s}/{l}": level1[(s, l)] for s in SPLITS for l in LEAVES},
        "level2_leaf_across_splits": level2,
        "level3_source_across_splits": level3_merged,
        "level4_split_across_sources": level4,
        "level5_grand_total": grand,
    }
    out_path = OUT / "reports" / "duration_stats.json"
    out_path.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"\nWrote {out_path}")


if __name__ == "__main__":
    main()
