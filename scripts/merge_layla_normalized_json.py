#!/usr/bin/env python3
"""Merge the 6 hand-normalized Layla batches into one `normalized_all.json`.

The normalization pass for the Layla Witheeb corpus was done in 6 separate
rounds (see `Layla/normalized_json/README.md` for provenance), each a
standalone `{"records": [...], "word_conversions": {...}}` file. This merges
their `records` into a single file so downstream consumers (`configs/full.yaml`,
`scripts/build_layla_shards.py`) only need to list one path, and fails loudly
if the 218-source set this was verified against ever regresses (duplicate or
missing source, mismatched total).

Re-run whenever a new normalization batch is added to `Layla/normalized_json/`.
"""
from __future__ import annotations

import json
from pathlib import Path

NORMALIZED_JSON_DIR = Path("/workspace/asr/Palestinian-ASR/Layla/normalized_json")
OUTPUT_PATH = NORMALIZED_JSON_DIR / "normalized_all.json"
EXPECTED_TOTAL = 218

# Explicit list (not a glob) so a stray file dropped into the directory can't
# silently change what gets merged.
INPUT_FILES = [
    "normalized_output_appended.json",
    "normalized_layla_batch_130_appended_131.json",
    "normalized_pasted_132_133_appended.json",
    "normalized_pasted_text_134_135_136_appended.json",
    "normalized_batch_137_appended.json",
    "normalized_layla_missing_22.json",
]


def main() -> None:
    merged: dict[str, dict] = {}
    per_file_counts = {}

    for name in INPUT_FILES:
        path = NORMALIZED_JSON_DIR / name
        obj = json.loads(path.read_text(encoding="utf-8"))
        records = obj.get("records", [])
        per_file_counts[name] = len(records)
        for record in records:
            source = record.get("source")
            if not source:
                raise SystemExit(f"{name}: record missing 'source': {record}")
            key = source.replace("./", "")
            if key in merged:
                raise SystemExit(
                    f"Duplicate source across normalization batches: {key} "
                    f"(already loaded from an earlier file, seen again in {name})"
                )
            merged[key] = {
                "source": source,
                "original": record.get("original", ""),
                "normalized": record.get("normalized", ""),
            }

    if len(merged) != EXPECTED_TOTAL:
        raise SystemExit(
            f"Expected {EXPECTED_TOTAL} merged records, got {len(merged)}. "
            f"Per-file counts: {per_file_counts}"
        )

    out = {
        "records": [merged[key] for key in sorted(merged)],
        "source_files": INPUT_FILES,
        "total_records": len(merged),
    }
    OUTPUT_PATH.write_text(
        json.dumps(out, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    print(f"Merged {len(merged)} records from {len(INPUT_FILES)} files -> {OUTPUT_PATH}")
    for name, count in per_file_counts.items():
        print(f"  {name}: {count}")


if __name__ == "__main__":
    main()
