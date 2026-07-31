#!/usr/bin/env python3
"""Export Layla transcripts in the input shape the normalization prompt expects.

The four hand-merged `normalized_*.json` files described in DATA_CURATION.md
("Layla Prompt Merge and Sharding") are gone from this box. This script
regenerates their *input* side: the raw Arabic transcript text keyed by the same
`source` values, batched into files small enough to paste into one prompt turn.

Feed each batch to the prompt in DATA_CURATION.md's "Layla Transcript Cleaning
Prompt" section, collect the JSON replies as `normalized_*.json`, then re-run:

    scripts/build_layla_shards.py --normalized-json <files...>

which will use the `normalized` text instead of the raw docx text.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from build_layla_shards import (
    DEFAULT_DATASET_ROOT,
    TRANSCRIPT_SUFFIX_RE,
    extract_docx_text,
    staged_source_value,
)

DEFAULT_OUTPUT_DIR = Path("/workspace/asr/Palestinian-ASR/Layla/normalization_input")
DEFAULT_BATCH_CHARS = 12_000


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--batch-chars",
        type=int,
        default=DEFAULT_BATCH_CHARS,
        help="Approximate transcript characters per batch file.",
    )
    parser.add_argument(
        "--only-sources",
        type=Path,
        help=(
            "JSON file holding a list of `source` values; export only those. "
            "Use it to re-prompt just the transcripts a previous round missed."
        ),
    )
    args = parser.parse_args()

    docx_paths = sorted(
        p for p in args.dataset_root.rglob("*.docx") if TRANSCRIPT_SUFFIX_RE.search(p.stem)
    )
    if not docx_paths:
        raise SystemExit(f"No *_Arabic_transcription.docx found under {args.dataset_root}")

    if args.only_sources:
        wanted = {
            s.lower() for s in json.loads(args.only_sources.read_text(encoding="utf-8"))
        }
        docx_paths = [
            p for p in docx_paths if staged_source_value(p, args.dataset_root).lower() in wanted
        ]
        found = {staged_source_value(p, args.dataset_root).lower() for p in docx_paths}
        for missing in sorted(wanted - found):
            print(f"  WARNING: no docx matches {missing}")
        if not docx_paths:
            raise SystemExit(f"None of the sources in {args.only_sources} matched a docx")

    records = [
        {
            "source": staged_source_value(p, args.dataset_root),
            "original": extract_docx_text(p).strip(),
        }
        for p in docx_paths
    ]

    args.output_dir.mkdir(parents=True, exist_ok=True)
    batches: list[list[dict]] = [[]]
    running = 0
    for record in records:
        size = len(record["original"])
        if batches[-1] and running + size > args.batch_chars:
            batches.append([])
            running = 0
        batches[-1].append(record)
        running += size

    for idx, batch in enumerate(batches):
        out_path = args.output_dir / f"layla_normalization_input_batch_{idx:03d}.json"
        out_path.write_text(
            json.dumps({"records": batch}, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        chars = sum(len(r["original"]) for r in batch)
        print(f"  {out_path.name}: {len(batch)} transcripts, {chars} chars")

    print(f"Wrote {len(batches)} batch files to {args.output_dir}")
    print(f"Total: {len(records)} transcripts, {sum(len(r['original']) for r in records)} chars")


if __name__ == "__main__":
    main()
