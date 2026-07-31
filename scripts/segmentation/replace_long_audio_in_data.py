#!/usr/bin/env python
"""Long-audio segmentation, Part B — splice the segmented rows into `data/`.

Part A (`segment_whisperx.py`) segmented every row flagged as > 30s
(`segmented/flagged_over30s.json`) into <=28s pieces under `segmented/v1/`, with full
provenance, and never touched `data/`. This script performs the in-place replacement:

For every source shard that had flagged rows:
  1. Back up the untouched original to `segmented/v1/backup_originals/`.
  2. Remove exactly the flagged (> 30s) rows.
  3. Insert the corresponding segmented sub-rows in their place, mapped into that
     shard's own column schema (same id/text/audio/duration columns every other row
     uses, so nothing downstream has to special-case segmented data). Non-text,
     non-flag columns (gender, speaker_id, language, ...) are carried over from the
     original row onto every sub-segment it produced.
  4. Recompute the `flag_contains_english` / `flag_contains_number` /
     `flag_contains_bracket_token` / `flag_audio_too_short` / `flag_missing_duration`
     family and `manual_normalized_transcript` with `pipeline.textnorm` — the same
     functions the `clean` stage uses — rather than carrying stale flags computed on
     the pre-split (and much longer) original text.
  5. Append 6 new provenance columns (`segment_source_file`, `segment_source_row_index`,
     `segment_idx`, `segment_start_offset`, `segment_end_offset`, `segment_align_score`)
     to every row in the shard (null on untouched rows) so segmented rows stay
     identifiable in `data/` itself.
  6. Write to a staging copy, verify row counts, then overwrite the original in place.

Resumable via `segmented/v1/replacement_checkpoint.txt` (one shard per line, same
pattern as Part A's `checkpoint.txt`).
"""
import argparse
import json
import os
import shutil
import sys
from collections import defaultdict

import pyarrow as pa
import pyarrow.parquet as pq

REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, REPO_ROOT)
from pipeline.shards import add_or_replace_column, align_table_to_schema  # noqa: E402
from pipeline.textnorm import (  # noqa: E402
    has_bracket_token,
    has_english,
    has_number,
    normalize_arabic_transcript,
)

DATA_DIR = os.path.join(REPO_ROOT, "data")
SEG_DIR = os.path.join(REPO_ROOT, "segmented", "v1")
BACKUP_DIR = os.path.join(SEG_DIR, "backup_originals")
STAGING_DIR = os.path.join(SEG_DIR, "staged_replacement")
CHECKPOINT = os.path.join(SEG_DIR, "replacement_checkpoint.txt")
MANIFEST = os.path.join(SEG_DIR, "replacement_manifest.json")

MIN_DURATION_SEC = 0.5

ID_COLUMN = {"layla": "seg_id", "masc_c_only": "video_id", "omnilingual_apc": "segment_id"}
TEXT_COLUMNS = {
    "layla": ["transcription"],
    "masc_c_only": ["text"],
    "omnilingual_apc": ["raw_text", "precheck_text_v2"],
}
NORMALIZED_COLUMN = "manual_normalized_transcript"
AUDIO_COLUMN = "audio"
DURATION_COLUMN = "duration"
PROVENANCE_SCHEMA = [
    ("segment_source_file", pa.string()),
    ("segment_source_row_index", pa.int64()),
    ("segment_idx", pa.int64()),
    ("segment_start_offset", pa.float64()),
    ("segment_end_offset", pa.float64()),
    ("segment_align_score", pa.float64()),
]


def log(msg):
    print(msg, flush=True)


def resolve_original_path(group, orig_file):
    if group == "layla":
        return os.path.join(DATA_DIR, orig_file)
    return os.path.join(DATA_DIR, "clean", orig_file)


def compute_flag(flag_col, text, duration):
    name = flag_col.lower()
    if "english" in name:
        return has_english(text)
    if "number" in name:
        return has_number(text)
    if "bracket" in name:
        return has_bracket_token(text)
    if "audio_too_short" in name:
        return duration < MIN_DURATION_SEC
    if "missing_duration" in name:
        return False
    if "placeholder" in name:
        return False
    return False


def load_segments_by_orig_file():
    by_file = defaultdict(list)
    for fname in sorted(os.listdir(SEG_DIR)):
        if not fname.endswith("__segmented.parquet"):
            continue
        t = pq.read_table(os.path.join(SEG_DIR, fname))
        for row in t.to_pylist():
            by_file[row["orig_file"]].append(row)
    return by_file


def process_shard(orig_file, group, segments, flagged_indices):
    orig_path = resolve_original_path(group, orig_file)
    rel = os.path.relpath(orig_path, REPO_ROOT)
    staged_path = os.path.join(STAGING_DIR, rel)
    backup_path = os.path.join(BACKUP_DIR, rel)

    table = pq.read_table(orig_path)
    names = table.column_names
    id_col = ID_COLUMN[group]
    text_cols = TEXT_COLUMNS[group]
    flag_cols = [c for c in names if c.startswith("flag_")]
    special = {id_col, AUDIO_COLUMN, DURATION_COLUMN, NORMALIZED_COLUMN, *text_cols, *flag_cols}
    carry_cols = [c for c in names if c not in special]

    n = table.num_rows
    keep_mask = [i not in flagged_indices for i in range(n)]
    n_removed = n - sum(keep_mask)
    if n_removed != len(flagged_indices):
        raise SystemExit(
            f"{orig_file}: expected to remove {len(flagged_indices)} flagged rows, "
            f"but only {n_removed}/{n} rows matched (index out of range?)"
        )

    # snapshot carry-over values + orig id for every flagged row before filtering
    carry_by_idx = {}
    id_by_idx = {}
    idx_list = sorted(flagged_indices)
    if idx_list:
        sub = table.take(pa.array(idx_list))
        sub_rows = sub.to_pylist()
        for idx, row in zip(idx_list, sub_rows):
            carry_by_idx[idx] = {c: row[c] for c in carry_cols}
            id_by_idx[idx] = row[id_col]

    retained = table.filter(pa.array(keep_mask, type=pa.bool_()))
    for name_, type_ in PROVENANCE_SCHEMA:
        retained = add_or_replace_column(retained, name_, [None] * retained.num_rows, type_)
    target_schema = retained.schema

    new_rows = []
    for seg in segments:
        idx = seg["orig_row_index"]
        text = seg["text"]
        dur = seg["duration"]
        row = dict(carry_by_idx[idx])
        row[id_col] = f"{id_by_idx[idx]}__seg{seg['segment_idx']:02d}"
        for tc in text_cols:
            row[tc] = text
        row[NORMALIZED_COLUMN] = normalize_arabic_transcript(text)
        row[AUDIO_COLUMN] = {"bytes": seg["audio"]["bytes"], "path": None}
        row[DURATION_COLUMN] = dur
        for fc in flag_cols:
            row[fc] = compute_flag(fc, text, dur)
        row["segment_source_file"] = orig_file
        row["segment_source_row_index"] = idx
        row["segment_idx"] = seg["segment_idx"]
        row["segment_start_offset"] = seg["start_offset"]
        row["segment_end_offset"] = seg["end_offset"]
        row["segment_align_score"] = seg["align_score"]
        new_rows.append(row)

    new_table = align_table_to_schema(pa.Table.from_pylist(new_rows), target_schema)
    final_table = pa.concat_tables([retained, new_table])

    expected_rows = (n - n_removed) + len(new_rows)
    if final_table.num_rows != expected_rows:
        raise SystemExit(
            f"{orig_file}: row-count mismatch after splice: got {final_table.num_rows}, "
            f"expected {expected_rows}"
        )

    os.makedirs(os.path.dirname(staged_path), exist_ok=True)
    pq.write_table(final_table, staged_path, compression="zstd")
    # verify the staged file is readable and has the right row count before touching data/
    check = pq.ParquetFile(staged_path)
    if check.metadata.num_rows != expected_rows:
        raise SystemExit(f"{orig_file}: staged file footer row count mismatch")

    os.makedirs(os.path.dirname(backup_path), exist_ok=True)
    shutil.copy2(orig_path, backup_path)
    shutil.copy2(staged_path, orig_path)

    return {
        "orig_file": orig_file,
        "group": group,
        "rows_before": n,
        "rows_removed": n_removed,
        "rows_added": len(new_rows),
        "rows_after": final_table.num_rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-files", type=int, default=None, help="smoke test: stop after N shards")
    args = ap.parse_args()

    os.makedirs(BACKUP_DIR, exist_ok=True)
    os.makedirs(STAGING_DIR, exist_ok=True)

    flagged = json.load(open(os.path.join(REPO_ROOT, "segmented", "flagged_over30s.json")))
    flagged_by_file = defaultdict(set)
    group_by_file = {}
    for r in flagged:
        flagged_by_file[r["file"]].add(r["row_index"])
        group_by_file[r["file"]] = r["group"]

    segments_by_file = load_segments_by_orig_file()

    done = set()
    if os.path.exists(CHECKPOINT):
        done = set(open(CHECKPOINT).read().split())

    manifest = json.load(open(MANIFEST)) if os.path.exists(MANIFEST) else []
    n_done_this_run = 0
    for orig_file in sorted(flagged_by_file):
        if orig_file in done:
            log(f"skip (done): {orig_file}")
            continue
        if args.limit_files and n_done_this_run >= args.limit_files:
            break
        group = group_by_file[orig_file]
        segs = segments_by_file.get(orig_file, [])
        if not segs:
            raise SystemExit(f"{orig_file}: no segmented rows found in {SEG_DIR} — run Part A first")
        entry = process_shard(orig_file, group, segs, flagged_by_file[orig_file])
        manifest.append(entry)
        log(
            f"{orig_file}: {entry['rows_before']} -> {entry['rows_after']} rows "
            f"(-{entry['rows_removed']} long, +{entry['rows_added']} segments)"
        )
        with open(CHECKPOINT, "a") as f:
            f.write(orig_file + "\n")
        with open(MANIFEST, "w") as f:
            json.dump(manifest, f, ensure_ascii=False, indent=1)
        n_done_this_run += 1

    log(f"DONE — {n_done_this_run} shard(s) processed this run, {len(manifest)} total in manifest")


if __name__ == "__main__":
    main()
