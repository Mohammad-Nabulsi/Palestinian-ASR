#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import pyarrow.parquet as pq

SOURCE_ROOT = Path("/home/MohammadNabulsi/whisper/data_cleaned_text_merged_v1")
DEST_ROOT = Path("/home/MohammadNabulsi/whisper/data")
OMNI_V2_CLEAN = Path("/home/MohammadNabulsi/whisper/data_cleaned_text_omnilingual_v2/clean")
OMNI_RECOVERED_CLEAN = Path("/home/MohammadNabulsi/whisper/data_cleaned_text_omnilingual_v3_recovered_from_v2/recovered_clean")
OMNI_STILL_ENGLISH = Path("/home/MohammadNabulsi/whisper/data_cleaned_text_omnilingual_v3_recovered_from_v2/still_contains_english")
INTERMEDIATE_ROOT = Path("/home/MohammadNabulsi/whisper/intermediate")


def count_rows(paths: list[Path]) -> int:
    return sum(pq.ParquetFile(path).metadata.num_rows for path in paths)


def hardlink_copytree(src: Path, dst: Path) -> None:
    shutil.copytree(src, dst, copy_function=os.link)


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def main() -> None:
    if DEST_ROOT.exists():
        raise SystemExit(f"Destination already exists: {DEST_ROOT}")

    v2_clean_paths = sorted(OMNI_V2_CLEAN.glob("*.parquet"))
    recovered_paths = sorted(OMNI_RECOVERED_CLEAN.glob("*.parquet"))
    still_english_paths = sorted(OMNI_STILL_ENGLISH.glob("*.parquet"))
    if not v2_clean_paths:
        raise SystemExit(f"No Omnilingual v2 clean shards found under {OMNI_V2_CLEAN}")
    if not recovered_paths:
        raise SystemExit(f"No Omnilingual recovered clean shards found under {OMNI_RECOVERED_CLEAN}")

    hardlink_copytree(SOURCE_ROOT, DEST_ROOT)

    clean_dir = DEST_ROOT / "clean"
    reports_dir = DEST_ROOT / "reports" / "generated"
    ensure_dir(reports_dir)

    old_omni_clean = sorted(clean_dir.glob("omnilingual_apc__*.parquet"))
    old_omni_clean_rows = count_rows(old_omni_clean)
    for path in old_omni_clean:
        path.unlink()

    written_v2 = []
    for path in v2_clean_paths:
        dest = clean_dir / f"omnilingual_apc__{path.name}"
        shutil.copy2(path, dest)
        written_v2.append(dest)

    written_recovered = []
    for path in recovered_paths:
        renamed = path.name.replace("__dropped.parquet", "__recovered_clean.parquet")
        dest = clean_dir / f"omnilingual_apc__{renamed}"
        shutil.copy2(path, dest)
        written_recovered.append(dest)

    manifest = {
        "script": str(Path(__file__).resolve()),
        "source_root": str(SOURCE_ROOT),
        "dest_root": str(DEST_ROOT),
        "mode": "hardlink_copy_then_replace_omnilingual_clean",
        "old_omnilingual_clean_files_removed": [str(p) for p in old_omni_clean],
        "old_omnilingual_clean_rows_removed": old_omni_clean_rows,
        "new_omnilingual_clean_files_from_v2": [str(p) for p in written_v2],
        "new_omnilingual_clean_files_from_recovered": [str(p) for p in written_recovered],
        "new_omnilingual_clean_rows": count_rows(written_v2) + count_rows(written_recovered),
        "saved_still_contains_english_rows_present": count_rows(still_english_paths) if still_english_paths else 0,
        "note": (
            "This step replaces only the Omnilingual clean shards in data/. "
            "Dropped Omnilingual shards were not replaced here because the saved still_contains_english "
            "materialization is incomplete relative to the earlier 101-row recovery summary."
        ),
    }
    manifest_path = reports_dir / "omnilingual_final_kept_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    if INTERMEDIATE_ROOT.exists():
        shutil.rmtree(INTERMEDIATE_ROOT)

    print(f"Created {DEST_ROOT}")
    print(f"Removed old Omnilingual clean rows: {old_omni_clean_rows}")
    print(f"New Omnilingual clean rows in data/: {manifest['new_omnilingual_clean_rows']}")
    print(f"Deleted intermediate directory: {INTERMEDIATE_ROOT.exists() is False}")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
