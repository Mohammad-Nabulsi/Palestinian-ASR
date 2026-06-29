#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from collections import Counter
from pathlib import Path

import pyarrow.parquet as pq

MERGED_ROOT = Path("/home/MohammadNabulsi/whisper/data_cleaned_text_merged_v1")
OMNI_V2_ROOT = Path("/home/MohammadNabulsi/whisper/data_cleaned_text_omnilingual_v2")
OMNI_RECOVERY_ROOT = Path("/home/MohammadNabulsi/whisper/data_cleaned_text_omnilingual_v3_recovered_from_v2")
BACKUP_ROOT = Path("/home/MohammadNabulsi/whisper/intermediate/omnilingual_replacement_backup")

MERGED_OMNI_CLEAN_PREFIX = "omnilingual_apc__"
V2_CLEAN_SUFFIX = "__clean.parquet"
RECOVERED_SUFFIX = "__dropped.parquet"
SUMMARY_PATH = OMNI_RECOVERY_ROOT / "reports" / "summary.txt"


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def count_rows(paths: list[Path]) -> int:
    total = 0
    for path in paths:
        total += pq.ParquetFile(path).metadata.num_rows
    return total


def merged_name_from_v2(path: Path, kind: str) -> str:
    base = path.name
    if kind == "clean":
        return f"{MERGED_OMNI_CLEAN_PREFIX}{base.replace(V2_CLEAN_SUFFIX, '__clean.parquet')}"
    return f"{MERGED_OMNI_CLEAN_PREFIX}{base.replace(RECOVERED_SUFFIX, '__dropped.parquet')}"


def backup_existing(paths: list[Path], backup_dir: Path) -> list[str]:
    ensure_dir(backup_dir)
    moved = []
    for path in paths:
        dest = backup_dir / path.name
        if dest.exists():
            dest.unlink()
        shutil.move(str(path), str(dest))
        moved.append(str(dest))
    return moved


def copy_with_renames(source_paths: list[Path], dest_dir: Path, kind: str) -> list[str]:
    ensure_dir(dest_dir)
    written = []
    for path in source_paths:
        dest = dest_dir / merged_name_from_v2(path, kind)
        shutil.copy2(path, dest)
        written.append(str(dest))
    return written


def parse_expected_remaining(summary_path: Path) -> int | None:
    if not summary_path.exists():
        return None
    for line in summary_path.read_text(encoding="utf-8").splitlines():
        if line.startswith("Still containing English after span removal:"):
            try:
                return int(line.rsplit(":", 1)[1].strip())
            except ValueError:
                return None
    return None


def collect_inputs() -> dict[str, list[Path]]:
    return {
        "merged_clean": sorted((MERGED_ROOT / "clean").glob(f"{MERGED_OMNI_CLEAN_PREFIX}*.parquet")),
        "merged_dropped_english": sorted((MERGED_ROOT / "dropped" / "contains_english").glob(f"{MERGED_OMNI_CLEAN_PREFIX}*.parquet")),
        "v2_clean": sorted((OMNI_V2_ROOT / "clean").glob("*.parquet")),
        "recovered_clean": sorted((OMNI_RECOVERY_ROOT / "recovered_clean").glob("*.parquet")),
        "still_contains_english": sorted((OMNI_RECOVERY_ROOT / "still_contains_english").glob("*.parquet")),
    }


def main() -> None:
    inputs = collect_inputs()
    expected_remaining = parse_expected_remaining(SUMMARY_PATH)

    merged_clean_dir = MERGED_ROOT / "clean"
    merged_drop_eng_dir = MERGED_ROOT / "dropped" / "contains_english"
    backup_clean_dir = BACKUP_ROOT / "clean"
    backup_drop_eng_dir = BACKUP_ROOT / "dropped_contains_english"
    reports_dir = MERGED_ROOT / "reports" / "generated"
    ensure_dir(reports_dir)

    replaced_counts = Counter()
    replaced_counts["old_clean_rows"] = count_rows(inputs["merged_clean"])
    replaced_counts["old_dropped_contains_english_rows"] = count_rows(inputs["merged_dropped_english"])
    replaced_counts["v2_clean_rows"] = count_rows(inputs["v2_clean"])
    replaced_counts["recovered_rows"] = count_rows(inputs["recovered_clean"])
    replaced_counts["new_clean_rows"] = replaced_counts["v2_clean_rows"] + replaced_counts["recovered_rows"]
    replaced_counts["new_dropped_contains_english_rows"] = count_rows(inputs["still_contains_english"])

    if expected_remaining is not None and replaced_counts["new_dropped_contains_english_rows"] != expected_remaining:
        raise SystemExit(
            "Refusing to replace merged Omnilingual shards because the saved recovery summary "
            f"expects {expected_remaining} still-English rows but "
            f"{replaced_counts['new_dropped_contains_english_rows']} rows were materialized in "
            f"{OMNI_RECOVERY_ROOT / 'still_contains_english'}."
        )

    backed_up_clean = backup_existing(inputs["merged_clean"], backup_clean_dir)
    backed_up_drop_eng = backup_existing(inputs["merged_dropped_english"], backup_drop_eng_dir)
    written_clean = copy_with_renames(inputs["v2_clean"], merged_clean_dir, kind="clean")
    written_recovered = copy_with_renames(inputs["recovered_clean"], merged_clean_dir, kind="recovered")
    written_drop_eng = copy_with_renames(inputs["still_contains_english"], merged_drop_eng_dir, kind="dropped")

    report = {
        "script": str(Path(__file__).resolve()),
        "merged_root": str(MERGED_ROOT),
        "omnilingual_v2_root": str(OMNI_V2_ROOT),
        "omnilingual_recovery_root": str(OMNI_RECOVERY_ROOT),
        "backup_root": str(BACKUP_ROOT),
        "expected_remaining_contains_english_rows": expected_remaining,
        "counts": dict(replaced_counts),
        "backed_up_clean_files": backed_up_clean,
        "backed_up_dropped_contains_english_files": backed_up_drop_eng,
        "written_clean_files_from_v2": written_clean,
        "written_clean_files_from_recovery": written_recovered,
        "written_dropped_contains_english_files": written_drop_eng,
    }
    report_path = reports_dir / "omnilingual_replacement_manifest.json"
    report_path.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = [
        f"Merged root: {MERGED_ROOT}",
        f"Old merged Omnilingual clean rows: {replaced_counts['old_clean_rows']}",
        f"Old merged Omnilingual dropped contains_english rows: {replaced_counts['old_dropped_contains_english_rows']}",
        f"New merged Omnilingual clean rows: {replaced_counts['new_clean_rows']}",
        f"New merged Omnilingual dropped contains_english rows: {replaced_counts['new_dropped_contains_english_rows']}",
        f"Expected remaining dropped rows from recovery summary: {expected_remaining}",
        f"Manifest: {report_path}",
    ]
    print("\n".join(lines))


if __name__ == "__main__":
    main()
