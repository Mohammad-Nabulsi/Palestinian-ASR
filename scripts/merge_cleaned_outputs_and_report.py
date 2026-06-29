#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow.parquet as pq


DEFAULT_SOURCES = [
    Path("/home/MohammadNabulsi/whisper/data_cleaned_text_qasr_casablanca_omni_v1"),
    Path("/home/MohammadNabulsi/whisper/data_cleaned_text_v1"),
]
DEFAULT_DEST = Path("/home/MohammadNabulsi/whisper/data_cleaned_text_merged_v1")
DEFAULT_INTERMEDIATE_ROOT = Path("/home/MohammadNabulsi/whisper/intermediate/merged_cleaned_sources")
ENGLISH_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")
OMNILINGUAL_TEXT_FIELDS = ("raw_text", "manual_normalized_transcript", "prompt")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Merge multiple cleaned-data roots into one destination, archive the emptied "
            "source roots, and generate a detailed omnilingual dropped-English report."
        )
    )
    parser.add_argument(
        "--source",
        action="append",
        dest="sources",
        type=Path,
        help="Source cleaned-data root. Can be passed more than once.",
    )
    parser.add_argument(
        "--dest",
        type=Path,
        default=DEFAULT_DEST,
        help=f"Merged destination root. Default: {DEFAULT_DEST}",
    )
    parser.add_argument(
        "--intermediate-root",
        type=Path,
        default=DEFAULT_INTERMEDIATE_ROOT,
        help=f"Where emptied source roots should be archived. Default: {DEFAULT_INTERMEDIATE_ROOT}",
    )
    parser.add_argument(
        "--delete-empty-sources",
        action="store_true",
        help="Delete emptied source roots instead of archiving them under --intermediate-root.",
    )
    parser.add_argument(
        "--overwrite-reports",
        action="store_true",
        help="Allow overwriting files in reports/source_reports/<source>/ if they already exist.",
    )
    return parser.parse_args()


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def move_file(src: Path, dst: Path) -> None:
    ensure_dir(dst.parent)
    if dst.exists():
        raise FileExistsError(f"Destination already exists: {dst}")
    shutil.move(str(src), str(dst))


def remove_empty_dirs(root: Path) -> None:
    for path in sorted((p for p in root.rglob("*") if p.is_dir()), reverse=True):
        try:
            path.rmdir()
        except OSError:
            pass


def archive_or_delete_source(
    source_root: Path,
    intermediate_root: Path,
    delete_empty_sources: bool,
) -> dict[str, Any]:
    remove_empty_dirs(source_root)
    remaining = sorted(source_root.rglob("*"))
    if remaining:
        raise RuntimeError(f"Source root is not empty after merge: {source_root}")

    if delete_empty_sources:
        source_root.rmdir()
        return {
            "source_root": str(source_root),
            "action": "deleted_empty_source",
        }

    ensure_dir(intermediate_root)
    archived_root = intermediate_root / source_root.name
    if archived_root.exists():
        raise FileExistsError(f"Archive destination already exists: {archived_root}")
    shutil.move(str(source_root), str(archived_root))
    return {
        "source_root": str(source_root),
        "action": "archived_empty_source",
        "archived_root": str(archived_root),
    }


def move_clean_and_dropped(source_root: Path, dest_root: Path) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    for section in ("clean", "dropped"):
        src_section = source_root / section
        if not src_section.exists():
            continue
        for src_file in sorted(p for p in src_section.rglob("*") if p.is_file()):
            rel = src_file.relative_to(src_section)
            dst_file = dest_root / section / rel
            move_file(src_file, dst_file)
            actions.append(
                {
                    "kind": "move",
                    "section": section,
                    "source": str(src_file),
                    "dest": str(dst_file),
                }
            )
    return actions


def move_reports(
    source_root: Path,
    dest_root: Path,
    overwrite_reports: bool,
) -> list[dict[str, Any]]:
    actions: list[dict[str, Any]] = []
    src_reports = source_root / "reports"
    if not src_reports.exists():
        return actions

    report_dest_root = dest_root / "reports" / "source_reports" / source_root.name
    ensure_dir(report_dest_root)

    for src_file in sorted(p for p in src_reports.rglob("*") if p.is_file()):
        rel = src_file.relative_to(src_reports)
        dst_file = report_dest_root / rel
        ensure_dir(dst_file.parent)
        if dst_file.exists():
            if not overwrite_reports:
                raise FileExistsError(f"Destination report already exists: {dst_file}")
            dst_file.unlink()
        shutil.move(str(src_file), str(dst_file))
        actions.append(
            {
                "kind": "move",
                "section": "reports",
                "source": str(src_file),
                "dest": str(dst_file),
            }
        )
    return actions


def english_tokens(value: Any) -> list[str]:
    if not isinstance(value, str) or not value.strip():
        return []
    return ENGLISH_TOKEN_RE.findall(value)


def build_omnilingual_english_report(dest_root: Path) -> dict[str, Any]:
    contains_english_dir = dest_root / "dropped" / "contains_english"
    shard_paths = sorted(contains_english_dir.glob("omnilingual_apc*.parquet"))

    overall_counter: Counter[str] = Counter()
    field_counters: dict[str, Counter[str]] = {}
    per_shard: list[dict[str, Any]] = []
    total_rows_with_english = 0

    for shard_path in shard_paths:
        table = pq.read_table(shard_path)
        rows = table.to_pylist()
        shard_counter: Counter[str] = Counter()
        shard_field_counters: dict[str, Counter[str]] = {}
        shard_rows: list[dict[str, Any]] = []

        for row_index, row in enumerate(rows):
            fields_with_english: dict[str, list[str]] = {}
            for key in OMNILINGUAL_TEXT_FIELDS:
                value = row.get(key)
                tokens = english_tokens(value)
                if not tokens:
                    continue
                fields_with_english[key] = tokens
                shard_counter.update(tokens)
                overall_counter.update(tokens)
                shard_field_counters.setdefault(key, Counter()).update(tokens)
                field_counters.setdefault(key, Counter()).update(tokens)

            if fields_with_english:
                total_rows_with_english += 1
                shard_rows.append(
                    {
                        "row_index": row_index,
                        "segment_id": row.get("segment_id"),
                        "source_file": row.get("source_file"),
                        "fields_with_english": fields_with_english,
                    }
                )

        per_shard.append(
            {
                "shard_path": str(shard_path),
                "num_rows": table.num_rows,
                "rows_with_english": len(shard_rows),
                "english_token_counts": dict(sorted(shard_counter.items())),
                "english_token_counts_by_field": {
                    field: dict(sorted(counter.items()))
                    for field, counter in sorted(shard_field_counters.items())
                },
                "rows": shard_rows,
            }
        )

    return {
        "report_type": "omnilingual_dropped_contains_english",
        "contains_english_dir": str(contains_english_dir),
        "num_shards": len(shard_paths),
        "total_rows_with_english": total_rows_with_english,
        "overall_english_token_counts": dict(sorted(overall_counter.items())),
        "overall_english_token_counts_by_field": {
            field: dict(sorted(counter.items()))
            for field, counter in sorted(field_counters.items())
        },
        "shards": per_shard,
    }


def write_json(path: Path, payload: dict[str, Any]) -> None:
    ensure_dir(path.parent)
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")


def write_markdown_report(path: Path, payload: dict[str, Any]) -> None:
    lines: list[str] = []
    lines.append("# Omnilingual Dropped `contains_english` Report")
    lines.append("")
    lines.append(f"- Shards scanned: {payload['num_shards']}")
    lines.append(f"- Rows with English tokens: {payload['total_rows_with_english']}")
    lines.append("")
    lines.append("## Overall Tokens")
    lines.append("")
    overall = payload["overall_english_token_counts"]
    if overall:
        for token, count in sorted(overall.items()):
            lines.append(f"- `{token}`: {count}")
    else:
        lines.append("- No English tokens found.")
    lines.append("")
    lines.append("## By Field")
    lines.append("")
    by_field = payload["overall_english_token_counts_by_field"]
    if by_field:
        for field, token_counts in sorted(by_field.items()):
            lines.append(f"### `{field}`")
            lines.append("")
            for token, count in sorted(token_counts.items()):
                lines.append(f"- `{token}`: {count}")
            lines.append("")
    else:
        lines.append("- No field-level English tokens found.")
        lines.append("")
    lines.append("## Per Shard")
    lines.append("")
    for shard in payload["shards"]:
        lines.append(f"### `{Path(shard['shard_path']).name}`")
        lines.append("")
        lines.append(f"- Rows with English tokens: {shard['rows_with_english']} / {shard['num_rows']}")
        if shard["english_token_counts"]:
            lines.append("- Token counts:")
            for token, count in sorted(shard["english_token_counts"].items()):
                lines.append(f"  - `{token}`: {count}")
        else:
            lines.append("- Token counts: none")
        lines.append("- Example rows:")
        example_rows = shard["rows"][:10]
        if example_rows:
            for row in example_rows:
                lines.append(
                    f"  - row={row['row_index']} segment_id={row.get('segment_id')} "
                    f"source_file={row.get('source_file')}"
                )
                for field, tokens in sorted(row["fields_with_english"].items()):
                    lines.append(f"    - {field}: {tokens}")
        else:
            lines.append("  - none")
        lines.append("")

    ensure_dir(path.parent)
    path.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    sources = args.sources or DEFAULT_SOURCES
    dest_root = args.dest.resolve()
    intermediate_root = args.intermediate_root.resolve()

    ensure_dir(dest_root / "clean")
    ensure_dir(dest_root / "dropped")
    ensure_dir(dest_root / "reports" / "source_reports")
    ensure_dir(dest_root / "reports" / "generated")

    merge_actions: list[dict[str, Any]] = []
    source_cleanup: list[dict[str, Any]] = []

    for source_root in sources:
        source_root = source_root.resolve()
        if not source_root.exists():
            raise FileNotFoundError(f"Missing source root: {source_root}")
        if source_root == dest_root:
            raise ValueError(f"Source root cannot equal destination root: {source_root}")

        merge_actions.extend(move_clean_and_dropped(source_root, dest_root))
        merge_actions.extend(move_reports(source_root, dest_root, args.overwrite_reports))
        source_cleanup.append(
            archive_or_delete_source(
                source_root=source_root,
                intermediate_root=intermediate_root,
                delete_empty_sources=args.delete_empty_sources,
            )
        )

    english_report = build_omnilingual_english_report(dest_root)
    report_json_path = dest_root / "reports" / "generated" / "omnilingual_contains_english_report.json"
    report_md_path = dest_root / "reports" / "generated" / "omnilingual_contains_english_report.md"
    write_json(report_json_path, english_report)
    write_markdown_report(report_md_path, english_report)

    manifest = {
        "sources": [str(Path(s).resolve()) for s in sources],
        "dest_root": str(dest_root),
        "intermediate_root": None if args.delete_empty_sources else str(intermediate_root),
        "merge_actions_count": len(merge_actions),
        "merge_actions": merge_actions,
        "source_cleanup": source_cleanup,
        "generated_reports": {
            "omnilingual_contains_english_report_json": str(report_json_path),
            "omnilingual_contains_english_report_md": str(report_md_path),
        },
    }
    write_json(dest_root / "reports" / "generated" / "merge_manifest.json", manifest)

    print(f"Merged {len(sources)} source roots into: {dest_root}")
    print(f"Generated JSON report: {report_json_path}")
    print(f"Generated Markdown report: {report_md_path}")


if __name__ == "__main__":
    main()
