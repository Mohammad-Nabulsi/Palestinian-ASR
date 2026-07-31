"""Stage 3 — assemble: build one curated data root from many cleaned shard trees.

Replaces four scripts that were all variations on "copy some cleaned shards into a
data root, then swap one source's shards for a newer generation":

- ``scripts/merge_cleaned_outputs_and_report.py``      (merge two cleaned roots + English audit)
- ``scripts/replace_merged_omnilingual_with_recovered.py``
- ``scripts/create_data_with_final_omnilingual.py``    (hardlink copy + replace Omnilingual clean)
- the manual "flatten ``data/clean`` into ``data/``" step in DATA_CURATION.md

Contributions are applied in order. A contribution with ``replaces: <name>`` removes
the files an earlier contribution wrote before adding its own, which is exactly the
"swap Omnilingual v1 shards for v2+v3 recovered shards" operation.
"""
from __future__ import annotations

import os
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from ..config import StageSpec
from ..context import RunContext, StageResult
from ..shards import count_rows, iter_table_batches

ENGLISH_TOKEN_RE = re.compile(r"[A-Za-z]+(?:[-'][A-Za-z]+)*")


def place(src: Path, dest: Path, mode: str) -> None:
    dest.parent.mkdir(parents=True, exist_ok=True)
    if dest.exists() or dest.is_symlink():
        dest.unlink()
    if mode == "copy":
        shutil.copy2(src, dest)
    elif mode == "hardlink":
        try:
            os.link(src, dest)
        except OSError:
            shutil.copy2(src, dest)
    elif mode == "symlink":
        dest.symlink_to(os.path.relpath(src, start=dest.parent))
    elif mode == "move":
        shutil.move(str(src), str(dest))
    else:
        raise ValueError(f"Unknown assemble mode: {mode}")


def audit_english_tokens(
    paths: list[Path], fields: list[str], max_examples: int = 20
) -> dict[str, Any]:
    """Report which fields contribute English tokens, and sample rows.

    Carried over from ``merge_cleaned_outputs_and_report.py``: English tokens in
    dropped Omnilingual shards often come from metadata fields (``prompt``,
    ``source_file``), not the transcript, so counts are reported per field.
    """
    token_counts_by_field: dict[str, Counter] = {}
    examples: list[dict[str, Any]] = []
    rows_scanned = 0

    for path in paths:
        for table in iter_table_batches(path, batch_size=2_000):
            present = [f for f in fields if f in table.column_names]
            if not present:
                continue
            columns = {f: table[f].to_pylist() for f in present}
            for idx in range(table.num_rows):
                rows_scanned += 1
                row_hits: dict[str, list[str]] = {}
                for field_name in present:
                    value = columns[field_name][idx]
                    if value is None:
                        continue
                    tokens = ENGLISH_TOKEN_RE.findall(str(value))
                    if tokens:
                        row_hits[field_name] = tokens
                        token_counts_by_field.setdefault(field_name, Counter()).update(tokens)
                if row_hits and len(examples) < max_examples:
                    examples.append({"file": path.name, "row": idx, "tokens_by_field": row_hits})

    return {
        "rows_scanned": rows_scanned,
        "files": [str(p) for p in paths],
        "token_counts_by_field": {
            field_name: dict(counter.most_common(50))
            for field_name, counter in token_counts_by_field.items()
        },
        "examples": examples,
    }


def run(ctx: RunContext, spec: StageSpec) -> StageResult:
    result = StageResult(stage_id=spec.id, stage=spec.stage)
    output_root = spec.require_path("output_root")
    default_mode = spec.get("mode", "hardlink")
    layout = spec.get("layout", "flat")
    if layout not in {"flat", "nested"}:
        raise SystemExit(f"assemble: layout must be 'flat' or 'nested', got {layout!r}")

    if output_root.exists() and spec.get("overwrite", True):
        if output_root.is_symlink():
            raise SystemExit(f"Refusing to overwrite a symlinked output root: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    reports_dir = ctx.stage_reports_dir(spec.id)
    written_by_contribution: dict[str, list[Path]] = {}
    manifest: list[dict[str, Any]] = []

    for contribution in spec.require("contributions"):
        name = contribution["name"]
        source = Path(contribution["input"])
        includes = contribution.get("include", ["*.parquet"])
        mode = contribution.get("mode", default_mode)
        prefix = contribution.get("prefix", name)
        subdir = contribution.get("subdir", "" if layout == "flat" else "clean")
        rename = contribution.get("rename", {})

        replaces = contribution.get("replaces")
        if replaces:
            removed = written_by_contribution.pop(replaces, [])
            removed_rows = sum(count_rows(p) for p in removed if p.exists())
            for path in removed:
                if path.exists():
                    path.unlink()
            ctx.log(
                f"contribution {name}: replaced {replaces} "
                f"({len(removed)} file(s), {removed_rows:,} row(s) removed)"
            )
            manifest.append(
                {
                    "contribution": name,
                    "replaced": replaces,
                    "removed_files": [str(p) for p in removed],
                    "removed_rows": removed_rows,
                }
            )

        if not source.exists():
            if contribution.get("optional", False):
                ctx.log(f"contribution {name}: source missing, skipped (optional)")
                result.counts[name] = {"status": "skipped_missing"}
                continue
            raise FileNotFoundError(f"assemble contribution {name!r}: missing input {source}")

        matched: list[Path] = []
        for pattern in includes:
            matched.extend(sorted(source.glob(pattern)))
        matched = [p for p in dict.fromkeys(matched) if p.is_file()]
        if not matched:
            if contribution.get("optional", False):
                ctx.log(f"contribution {name}: no files matched, skipped (optional)")
                result.counts[name] = {"status": "skipped_empty"}
                continue
            raise FileNotFoundError(
                f"assemble contribution {name!r}: no files matched {includes} under {source}"
            )

        dest_dir = output_root / subdir if subdir else output_root
        written: list[Path] = []
        for path in matched:
            dest_name = path.name
            for needle, replacement in rename.items():
                dest_name = dest_name.replace(needle, replacement)
            if prefix and not dest_name.startswith(f"{prefix}__"):
                dest_name = f"{prefix}__{dest_name}"
            dest = dest_dir / dest_name
            if ctx.dry_run:
                written.append(dest)
                continue
            place(path, dest, mode)
            written.append(dest)

        rows = 0 if ctx.dry_run else sum(count_rows(p) for p in written)
        written_by_contribution[name] = written
        result.counts[name] = {"files": len(written), "rows": rows, "mode": mode}
        manifest.append(
            {
                "contribution": name,
                "input": str(source),
                "mode": mode,
                "files": [str(p) for p in written],
                "rows": rows,
            }
        )
        ctx.log(f"contribution {name}: {len(written)} file(s), {rows:,} row(s) via {mode}")

    audit = spec.get("audit_english")
    if audit and not ctx.dry_run:
        audit_root = Path(audit["input"])
        audit_paths: list[Path] = []
        for pattern in audit.get("include", ["*.parquet"]):
            audit_paths.extend(sorted(audit_root.glob(pattern)))
        if audit_paths:
            report = audit_english_tokens(
                audit_paths,
                audit.get("fields", ["raw_text", "manual_normalized_transcript", "prompt"]),
            )
            ctx.write_json(reports_dir / "english_token_audit.json", report)
            result.counts["__english_audit__"] = {
                "rows_scanned": report["rows_scanned"],
                "fields_with_tokens": sorted(report["token_counts_by_field"]),
            }
            ctx.log(f"english audit: scanned {report['rows_scanned']:,} dropped row(s)")
        else:
            ctx.log(f"english audit: no files matched under {audit_root}, skipped")

    if not ctx.dry_run:
        ctx.write_json(reports_dir / "assemble_manifest.json", manifest)
        total_files = sum(len(v) for v in written_by_contribution.values())
        total_rows = sum(
            c.get("rows", 0) for c in result.counts.values() if isinstance(c, dict)
        )
        result.counts["__totals__"] = {"files": total_files, "rows": total_rows}
        ctx.log(f"assembled {total_files} file(s), {total_rows:,} row(s) -> {output_root}")

    result.outputs["root"] = str(output_root)
    return result
