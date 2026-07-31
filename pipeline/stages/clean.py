"""Stage 2 — clean: the unified fast text-cleaning pass.

This single stage replaces six near-identical implementations:

- ``.logs/clean_broad_v1.py``                              (MASC-C, recursive discovery)
- ``.logs/clean_targeted_qasr_casablanca_omni_v1.py``      (QASR + Casablanca + Omni, glob discovery)
- ``.logs/clean_qasr_part2.py``                            (QASR part 2, glob discovery)
- ``preprocess/fast_asr_data_cleaning_text_only_arrow_parquet*.ipynb``  (x3, same code)
- ``scripts/reclean_omnilingual_v2.py``                    (= this stage + ``precheck: [placeholders]``)
- ``scripts/reclean_omnilingual_v3.py`` /
  ``scripts/recover_omnilingual_token_span_rows_v3.py``    (= ``mode: recover`` + ``precheck: [spans, placeholders]``)

A diff of the first three showed they differed *only* in ``INPUT_ROOT``,
``OUTPUT_ROOT`` and the discovery globs; the Omnilingual variants additionally ran a
text pre-check before the drop rules. Both differences are now per-dataset config.

Drop rules (unchanged, applied in this order):
  1. ``missing_transcript`` — no recognizable text column
  2. ``contains_english``   — any ``[A-Za-z]``
  3. ``contains_number``    — any ASCII/Arabic/Persian digit
  4. ``audio_too_short``    — duration < ``min_duration_sec``
  5. ``missing_duration``   — only when ``drop_missing_duration: true``

Audio is never decoded; this is a text-first pass.
"""
from __future__ import annotations

import shutil
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa

from ..config import StageSpec
from ..context import RunContext, StageResult
from ..shards import (
    TableShardWriter,
    add_or_replace_column,
    dataset_from_file,
    find_first_column,
    iter_table_batches,
    safe_float,
    stable_file_id,
)
from ..textnorm import (
    apply_precheck,
    has_bracket_token,
    has_english,
    has_number,
    normalize_arabic_transcript,
)

TEXT_COLUMN_CANDIDATES = [
    "transcript",
    "transcription",
    "text",
    "raw_text",
    "sentence",
    "normalized_text",
]

DURATION_COLUMN_CANDIDATES = [
    "duration",
    "duration_sec",
    "duration_seconds",
    "audio_duration",
    "length_sec",
]

DATASET_COLUMN_CANDIDATES = ["dataset", "source", "corpus", "config"]

DROP_REASONS = (
    "missing_transcript",
    "contains_english",
    "contains_number",
    "audio_too_short",
    "missing_duration",
)


class DatasetPolicy:
    """Per-dataset knobs, resolved against the stage-level defaults."""

    def __init__(self, entry: dict[str, Any], defaults: dict[str, Any]) -> None:
        merged = {**defaults, **entry}
        self.name: str = merged["name"]
        self.globs: list[str] = merged.get("globs") or ["**/*"]
        self.recursive: bool = bool(merged.get("recursive", False))
        self.min_duration_sec: float = float(merged.get("min_duration_sec", 0.5))
        self.drop_missing_duration: bool = bool(merged.get("drop_missing_duration", False))
        self.precheck: list[str] = list(merged.get("precheck") or [])
        self.column_suffix: str = str(merged.get("column_suffix", ""))
        self.text_columns: list[str] = list(merged.get("text_columns") or TEXT_COLUMN_CANDIDATES)
        self.duration_columns: list[str] = list(
            merged.get("duration_columns") or DURATION_COLUMN_CANDIDATES
        )
        self.dataset_columns: list[str] = list(
            merged.get("dataset_columns") or DATASET_COLUMN_CANDIDATES
        )
        self.batch_size: int = int(merged.get("batch_size", 20_000))
        self.compression: str = str(merged.get("compression", "zstd"))
        self.max_files: int | None = merged.get("max_files")

    def col(self, base: str) -> str:
        return f"{base}{self.column_suffix}"


def discover(input_root: Path, policy: DatasetPolicy) -> list[Path]:
    matched: list[Path] = []
    for pattern in policy.globs:
        found = input_root.rglob(pattern) if policy.recursive else input_root.glob(pattern)
        matched.extend(sorted(p for p in found if p.is_file()))
    matched = [p for p in dict.fromkeys(matched) if p.suffix in {".parquet", ".arrow", ".jsonl"}]
    if policy.max_files is not None:
        matched = matched[: policy.max_files]
    return matched


def clean_table(
    table: pa.Table,
    source_file: Path,
    input_root: Path,
    policy: DatasetPolicy,
    writer: TableShardWriter,
    clean_path: Path,
    dropped_base: Path,
    totals: Counter,
    per_dataset: dict[str, Counter],
) -> None:
    n = table.num_rows
    if n == 0:
        return

    names = table.column_names
    text_col = find_first_column(names, policy.text_columns)
    duration_col = find_first_column(names, policy.duration_columns)
    dataset_col = find_first_column(names, policy.dataset_columns)
    file_id = stable_file_id(source_file, input_root)

    if text_col is None:
        table = add_or_replace_column(table, "source_file", [str(source_file)] * n)
        writer.write(dropped_base / "missing_transcript" / f"{file_id}__dropped.parquet", table)
        ds = dataset_from_file(source_file, input_root)
        totals["total"] += n
        totals["dropped_missing_transcript"] += n
        per_dataset[ds]["total"] += n
        per_dataset[ds]["dropped_missing_transcript"] += n
        return

    text_values = table[text_col].to_pylist()
    duration_values = (
        table[duration_col].to_pylist() if duration_col is not None else [None] * n
    )
    if dataset_col is not None:
        fallback = dataset_from_file(source_file, input_root)
        dataset_values = [
            str(x) if x is not None and str(x).strip() else fallback
            for x in table[dataset_col].to_pylist()
        ]
    else:
        dataset_values = [dataset_from_file(source_file, input_root)] * n

    precheck_texts: list[str] = []
    precheck_changed: list[bool] = []
    normalized: list[str] = []
    english_flags: list[bool] = []
    number_flags: list[bool] = []
    bracket_flags: list[bool] = []
    bracket_after_flags: list[bool] = []
    too_short_flags: list[bool] = []
    missing_duration_flags: list[bool] = []
    drop_reasons: list[str | None] = []
    keep_mask: list[bool] = []

    for raw_text, dur_raw in zip(text_values, duration_values):
        # The pre-check runs before the drop rules; with an empty strategy list it is
        # a no-op and the behavior matches the original fast pass exactly.
        checked = apply_precheck(raw_text, policy.precheck) if policy.precheck else raw_text
        changed = policy.precheck and str(checked) != str(raw_text or "")

        eng = has_english(checked)
        num = has_number(checked)
        bracket = has_bracket_token(raw_text)
        bracket_after = has_bracket_token(checked)

        dur = safe_float(dur_raw)
        missing_dur = dur is None
        too_short = False if missing_dur else dur < policy.min_duration_sec

        reason: str | None = None
        if eng:
            reason = "contains_english"
        elif num:
            reason = "contains_number"
        elif too_short:
            reason = "audio_too_short"
        elif missing_dur and policy.drop_missing_duration:
            reason = "missing_duration"

        precheck_texts.append("" if checked is None else str(checked))
        precheck_changed.append(bool(changed))
        normalized.append(normalize_arabic_transcript(checked))
        english_flags.append(eng)
        number_flags.append(num)
        bracket_flags.append(bracket)
        bracket_after_flags.append(bracket_after)
        too_short_flags.append(too_short)
        missing_duration_flags.append(missing_dur)
        drop_reasons.append(reason)
        keep_mask.append(reason is None)

    table = add_or_replace_column(
        table, policy.col("manual_normalized_transcript"), normalized, pa.string()
    )
    table = add_or_replace_column(table, "source_file", [str(source_file)] * n, pa.string())
    table = add_or_replace_column(
        table, "flag_contains_bracket_token", bracket_flags, pa.bool_()
    )
    table = add_or_replace_column(
        table, policy.col("flag_contains_english"), english_flags, pa.bool_()
    )
    table = add_or_replace_column(
        table, policy.col("flag_contains_number"), number_flags, pa.bool_()
    )
    table = add_or_replace_column(table, "flag_audio_too_short", too_short_flags, pa.bool_())
    table = add_or_replace_column(
        table, "flag_missing_duration", missing_duration_flags, pa.bool_()
    )
    if policy.precheck:
        # Preserve the untouched original next to the pre-checked text, the way the
        # Omnilingual v2/v3 scripts did with raw_text / precheck_text_vN.
        if "raw_text" not in table.column_names:
            table = add_or_replace_column(
                table, "raw_text", [None if t is None else str(t) for t in text_values], pa.string()
            )
        table = add_or_replace_column(
            table, policy.col("precheck_text"), precheck_texts, pa.string()
        )
        table = add_or_replace_column(
            table, policy.col("flag_precheck_changed_text"), precheck_changed, pa.bool_()
        )
        table = add_or_replace_column(
            table,
            policy.col("flag_contains_bracket_token_after_precheck"),
            bracket_after_flags,
            pa.bool_(),
        )

    for ds, eng, num, bracket, short, missing_dur, changed, reason in zip(
        dataset_values,
        english_flags,
        number_flags,
        bracket_flags,
        too_short_flags,
        missing_duration_flags,
        precheck_changed,
        drop_reasons,
    ):
        for counter in (totals, per_dataset[ds]):
            counter["total"] += 1
            counter["contains_english"] += int(eng)
            counter["contains_number"] += int(num)
            counter["contains_bracket_token"] += int(bracket)
            counter["audio_too_short"] += int(short)
            counter["missing_duration"] += int(missing_dur)
            counter["precheck_changed_text"] += int(changed)
            if reason is None:
                counter["kept"] += 1
            else:
                counter[f"dropped_{reason}"] += 1

    writer.write(clean_path, table.filter(pa.array(keep_mask, type=pa.bool_())))

    for reason in sorted({r for r in drop_reasons if r is not None}):
        mask = [r == reason for r in drop_reasons]
        writer.write(
            dropped_base / reason / f"{file_id}__dropped.parquet",
            table.filter(pa.array(mask, type=pa.bool_())),
        )


def run(ctx: RunContext, spec: StageSpec) -> StageResult:
    result = StageResult(stage_id=spec.id, stage=spec.stage)
    input_root = spec.require_path("input_root")
    output_root = spec.require_path("output_root")
    mode = spec.get("mode", "clean")
    if mode not in {"clean", "recover"}:
        raise SystemExit(f"clean stage: mode must be 'clean' or 'recover', got {mode!r}")

    # `recover` re-runs the same rules over rows a previous pass dropped, so its
    # outputs are named to keep the two generations of shards distinguishable.
    clean_dir_name = "recovered_clean" if mode == "recover" else "clean"
    dropped_dir_name = "still_dropped" if mode == "recover" else "dropped"

    if not input_root.exists():
        raise FileNotFoundError(f"clean stage input_root does not exist: {input_root}")

    if output_root.exists() and spec.get("overwrite", True):
        if output_root.is_symlink():
            raise SystemExit(
                f"Refusing to overwrite a symlinked output root ({output_root}). "
                "Point output_root at the real path."
            )
        shutil.rmtree(output_root)

    clean_dir = output_root / clean_dir_name
    dropped_dir = output_root / dropped_dir_name
    reports_dir = ctx.stage_reports_dir(spec.id)
    clean_dir.mkdir(parents=True, exist_ok=True)
    dropped_dir.mkdir(parents=True, exist_ok=True)

    defaults = spec.get("defaults", {}) or {}
    entries = spec.require("datasets")
    grand_totals: Counter = Counter()
    per_dataset_all: dict[str, Counter] = defaultdict(Counter)
    manifest: list[dict[str, Any]] = []

    for entry in entries:
        policy = DatasetPolicy(entry, defaults)
        files = discover(input_root, policy)
        ctx.log(f"dataset {policy.name}: {len(files)} file(s) matched {policy.globs}")
        if not files:
            raise FileNotFoundError(
                f"clean stage: dataset {policy.name!r} matched no files under {input_root} "
                f"(globs={policy.globs}, recursive={policy.recursive})"
            )
        if ctx.dry_run:
            manifest.append({"dataset": policy.name, "files": [str(p) for p in files]})
            continue

        totals: Counter = Counter()
        per_dataset: dict[str, Counter] = defaultdict(Counter)
        with TableShardWriter(policy.compression) as writer:
            for path in files:
                file_id = stable_file_id(path, input_root)
                clean_path = clean_dir / f"{file_id}__clean.parquet"
                rows_before = totals["total"]
                for table in iter_table_batches(path, policy.batch_size):
                    clean_table(
                        table,
                        path,
                        input_root,
                        policy,
                        writer,
                        clean_path,
                        dropped_dir,
                        totals,
                        per_dataset,
                    )
                manifest.append(
                    {
                        "dataset": policy.name,
                        "input_file": str(path),
                        "file_id": file_id,
                        "rows": totals["total"] - rows_before,
                        "precheck": policy.precheck,
                    }
                )
                ctx.log(f"  {path.name}: total={totals['total']:,} kept={totals['kept']:,}")

        result.counts[policy.name] = dict(totals)
        grand_totals.update(totals)
        for ds, counter in per_dataset.items():
            per_dataset_all[ds].update(counter)

    if not ctx.dry_run:
        ctx.write_json(reports_dir / "manifest.json", manifest)
        ctx.write_json(
            reports_dir / "cleaning_report.json",
            {
                "mode": mode,
                "input_root": str(input_root),
                "output_root": str(output_root),
                "totals": dict(grand_totals),
                "per_dataset": {k: dict(v) for k, v in per_dataset_all.items()},
            },
        )
        summary_lines = [
            f"mode: {mode}",
            f"total rows: {grand_totals['total']:,}",
            f"kept rows: {grand_totals['kept']:,}",
        ] + [
            f"dropped {reason}: {grand_totals[f'dropped_{reason}']:,}"
            for reason in DROP_REASONS
            if grand_totals[f"dropped_{reason}"]
        ]
        ctx.write_text(reports_dir / "summary.txt", "\n".join(summary_lines))
        ctx.log(" | ".join(summary_lines))

    result.counts["__totals__"] = dict(grand_totals)
    result.outputs["clean"] = str(clean_dir)
    result.outputs["dropped"] = str(dropped_dir)
    return result
