"""Stage -- speaker_aggregate: average text-dialect scores per speaker before
the audio model runs.

Sits between ``dialect(text)`` and ``dialect(audio)`` in the data flow:

    dialect(text) --> speaker_aggregate --> dialect(audio) --> split

The text stage scores every row independently, so rows from the same speaker
can land on both sides of the lev/non_lev threshold. This stage re-groups
those per-row scores by a per-source speaker key -- ``speaker_id`` for qasr
(a real speaker id), ``video_id`` for masc_c (no real speaker id exists, so
the video is the closest available proxy) -- and replaces every row's score
with its group's mean before anything downstream reads it.

The audio dialect stage is not touched: it still reads a
``row_probabilities.jsonl`` via ``candidates_from`` in the same schema
``dialect.py`` already produces, so pointing its ``candidates_from.path`` at
this stage's output is the only wiring change needed. The audio model still
runs strictly after this stage, just against speaker-averaged candidates
instead of raw per-row ones.

Sources with no entry in ``speaker_key_columns`` (omni, layla, casa/*) pass
through untouched: each of their rows is its own singleton group, so their
score is unchanged.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import StageSpec
from ..context import RunContext, StageResult
from ..shards import iter_table_batches

_ROW_GROUP = "__row__"


def _load_text_records(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            if record.get("status", "ok") == "ok":
                records.append(record)
    return records


def _resolve_speaker_keys(
    records: list[dict[str, Any]],
    speaker_key_columns: dict[str, str],
) -> dict[tuple[str, int], Any]:
    """(source_file, row_idx) -> speaker key, resolved by re-reading each shard
    once (in the same row order dialect.py used to assign row_idx) and pulling
    the configured key column."""
    by_file: dict[str, tuple[str, set[int]]] = {}
    for rec in records:
        source = rec["source"]
        if source not in speaker_key_columns:
            continue
        source_file = rec["source_file"]
        if source_file not in by_file:
            by_file[source_file] = (source, set())
        by_file[source_file][1].add(rec["row_idx"])

    keys: dict[tuple[str, int], Any] = {}
    for source_file, (source, wanted) in by_file.items():
        col_name = speaker_key_columns[source]
        row_idx = -1
        for table in iter_table_batches(Path(source_file), batch_size=20_000):
            if col_name not in table.column_names:
                raise SystemExit(
                    f"speaker_aggregate: column {col_name!r} not found in "
                    f"{source_file} (source={source}); has {table.column_names}"
                )
            values = table[col_name].to_pylist()
            for i in range(table.num_rows):
                row_idx += 1
                if row_idx in wanted:
                    keys[(source_file, row_idx)] = values[i]
    return keys


def run(ctx: RunContext, spec: StageSpec) -> StageResult:
    result = StageResult(stage_id=spec.id, stage=spec.stage)

    text_row_probs = spec.require_path("text_row_probs")
    output_dir = spec.require_path("output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)

    label = spec.get("label", "LEV")
    speaker_key_columns: dict[str, str] = spec.get("speaker_key_columns", {})

    row_probs_path = output_dir / "row_probabilities.jsonl"
    speaker_scores_path = output_dir / "speaker_scores.jsonl"

    if ctx.dry_run:
        result.counts["__totals__"] = {"status": "dry_run"}
        result.outputs["row_probabilities"] = str(row_probs_path)
        result.outputs["speaker_scores"] = str(speaker_scores_path)
        return result

    ctx.log(f"reading text scores from {text_row_probs}")
    records = _load_text_records(text_row_probs)
    ctx.log(f"{len(records):,} scored row(s) loaded")

    ctx.log(f"resolving speaker keys via {speaker_key_columns}")
    keys = _resolve_speaker_keys(records, speaker_key_columns)

    groups: dict[tuple[str, Any], list[dict[str, Any]]] = defaultdict(list)
    for rec in records:
        source = rec["source"]
        speaker_key = keys.get((rec["source_file"], rec["row_idx"])) if source in speaker_key_columns else None
        if speaker_key is None:
            group_id = (source, (_ROW_GROUP, rec["source_file"], rec["row_idx"]))
        else:
            group_id = (source, speaker_key)
        groups[group_id].append(rec)

    counts = {
        "rows_in": len(records),
        "groups": 0,
        "rows_with_speaker_key": 0,
        "rows_without_speaker_key": 0,
    }
    per_source_groups: dict[str, int] = defaultdict(int)

    with speaker_scores_path.open("w", encoding="utf-8") as speaker_handle, \
            row_probs_path.open("w", encoding="utf-8") as row_handle:
        for (source, speaker_key), group_records in groups.items():
            counts["groups"] += 1
            per_source_groups[source] += 1
            is_real_key = not (isinstance(speaker_key, tuple) and speaker_key and speaker_key[0] == _ROW_GROUP)
            if is_real_key:
                counts["rows_with_speaker_key"] += len(group_records)
            else:
                counts["rows_without_speaker_key"] += len(group_records)

            scores = [
                float(r.get("target_label_score", (r.get("label_scores") or {}).get(label, 0.0)))
                for r in group_records
            ]
            mean_score = sum(scores) / len(scores)

            all_labels: set[str] = set()
            for r in group_records:
                all_labels.update((r.get("label_scores") or {}).keys())
            mean_label_scores = {
                lbl: sum(float((r.get("label_scores") or {}).get(lbl, 0.0)) for r in group_records) / len(group_records)
                for lbl in all_labels
            }

            speaker_handle.write(json.dumps({
                "source": source,
                "speaker_key": speaker_key if is_real_key else None,
                "row_count": len(group_records),
                "mean_target_label_score": mean_score,
                "min_target_label_score": min(scores),
                "max_target_label_score": max(scores),
                "mean_label_scores": mean_label_scores,
            }, ensure_ascii=False) + "\n")

            for r in group_records:
                row_handle.write(json.dumps({
                    "source": r["source"],
                    "source_file": r["source_file"],
                    "row_idx": r["row_idx"],
                    "status": "ok",
                    "speaker_key": speaker_key if is_real_key else None,
                    "speaker_group_size": len(group_records),
                    "row_target_label_score": float(r.get("target_label_score", 0.0)),
                    "target_label": r.get("target_label", label),
                    "target_label_score": mean_score,
                    "label_scores": mean_label_scores,
                }, ensure_ascii=False) + "\n")

    summary = {
        "text_row_probs": str(text_row_probs),
        "speaker_key_columns": speaker_key_columns,
        "label": label,
        "counts": counts,
        "groups_per_source": dict(per_source_groups),
    }
    ctx.write_json(output_dir / "summary.json", summary)
    ctx.write_json(ctx.stage_reports_dir(spec.id) / "summary.json", summary)

    result.counts = counts
    result.outputs["row_probabilities"] = str(row_probs_path)
    result.outputs["speaker_scores"] = str(speaker_scores_path)
    ctx.log(f"speaker_aggregate done: {counts}")
    return result
