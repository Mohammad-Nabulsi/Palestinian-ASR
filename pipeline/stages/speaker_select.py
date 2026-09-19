"""Stage -- speaker_select: choose which speakers form val / test / train.

Runs *after both* dialect stages:

    dialect(text) --> dialect(audio) --> speaker_select --> split

It replaces the earlier ``speaker_aggregate`` stage, which sat between the two
dialect passes and only smoothed text scores per speaker. That placement could
not see the audio evidence at all, so it could not answer the question the split
actually needs answered -- *which speakers are we most confident keep speaking
Levantine* -- and it left ``split`` assigning rows by hash, which is what let the
same recording appear in train and test at once.

Here the audio stage's ``row_probabilities.jsonl`` is the input, because it
carries both modalities per row (audio ``top_label``/``label_scores`` plus the
text ``text_top_label``/``text_label_scores`` copied forward). Every row in it is
a row both models scored. Those are grouped per speaker, scored (see
``pipeline/speaker_scoring.py``), ranked, and cut into consecutive blocks whose
hour targets come from config. Blocks are filled from the top, so name them in
the order of how much confidence each needs -- ``val`` and ``test`` first, then
``train``.

The output is ``speaker_assignments.json``: speaker -> split, which ``split``
consumes instead of hashing. Speakers ranked below the last block are left
unassigned; what happens to their rows is ``split``'s ``unassigned_split``
decision, not this stage's.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path
from typing import Any

from ..config import StageSpec
from ..context import RunContext, StageResult
from ..speaker_scoring import (
    accumulate_audio_row,
    build_speakers,
    carve_blocks,
    cumulative_curve,
    extract_group_key,
    new_pool_acc,
    rank,
    verify_disjoint,
)


def load_pool(path: Path, align_threshold: float, ctx: RunContext) -> tuple[dict, dict]:
    """Accumulate the dual-scored rows per speaker.

    Deduplicates on (source, source_file, row_idx) because a scan directory can
    hold several complementary candidate-band runs, and a row counted twice
    would double-weight its speaker.
    """
    pool: dict[tuple[str, str], dict] = defaultdict(new_pool_acc)
    seen: set[tuple[str, str, int]] = set()
    stats = {"rows_read": 0, "rows_ok": 0, "rows_skipped_status": 0,
             "rows_duplicate": 0, "rows_unresolved_group_key": 0}

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            stats["rows_read"] += 1

            row_key = (record["source"], record["source_file"], int(record["row_idx"]))
            if row_key in seen:
                stats["rows_duplicate"] += 1
                continue
            seen.add(row_key)

            group_key = extract_group_key(record["source"], record["uid"])
            if group_key is None:
                stats["rows_unresolved_group_key"] += 1
                continue
            acc = pool[(record["source"], group_key)]

            if record.get("status") != "ok":
                stats["rows_skipped_status"] += 1
                acc["skipped_short"] += 1
                continue

            accumulate_audio_row(acc, record, align_threshold)
            stats["rows_ok"] += 1

    ctx.log(f"pool: {stats['rows_ok']:,} dual-scored row(s) over {len(pool):,} speaker(s)")
    return pool, stats


def run(ctx: RunContext, spec: StageSpec) -> StageResult:
    result = StageResult(stage_id=spec.id, stage=spec.stage)

    audio_row_probs = spec.require_path("audio_row_probs")
    output_dir = spec.require_path("output_dir")
    threshold = float(spec.get("threshold", 0.80))
    align_threshold = float(spec.get("align_threshold", 0.80))
    rank_by = spec.get("rank_by", "score_product_align")
    hours_field = spec.get("hours_field", "hours")
    min_rows = int(spec.get("min_rows_per_speaker", 1))
    min_hours = float(spec.get("min_hours_per_speaker", 0.0))
    blocks_spec = spec.require("blocks")

    assignments_path = output_dir / "speaker_assignments.json"
    if ctx.dry_run:
        result.counts["__totals__"] = {"status": "dry_run"}
        result.outputs["speaker_assignments"] = str(assignments_path)
        return result

    if not audio_row_probs.exists():
        raise FileNotFoundError(f"speaker_select: audio_row_probs not found: {audio_row_probs}")
    output_dir.mkdir(parents=True, exist_ok=True)

    pool, row_stats = load_pool(audio_row_probs, align_threshold, ctx)
    speakers = build_speakers(pool, None, threshold, min_rows, min_hours)
    if not speakers:
        raise SystemExit("speaker_select: no speakers survived min_rows/min_hours")

    ranked = rank(speakers, rank_by, "rows_dual_scored")
    block_names = [b["name"] for b in blocks_spec]
    block_hours = [float(b["hours"]) for b in blocks_spec]
    blocks, assigned = carve_blocks(ranked, block_hours, block_names, hours_field, rank_by)

    disjoint = verify_disjoint(assigned)
    if not disjoint["disjoint"]:
        # Consecutive slices cannot overlap, so this means the ranking held one
        # speaker twice -- a duplicated row source, not a carving bug.
        raise SystemExit(
            f"speaker_select: blocks are not speaker-disjoint: "
            f"{disjoint['speakers_in_more_than_one_block']}"
        )

    for block in blocks:
        if not block["filled"]:
            ctx.log(
                f"WARNING: block {block['name']!r} wanted {block['target_hours']}h but the "
                f"ranking only had {block['hours']}h left"
            )

    selected = [s for s in assigned if s.get("block")]
    payload = {
        "meta": {
            "audio_row_probs": str(audio_row_probs),
            "threshold": threshold,
            "align_threshold": align_threshold,
            "rank_by": rank_by,
            "hours_field": hours_field,
            "min_rows_per_speaker": min_rows,
            "min_hours_per_speaker": min_hours,
            "blocks": blocks,
        },
        "assignments": [
            {"source": s["source"], "speaker_key": s["speaker_key"], "split": s["block"],
             "rows_dual_scored": s["rows_dual_scored"], "hours": round(s["hours"], 4),
             "score": s[rank_by]}
            for s in selected
        ],
    }
    ctx.write_json(assignments_path, payload)

    per_source: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))
    for s in selected:
        per_source[s["source"]][s["block"]] += 1

    summary: dict[str, Any] = {
        "row_stats": row_stats,
        "pool": {
            "speakers": len(speakers),
            "rows_dual_scored": sum(s["rows_dual_scored"] for s in speakers),
            "total_hours": round(sum(s["hours"] for s in speakers), 2),
            "total_hours_audio_lev": round(sum(s["hours_audio_lev"] for s in speakers), 2),
            "total_hours_both_agree": round(sum(s["hours_both_agree"] for s in speakers), 2),
        },
        "settings": payload["meta"],
        "blocks": blocks,
        "disjointness": disjoint,
        "speakers_per_source_per_block": {k: dict(v) for k, v in per_source.items()},
        "unassigned_speakers": len(speakers) - len(selected),
        "unassigned_hours": round(sum(s["hours"] for s in assigned if not s.get("block")), 2),
        "cumulative_hours_curve": cumulative_curve(
            ranked, [b["hours"] for b in blocks], hours_field, rank_by),
    }
    ctx.write_json(output_dir / "summary.json", summary)
    ctx.write_json(ctx.stage_reports_dir(spec.id) / "summary.json", summary)

    result.counts = {
        "speakers_total": len(speakers),
        "speakers_assigned": len(selected),
        "blocks": {b["name"]: {"speakers": b["speakers"], "hours": b["hours"]} for b in blocks},
        "unassigned_speakers": summary["unassigned_speakers"],
    }
    result.outputs["speaker_assignments"] = str(assignments_path)
    ctx.log(f"speaker_select done: {result.counts['blocks']}")
    return result
