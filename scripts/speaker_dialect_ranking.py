#!/usr/bin/env python3
"""Rank speakers by dialect confidence and carve speaker-disjoint hour blocks.

Analysis CLI over the raw dialect-ID scans. The scoring itself lives in
`pipeline/speaker_scoring.py`, shared with the `speaker_select` pipeline stage
that produces the real split -- this script is the offline way to explore the
same numbers (alternative scores, alignment diagnostics, sample dumps) without
running the pipeline.

Why the scans and not `data_lev_custom_split_v1`: the curated parquet stores only
a scalar `text_lev`/`audio_lev` per row, and hard voting needs each row's argmax
over the full 5-class distribution. The scans keep it, and their audio runs carry
BOTH modalities inline per row:

    audio:  top_label, label_scores{Levantine, MSA, Egyptian, Gulf, Maghrebi}
    text:   text_top_label, text_label_scores{LEV, GLF, EGY, MSA, MAGHREB}
    plus    duration_sec, source, uid (embeds the speaker key)

so an audio-scan row *is* a row with both classifications. The text runs cover
every qasr+masc_c row and are read separately (`--text-scan`) for each speaker's
unconditioned text statistics.

One caveat the output makes explicit: the audio model only ever ran on rows the
text model had already put at p(LEV) >= 0.5 *with LEV as argmax*, so on the
dual-scored pool the text hard vote is 1.0 for every row by construction. The
informative text hard vote is the one over the full text scan.

Usage:
    python scripts/speaker_dialect_ranking.py \\
        --audio-scan dialect_id_scans/dialect_scan_badrex_*/row_probabilities.jsonl \\
        --text-scan  dialect_id_scans/text_dialect_scan_*/row_probabilities.jsonl \\
        --rank-by score_product_align \\
        --block-hours 8 8 200 --block-names val test train \\
        --emit-samples 50 --out-dir outputs/speaker_ranking
"""
from __future__ import annotations

import argparse
import json
import math
import random
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from pipeline.speaker_scoring import (  # noqa: E402
    TEXT_LEV,
    accumulate_audio_row,
    audio_prob,
    build_speakers,
    carve_blocks,
    cumulative_curve,
    extract_group_key,
    new_pool_acc,
    rank,
    text_prob,
    verify_disjoint,
)


# ------------------------------------------------------------------ loading


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def load_audio_scans(
    paths: Iterable[Path], align_threshold: float
) -> tuple[dict[tuple[str, str], dict], dict]:
    """Dual-scored rows, accumulated per speaker.

    Deduplicates on (source, source_file, row_idx): the four candidate-band runs
    are meant to be complementary, and a row counted twice would silently
    double-weight that speaker.
    """
    pool: dict[tuple[str, str], dict] = defaultdict(new_pool_acc)
    seen: set[tuple[str, str, int]] = set()
    stats = {
        "rows_read": 0, "rows_ok": 0, "rows_skipped_status": 0,
        "rows_duplicate": 0, "rows_unresolved_group_key": 0,
        "text_argmax_lev_rows": 0, "per_file": {},
    }

    for path in paths:
        file_stats = {"rows": 0, "ok": 0, "duplicate": 0, "unresolved": 0}
        for record in iter_jsonl(path):
            stats["rows_read"] += 1
            file_stats["rows"] += 1

            row_key = (record["source"], sys.intern(record["source_file"]), int(record["row_idx"]))
            if row_key in seen:
                stats["rows_duplicate"] += 1
                file_stats["duplicate"] += 1
                continue
            seen.add(row_key)

            group_key = extract_group_key(record["source"], record["uid"])
            if group_key is None:
                stats["rows_unresolved_group_key"] += 1
                file_stats["unresolved"] += 1
                continue
            acc = pool[(record["source"], group_key)]

            if record.get("status") != "ok":
                # `skipped_short`: clip too short for the audio model, so the row
                # has no audio classification and is not part of the pool.
                stats["rows_skipped_status"] += 1
                acc["skipped_short"] += 1
                continue

            accumulate_audio_row(acc, record, align_threshold)
            stats["rows_ok"] += 1
            file_stats["ok"] += 1
            if record.get("text_top_label") == TEXT_LEV:
                stats["text_argmax_lev_rows"] += 1

        stats["per_file"][str(path)] = file_stats
        print(f"{path.parent.name}: {file_stats['rows']} rows, {file_stats['ok']} dual-scored", file=sys.stderr)

    return pool, stats


def load_text_scans(paths: Iterable[Path]) -> tuple[dict[tuple[str, str], dict], dict]:
    """Every text-scored row per speaker -- the unconditioned text view."""
    full: dict[tuple[str, str], dict] = defaultdict(lambda: {"n": 0, "sum": 0.0, "votes": 0})
    seen: set[tuple[str, str, int]] = set()
    stats = {"rows_read": 0, "rows_ok": 0, "rows_duplicate": 0, "rows_unresolved_group_key": 0, "per_file": {}}

    for path in paths:
        file_stats = {"rows": 0, "ok": 0}
        for record in iter_jsonl(path):
            stats["rows_read"] += 1
            file_stats["rows"] += 1
            if record.get("status", "ok") != "ok":
                continue

            row_key = (record["source"], sys.intern(record["source_file"]), int(record["row_idx"]))
            if row_key in seen:
                stats["rows_duplicate"] += 1
                continue
            seen.add(row_key)

            group_key = extract_group_key(record["source"], record["uid"])
            if group_key is None:
                stats["rows_unresolved_group_key"] += 1
                continue

            acc = full[(record["source"], group_key)]
            acc["n"] += 1
            acc["sum"] += text_prob(record)
            if record.get("top_label") == TEXT_LEV:
                acc["votes"] += 1
            stats["rows_ok"] += 1
            file_stats["ok"] += 1

        stats["per_file"][str(path)] = file_stats
        print(f"{path.parent.name}: {file_stats['rows']} text rows", file=sys.stderr)

    return full, stats


# ------------------------------------------------------------- diagnostics


def spearman(xs: list[float], ys: list[float]) -> float | None:
    """Rank correlation with average ranks for ties (no scipy dependency)."""
    n = len(xs)
    if n < 2:
        return None

    def ranks(values: list[float]) -> list[float]:
        order = sorted(range(n), key=lambda i: values[i])
        out = [0.0] * n
        i = 0
        while i < n:
            j = i
            while j + 1 < n and values[order[j + 1]] == values[order[i]]:
                j += 1
            avg = (i + j) / 2.0 + 1.0
            for k in range(i, j + 1):
                out[order[k]] = avg
            i = j + 1
        return out

    rx, ry = ranks(xs), ranks(ys)
    mx, my = sum(rx) / n, sum(ry) / n
    num = sum((a - mx) * (b - my) for a, b in zip(rx, ry))
    den = math.sqrt(sum((a - mx) ** 2 for a in rx) * sum((b - my) ** 2 for b in ry))
    return num / den if den else None


def paired_spearman(speakers: list[dict[str, Any]], field_a: str, field_b: str) -> dict[str, Any] | None:
    """Rank correlation over the speakers carrying both fields.

    Not every speaker has the full-text fields -- a text scan covers one source,
    so running with only the masc_c scan leaves qasr speakers without them.
    """
    usable = [s for s in speakers if field_a in s and field_b in s]
    if len(usable) < 2:
        return None
    return {
        "speakers": len(usable),
        "rho": spearman([s[field_a] for s in usable], [s[field_b] for s in usable]),
    }


def agreement(speakers: list[dict[str, Any]], soft_field: str, hard_field: str,
              mean_field: str, frac_field: str) -> dict[str, Any]:
    """Soft-vs-hard alignment for one modality."""
    usable = [s for s in speakers if soft_field in s and hard_field in s]
    if not usable:
        return {"speakers": 0}
    matrix = {"soft_lev_hard_lev": 0, "soft_lev_hard_non": 0, "soft_non_hard_lev": 0, "soft_non_hard_non": 0}
    hours = {k: 0.0 for k in matrix}
    for s in usable:
        cell = f"soft_{'lev' if s[soft_field] else 'non'}_hard_{'lev' if s[hard_field] else 'non'}"
        matrix[cell] += 1
        hours[cell] += s["hours"]
    agreed = matrix["soft_lev_hard_lev"] + matrix["soft_non_hard_non"]
    return {
        "speakers": len(usable),
        "agreement_rate": agreed / len(usable),
        "confusion_speakers": matrix,
        "confusion_hours": {k: round(v, 2) for k, v in hours.items()},
        "rank_correlation_soft_vs_hard": spearman(
            [s[mean_field] for s in usable], [s[frac_field] for s in usable]
        ),
    }


# ----------------------------------------------------------------- outputs


SELECTION_COLUMNS = (
    "block", "source", "speaker_key", "rows_dual_scored", "rows_both_agree", "rows_text_total",
    "hours", "hours_audio_lev", "hours_both_agree", "text_mean_prob", "text_full_vote_frac",
    "text_full_vote_lb", "audio_mean_prob", "audio_vote_frac", "audio_vote_lb", "audio_prob_sd",
    "align_frac", "align_lb", "score_sum", "score_product", "score_sum_align",
    "score_product_align", "combined_vote_lb",
)


def emit_samples(
    paths: Iterable[Path],
    assignments: list[dict[str, Any]],
    per_block: int,
    align_threshold: float,
    seed: int = 42,
    max_per_speaker: int = 3,
) -> list[dict[str, Any]]:
    """Pull real rows out of the selected speakers, for eyeball inspection.

    A second pass over the scans, because aggregation keeps no row detail. The
    sample is uniform-random within a block (seeded), capped per speaker so it
    spans many speakers rather than a handful of long recordings -- an inspector
    needs typical rows, including the bad ones, not a curated best-of.
    """
    wanted: dict[tuple[str, str], str] = {
        (s["source"], s["speaker_key"]): s["block"] for s in assignments if s.get("block")
    }
    by_block: dict[str, list[dict[str, Any]]] = defaultdict(list)

    for path in paths:
        for record in iter_jsonl(path):
            if record.get("status") != "ok":
                continue
            group_key = extract_group_key(record["source"], record["uid"])
            block = wanted.get((record["source"], group_key)) if group_key else None
            if block is None:
                continue
            t_prob = text_prob(record, prefix="text_")
            a_prob = audio_prob(record)
            by_block[block].append({
                "block": block,
                "source": record["source"],
                "speaker_key": group_key,
                "uid": record["uid"],
                "duration_sec": record.get("duration_sec"),
                "text": record.get("text", ""),
                "text_top_label": record.get("text_top_label"),
                "text_lev": round(t_prob, 4),
                "audio_top_label": record.get("top_label"),
                "audio_lev": round(a_prob, 4),
                "both_agree": t_prob >= align_threshold and a_prob >= align_threshold,
            })

    out: list[dict[str, Any]] = []
    for block, rows in by_block.items():
        rng = random.Random(f"{seed}:{block}")
        rng.shuffle(rows)
        taken: dict[str, int] = defaultdict(int)
        picked: list[dict[str, Any]] = []
        for row in rows:
            if len(picked) >= per_block:
                break
            key = f"{row['source']}:{row['speaker_key']}"
            if taken[key] >= max_per_speaker:
                continue
            taken[key] += 1
            picked.append(row)
        out.extend(picked)
        print(f"samples[{block}]: {len(picked)} of {len(rows)} rows, {len(taken)} speakers", file=sys.stderr)
    return out


def write_selection_csv(path: Path, assignments: list[dict[str, Any]]) -> int:
    """The speakers that landed in a block -- small enough to keep in git,
    unlike the full per-speaker JSONL."""
    import csv

    selected = [s for s in assignments if s.get("block")]
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(SELECTION_COLUMNS), extrasaction="ignore")
        writer.writeheader()
        for row in selected:
            writer.writerow({k: row.get(k) for k in SELECTION_COLUMNS})
    return len(selected)


def write_assignments(path: Path, assignments: list[dict[str, Any]], meta: dict[str, Any]) -> int:
    """speaker -> split, in the shape `pipeline/stages/split.py` consumes."""
    selected = [s for s in assignments if s.get("block")]
    payload = {
        "meta": meta,
        "assignments": [
            {"source": s["source"], "speaker_key": s["speaker_key"], "split": s["block"],
             "rows_dual_scored": s["rows_dual_scored"], "hours": round(s["hours"], 4),
             "score": s[meta["rank_by"]]}
            for s in selected
        ],
    }
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return len(selected)


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def slim(speaker: dict[str, Any]) -> dict[str, Any]:
    keys = ("source", "speaker_key", "rows_dual_scored", "rows_both_agree", "hours",
            "text_mean_prob", "audio_mean_prob", "align_frac", "align_lb",
            "score_sum", "score_product", "score_sum_align", "score_product_align")
    return {k: speaker[k] for k in keys if k in speaker}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--audio-scan", nargs="+", required=True, type=Path,
                        help="row_probabilities.jsonl from each audio (badrex) run")
    parser.add_argument("--text-scan", nargs="*", default=[], type=Path,
                        help="row_probabilities.jsonl from each text (marbertv2) run")
    parser.add_argument("--threshold", type=float, default=0.80, help="soft (mean-prob) lev threshold")
    parser.add_argument("--align-threshold", type=float, default=0.80,
                        help="a row counts as agreed when BOTH models clear this")
    parser.add_argument("--min-rows-per-speaker", type=int, default=1)
    parser.add_argument("--min-hours-per-speaker", type=float, default=0.0)
    parser.add_argument("--rank-by", default="score_product_align",
                        choices=["score_sum", "score_product", "score_sum_align", "score_product_align",
                                 "combined_prob", "combined_vote_lb", "combined_prob_full_text"],
                        help="score the blocks are carved on")
    parser.add_argument("--hours-field", default="hours",
                        choices=["hours", "hours_audio_lev", "hours_both_agree"],
                        help="hours counted per speaker when filling blocks")
    parser.add_argument("--block-hours", nargs="+", type=float, default=[8.0, 8.0, 200.0])
    parser.add_argument("--block-names", nargs="+", default=["val", "test", "train"])
    parser.add_argument("--block-min-rows", nargs="*", type=int, default=None,
                        help="per-block floor on rows/speaker, e.g. 20 20 0 to keep the "
                             "eval blocks well-evidenced while train takes short speakers")
    parser.add_argument("--curve-marks", nargs="*", type=float,
                        default=[1, 2, 4, 8, 16, 24, 32, 50, 100, 200])
    parser.add_argument("--emit-samples", type=int, default=0,
                        help="rows per block to dump for inspection (0 = off)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/speaker_ranking"))
    args = parser.parse_args()

    if len(args.block_hours) != len(args.block_names):
        parser.error("--block-hours and --block-names must have the same length")
    if args.block_min_rows and len(args.block_min_rows) != len(args.block_names):
        parser.error("--block-min-rows must have one value per block")

    pool, audio_stats = load_audio_scans(args.audio_scan, args.align_threshold)
    full_text, text_stats = load_text_scans(args.text_scan) if args.text_scan else ({}, {})

    speakers = build_speakers(
        pool, full_text, args.threshold, args.min_rows_per_speaker, args.min_hours_per_speaker)
    if not speakers:
        raise SystemExit("no speakers survived the --min-rows/--min-hours filters")

    has_full_text = any("text_full_vote_frac" in s for s in speakers)
    text_rank_field = "text_full_vote_frac" if has_full_text else "text_vote_frac"

    by_text = rank(speakers, text_rank_field, "text_full_mean_prob" if has_full_text else "text_mean_prob")
    by_audio = rank(speakers, "audio_vote_frac", "audio_mean_prob")
    by_score = rank(speakers, args.rank_by, "rows_dual_scored")
    if not by_score:
        raise SystemExit(f"--rank-by {args.rank_by} is not available (need --text-scan for the full-text scores)")

    blocks, assignments = carve_blocks(
        by_score, args.block_hours, args.block_names, args.hours_field, args.rank_by,
        args.block_min_rows)
    disjoint = verify_disjoint(assignments)
    curve = cumulative_curve(by_score, args.curve_marks, args.hours_field, args.rank_by)

    per_source: dict[str, dict[str, float]] = defaultdict(lambda: {"speakers": 0, "hours": 0.0, "rows": 0})
    for s in speakers:
        bucket = per_source[s["source"]]
        bucket["speakers"] += 1
        bucket["hours"] += s["hours"]
        bucket["rows"] += s["rows_dual_scored"]

    alignment = {
        "text_on_dual_scored_pool": agreement(
            speakers, "text_soft_lev", "text_hard_lev", "text_mean_prob", "text_vote_frac"),
        "audio": agreement(
            speakers, "audio_soft_lev", "audio_hard_lev", "audio_mean_prob", "audio_vote_frac"),
    }
    if has_full_text:
        alignment["text_on_full_text_scan"] = agreement(
            speakers, "text_full_soft_lev", "text_full_hard_lev",
            "text_full_mean_prob", "text_full_vote_frac")

    settings = {
        "soft_threshold": args.threshold,
        "align_threshold": args.align_threshold,
        "hard_vote": "per-row argmax label (top_label)",
        "rank_by": args.rank_by,
        "hours_field": args.hours_field,
        "min_rows_per_speaker": args.min_rows_per_speaker,
        "min_hours_per_speaker": args.min_hours_per_speaker,
        "block_hours": args.block_hours,
        "block_names": args.block_names,
        "block_min_rows": args.block_min_rows,
    }

    summary = {
        "inputs": {
            "audio_scans": [str(p) for p in args.audio_scan],
            "text_scans": [str(p) for p in args.text_scan],
        },
        "row_stats": {"audio": audio_stats, "text": text_stats},
        "settings": settings,
        "pool": {
            "speakers": len(speakers),
            "rows_dual_scored": sum(s["rows_dual_scored"] for s in speakers),
            "total_hours": round(sum(s["hours"] for s in speakers), 2),
            "total_hours_audio_lev": round(sum(s["hours_audio_lev"] for s in speakers), 2),
            "total_hours_both_agree": round(sum(s["hours_both_agree"] for s in speakers), 2),
            "per_source": {
                k: {"speakers": v["speakers"], "hours": round(v["hours"], 2), "rows": v["rows"]}
                for k, v in sorted(per_source.items())
            },
        },
        "soft_vs_hard_alignment": alignment,
        "ranking_agreement": {
            "text_vs_audio_hard": paired_spearman(speakers, text_rank_field, "audio_vote_frac"),
            "score_vs_text_hard": paired_spearman(speakers, args.rank_by, text_rank_field),
            "score_vs_audio_hard": paired_spearman(speakers, args.rank_by, "audio_vote_frac"),
            "sum_align_vs_product_align": paired_spearman(speakers, "score_sum_align", "score_product_align"),
        },
        "blocks": blocks,
        "disjointness": disjoint,
        "cumulative_hours_curve": curve,
        "top_15_by_rank_field": [slim(s) for s in by_score[:15]],
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(args.out_dir / "speakers_by_score.jsonl", assignments)
    write_jsonl(args.out_dir / "speakers_by_text_vote.jsonl", by_text)
    write_jsonl(args.out_dir / "speakers_by_audio_vote.jsonl", by_audio)
    write_selection_csv(args.out_dir / "selection.csv", assignments)
    write_assignments(args.out_dir / "speaker_assignments.json", assignments, settings)

    if args.emit_samples:
        samples = emit_samples(args.audio_scan, assignments, args.emit_samples, args.align_threshold)
        write_jsonl(args.out_dir / "samples.jsonl", samples)
        summary["samples"] = {"per_block_requested": args.emit_samples, "written": len(samples)}

    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    if not disjoint["disjoint"]:
        raise SystemExit("BUG: carved blocks are not speaker-disjoint (see summary.json)")
    print(f"wrote {len(speakers)} speaker row(s) to {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
