#!/usr/bin/env python3
"""Rank speakers by dialect confidence and carve speaker-disjoint hour blocks.

Third script in the speaker-disjoint audit series, after
`r2_speaker_group_disjointness.py` (does the current split leak? yes) and
`r2_speaker_group_dialect_classify.py` (group-level lev labeling from
`data_lev_custom_split_v1`). This one builds the *ranking* a speaker-disjoint
split can be cut from, and it reads the raw dialect-ID scans rather than the
curated parquet, because only the scans carry each row's full per-label
distribution -- i.e. the argmax needed for hard voting.

Input is `dialect_id_scans.tar.zst` (R2: `backup/transfer/`, see
`DIALECT_ID_SCANS.md`), whose audio runs already carry BOTH modalities per row:

    audio:  top_label, label_scores{Levantine, MSA, Egyptian, Gulf, Maghrebi}
    text:   text_top_label, text_label_scores{LEV, GLF, EGY, MSA, MAGHREB}
    plus    duration_sec, source, uid (embeds the speaker key)

so an audio-scan row *is* a row with both classifications (step 1). The text
runs cover every qasr+masc_c row and are read separately (`--text-scan`) to get
each speaker's unconditioned text statistics.

What it computes, per speaker (qasr recording_id / masc_c video_id):

  2. group every dual-scored row by speaker
  3. text score two ways -- soft (mean p(LEV)) and hard (majority over per-row
     argmax) -- plus their alignment (confusion + rank correlation)
  4. same for audio (mean p(Levantine) / majority over argmax)
  5. combined = mean(text prob) * mean(audio prob)
  6. three rankings: hard-vote text, hard-vote audio, combined product
  7. the product is the primary ranking
  8. walk the product ranking and carve consecutive, speaker-disjoint hour
     blocks (default 8h + 8h), reporting speakers/hours each needs

Caveat the output makes explicit: the audio model was only ever run on rows the
text model had already put at p(LEV) >= 0.5 with LEV as argmax, so on the
dual-scored pool the text hard vote is 1.0 for every row by construction. The
informative text hard vote is the one over the full text scan
(`--text-scan`), which sees each speaker's non-LEV rows too.

Usage:
    python scripts/speaker_dialect_ranking.py \\
        --audio-scan dialect_id_scans/dialect_scan_badrex_*/row_probabilities.jsonl \\
        --text-scan  dialect_id_scans/text_dialect_scan_*/row_probabilities.jsonl \\
        --block-hours 8 8 --block-names train val \\
        --out-dir outputs/speaker_ranking
"""
from __future__ import annotations

import argparse
import json
import math
import sys
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable

from r2_speaker_group_disjointness import extract_group_key

TEXT_LEV = "LEV"
AUDIO_LEV = "Levantine"


# ------------------------------------------------------------------ loading


def iter_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                yield json.loads(line)


def text_prob(record: dict[str, Any], prefix: str = "") -> float:
    """p(LEV) from a text scan record, or from the text half of an audio record."""
    scores = record.get(f"{prefix}label_scores") or {}
    if TEXT_LEV in scores:
        return float(scores[TEXT_LEV])
    value = record.get(f"{prefix}target_label_score")
    return float(value) if value is not None else 0.0


def audio_prob(record: dict[str, Any]) -> float:
    scores = record.get("label_scores") or {}
    if AUDIO_LEV in scores:
        return float(scores[AUDIO_LEV])
    value = record.get("target_label_score")
    return float(value) if value is not None else 0.0


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    """Lower end of the Wilson score interval for a vote fraction.

    The raw mean/vote fraction has no notion of evidence: a masc_c video with a
    single dual-scored utterance scores 1.0 on one lucky row and outranks a
    speaker with 500 consistent ones. The Wilson bound answers "how high is
    this speaker's true lev rate, pessimistically" and so grows with n.
    """
    if n <= 0:
        return 0.0
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def new_pool_acc() -> dict[str, Any]:
    return {
        "n": 0, "hours": 0.0, "hours_audio_lev": 0.0,
        "text_sum": 0.0, "text_votes": 0,
        "audio_sum": 0.0, "audio_votes": 0, "audio_sq": 0.0,
        "both_agree": 0, "hours_both_agree": 0.0,
        "skipped_short": 0,
        "audio_label_counts": defaultdict(int),
    }


def load_audio_scans(
    paths: Iterable[Path], align_threshold: float = 0.80
) -> tuple[dict[tuple[str, str], dict], dict]:
    """Dual-scored rows (step 1), accumulated per speaker (step 2).

    Deduplicates on (source, source_file, row_idx): the four candidate-band runs
    are meant to be complementary, and a row counted twice would silently
    double-weight that speaker.

    `align_threshold` defines a row where the two models *agree on that sample*:
    both p(LEV) and p(Levantine) clear it. Counting those per speaker is what
    separates "the averages happen to be high" from "the two models keep landing
    on Levantine together, utterance after utterance".
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
                # `skipped_short`: clip too short for the audio model, so this
                # row has no audio classification and is not part of the pool.
                stats["rows_skipped_status"] += 1
                acc["skipped_short"] += 1
                continue

            stats["rows_ok"] += 1
            file_stats["ok"] += 1

            audio_top = record.get("top_label")
            duration = float(record.get("duration_sec") or 0.0)
            t_prob = text_prob(record, prefix="text_")
            a_prob = audio_prob(record)
            acc["n"] += 1
            acc["hours"] += duration / 3600.0
            acc["text_sum"] += t_prob
            acc["audio_sum"] += a_prob
            acc["audio_sq"] += a_prob * a_prob
            acc["audio_label_counts"][audio_top] += 1
            if t_prob >= align_threshold and a_prob >= align_threshold:
                acc["both_agree"] += 1
                acc["hours_both_agree"] += duration / 3600.0
            if record.get("text_top_label") == TEXT_LEV:
                acc["text_votes"] += 1
                stats["text_argmax_lev_rows"] += 1
            if audio_top == AUDIO_LEV:
                acc["audio_votes"] += 1
                acc["hours_audio_lev"] += duration / 3600.0

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


# -------------------------------------------------------------- aggregation


def build_speakers(
    pool: dict[tuple[str, str], dict],
    full_text: dict[tuple[str, str], dict],
    threshold: float,
    min_rows: int,
    min_hours: float,
) -> list[dict[str, Any]]:
    speakers: list[dict[str, Any]] = []
    for (source, key), acc in pool.items():
        n = acc["n"]
        if n < min_rows or acc["hours"] < min_hours:
            continue

        text_mean = acc["text_sum"] / n
        audio_mean = acc["audio_sum"] / n
        text_frac = acc["text_votes"] / n
        audio_frac = acc["audio_votes"] / n

        align_frac = acc["both_agree"] / n
        align_lb = wilson_lower_bound(acc["both_agree"], n)
        variance = max(0.0, acc["audio_sq"] / n - audio_mean * audio_mean)
        mean_of_means = (text_mean + audio_mean) / 2.0

        row = {
            "source": source,
            "speaker_key": key,
            "rows_dual_scored": n,
            "rows_skipped_short": acc["skipped_short"],
            "hours": acc["hours"],
            "hours_audio_lev": acc["hours_audio_lev"],
            "hours_both_agree": acc["hours_both_agree"],
            # step 3 -- text, over the dual-scored pool
            "text_mean_prob": text_mean,
            "text_soft_lev": text_mean >= threshold,
            "text_vote_frac": text_frac,
            "text_hard_lev": text_frac >= 0.5,
            # step 4 -- audio
            "audio_mean_prob": audio_mean,
            "audio_soft_lev": audio_mean >= threshold,
            "audio_vote_frac": audio_frac,
            "audio_hard_lev": audio_frac >= 0.5,
            "audio_label_counts": dict(acc["audio_label_counts"]),
            # steps 5 / 7 -- the primary score
            "combined_prob": text_mean * audio_mean,
            "combined_vote_frac": text_frac * audio_frac,
            # evidence-aware alternative: same product, but on vote fractions
            # discounted by group size (see wilson_lower_bound)
            "audio_vote_lb": wilson_lower_bound(acc["audio_votes"], n),

            # --- per-sample agreement between the two models -----------------
            # rows_both_agree: utterances where BOTH models clear the threshold.
            # align_lb discounts that rate by how many rows back it, so a
            # speaker is only "sure" when the agreement repeats across samples.
            "rows_both_agree": acc["both_agree"],
            "align_frac": align_frac,
            "align_lb": align_lb,
            "audio_prob_sd": math.sqrt(variance),

            # --- selectable composite scores ---------------------------------
            "score_sum": mean_of_means,
            "score_product": text_mean * audio_mean,
            "score_sum_align": mean_of_means * align_lb,
            "score_product_align": text_mean * audio_mean * align_lb,
        }

        ft = full_text.get((source, key))
        if ft and ft["n"]:
            full_mean = ft["sum"] / ft["n"]
            full_frac = ft["votes"] / ft["n"]
            text_lb = wilson_lower_bound(ft["votes"], ft["n"])
            row.update({
                "rows_text_total": ft["n"],
                "audio_coverage": n / ft["n"],
                "text_full_mean_prob": full_mean,
                "text_full_soft_lev": full_mean >= threshold,
                "text_full_vote_frac": full_frac,
                "text_full_hard_lev": full_frac >= 0.5,
                "text_full_vote_lb": text_lb,
                "combined_prob_full_text": full_mean * audio_mean,
                "combined_vote_lb": text_lb * row["audio_vote_lb"],
            })

        speakers.append(row)
    return speakers


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
    """Rank correlation over the speakers that carry both fields.

    Not every speaker has the full-text fields -- a text scan covers one source,
    so running with only the masc_c text scan leaves qasr speakers without them.
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
    """Soft-vs-hard alignment (the last clause of steps 3 and 4)."""
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


def rank(speakers: list[dict[str, Any]], field: str, tiebreak: str) -> list[dict[str, Any]]:
    have = [s for s in speakers if field in s]
    return sorted(have, key=lambda s: (-s[field], -s.get(tiebreak, 0.0), s["source"], str(s["speaker_key"])))


def carve_blocks(
    ranked: list[dict[str, Any]],
    block_hours: list[float],
    block_names: list[str],
    hours_field: str,
    score_field: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Walk the ranking top-down, filling each block to its hour target.

    Blocks are consecutive slices of one ranking, so they are speaker-disjoint
    by construction: a speaker lands in exactly one block or in the leftover
    pool.
    """
    blocks = [
        {"name": name, "target_hours": target, "speakers": 0, "hours": 0.0,
         "rows": 0, "min_score": None, "max_score": None}
        for name, target in zip(block_names, block_hours)
    ]
    assignments: list[dict[str, Any]] = []

    block_idx = 0
    for s in ranked:
        assigned = None
        while block_idx < len(blocks):
            block = blocks[block_idx]
            if block["hours"] >= block["target_hours"]:
                block_idx += 1
                continue
            block["speakers"] += 1
            block["hours"] += s[hours_field]
            block["rows"] += s["rows_dual_scored"]
            if block["max_score"] is None:
                block["max_score"] = s[score_field]
            block["min_score"] = s[score_field]
            assigned = block["name"]
            break
        assignments.append({**s, "block": assigned})

    for block in blocks:
        block["filled"] = block["hours"] >= block["target_hours"]
        block["hours"] = round(block["hours"], 3)
    return blocks, assignments


def cumulative_curve(ranked: list[dict[str, Any]], marks: list[float], hours_field: str,
                     score_field: str) -> list[dict[str, Any]]:
    """Speakers/rows needed to reach each cumulative-hour mark."""
    out: list[dict[str, Any]] = []
    remaining = sorted(marks)
    cum_hours = 0.0
    cum_rows = 0
    for i, s in enumerate(ranked, start=1):
        cum_hours += s[hours_field]
        cum_rows += s["rows_dual_scored"]
        while remaining and cum_hours >= remaining[0]:
            out.append({
                "target_hours": remaining.pop(0),
                "speakers_needed": i,
                "hours_reached": round(cum_hours, 3),
                "rows": cum_rows,
                "score_at_cut": s[score_field],
            })
    for target in remaining:
        out.append({"target_hours": target, "speakers_needed": None, "reason": "pool exhausted"})
    return out


SELECTION_COLUMNS = (
    "block", "source", "speaker_key", "rows_dual_scored", "rows_both_agree", "rows_text_total",
    "hours", "hours_audio_lev", "hours_both_agree", "text_mean_prob", "text_full_vote_frac",
    "text_full_vote_lb", "audio_mean_prob", "audio_vote_frac", "audio_vote_lb", "audio_prob_sd",
    "align_frac", "align_lb", "score_sum", "score_product", "score_sum_align",
    "score_product_align", "combined_vote_lb",
)


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
    needs to see typical rows, including the bad ones, not a curated best-of.
    """
    import random

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
        taken_per_speaker: dict[str, int] = defaultdict(int)
        picked = []
        for row in rows:
            if len(picked) >= per_block:
                break
            key = f"{row['source']}:{row['speaker_key']}"
            if taken_per_speaker[key] >= max_per_speaker:
                continue
            taken_per_speaker[key] += 1
            picked.append(row)
        out.extend(picked)
        print(f"samples[{block}]: {len(picked)} of {len(rows)} rows, "
              f"{len(taken_per_speaker)} speakers", file=sys.stderr)
    return out


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
    parser.add_argument("--min-rows-per-speaker", type=int, default=1)
    parser.add_argument("--min-hours-per-speaker", type=float, default=0.0)
    parser.add_argument("--align-threshold", type=float, default=0.80,
                        help="a row counts as agreed when BOTH models clear this")
    parser.add_argument("--rank-by", default="combined_prob",
                        choices=["combined_prob", "combined_vote_lb", "combined_prob_full_text",
                                 "score_sum", "score_product",
                                 "score_sum_align", "score_product_align"],
                        help="score the blocks are carved on (step 7)")
    parser.add_argument("--hours-field", default="hours",
                        choices=["hours", "hours_audio_lev", "hours_both_agree"],
                        help="hours counted per speaker when filling blocks")
    parser.add_argument("--block-hours", nargs="+", type=float, default=[8.0, 8.0])
    parser.add_argument("--block-names", nargs="+", default=["train", "val"])
    parser.add_argument("--curve-marks", nargs="*", type=float,
                        default=[1, 2, 4, 8, 16, 24, 32, 50, 100, 200])
    parser.add_argument("--emit-samples", type=int, default=0,
                        help="rows per block to dump for inspection (0 = off)")
    parser.add_argument("--out-dir", type=Path, default=Path("outputs/speaker_ranking"))
    args = parser.parse_args()

    if len(args.block_hours) != len(args.block_names):
        parser.error("--block-hours and --block-names must have the same length")

    pool, audio_stats = load_audio_scans(args.audio_scan, args.align_threshold)
    full_text, text_stats = load_text_scans(args.text_scan) if args.text_scan else ({}, {})

    speakers = build_speakers(
        pool, full_text, args.threshold, args.min_rows_per_speaker, args.min_hours_per_speaker)
    if not speakers:
        raise SystemExit("no speakers survived the --min-rows/--min-hours filters")

    has_full_text = any("text_full_vote_frac" in s for s in speakers)
    text_rank_field = "text_full_vote_frac" if has_full_text else "text_vote_frac"
    text_rank_tiebreak = "text_full_mean_prob" if has_full_text else "text_mean_prob"

    by_text = rank(speakers, text_rank_field, text_rank_tiebreak)
    by_audio = rank(speakers, "audio_vote_frac", "audio_mean_prob")
    by_combined = rank(speakers, args.rank_by, "combined_vote_frac")
    if not by_combined:
        raise SystemExit(f"--rank-by {args.rank_by} is not available (need --text-scan for the full-text scores)")

    blocks, assignments = carve_blocks(
        by_combined, args.block_hours, args.block_names, args.hours_field, args.rank_by)
    curve = cumulative_curve(by_combined, args.curve_marks, args.hours_field, args.rank_by)

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

    summary = {
        "inputs": {
            "audio_scans": [str(p) for p in args.audio_scan],
            "text_scans": [str(p) for p in args.text_scan],
        },
        "row_stats": {"audio": audio_stats, "text": text_stats},
        "settings": {
            "soft_threshold": args.threshold,
            "hard_vote": "per-row argmax label (top_label)",
            "align_threshold": args.align_threshold,
            "hours_field": args.hours_field,
            "rank_by": args.rank_by,
            "min_rows_per_speaker": args.min_rows_per_speaker,
            "min_hours_per_speaker": args.min_hours_per_speaker,
        },
        "pool": {
            "speakers": len(speakers),
            "rows_dual_scored": sum(s["rows_dual_scored"] for s in speakers),
            "total_hours": round(sum(s["hours"] for s in speakers), 2),
            "total_hours_audio_lev": round(sum(s["hours_audio_lev"] for s in speakers), 2),
            "per_source": {
                k: {"speakers": v["speakers"], "hours": round(v["hours"], 2), "rows": v["rows"]}
                for k, v in sorted(per_source.items())
            },
        },
        "soft_vs_hard_alignment": alignment,
        "ranking_agreement": {
            "text_vs_audio_hard": paired_spearman(speakers, text_rank_field, "audio_vote_frac"),
            "combined_vs_text_hard": paired_spearman(speakers, "combined_prob", text_rank_field),
            "combined_vs_audio_hard": paired_spearman(speakers, "combined_prob", "audio_vote_frac"),
            "combined_pool_vs_full_text": paired_spearman(speakers, "combined_prob", "combined_prob_full_text"),
        },
        "blocks": blocks,
        "cumulative_hours_curve": curve,
        "top_15_by_rank_field": [slim(s) for s in by_combined[:15]],
        "rows_per_speaker_in_blocks": {
            block["name"]: sorted(
                s["rows_dual_scored"] for s in assignments if s["block"] == block["name"]
            )[:1] + [
                round(sum(s["rows_dual_scored"] for s in assignments if s["block"] == block["name"])
                      / max(1, block["speakers"]), 1)
            ]
            for block in blocks
        },
    }

    args.out_dir.mkdir(parents=True, exist_ok=True)
    (args.out_dir / "summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_jsonl(args.out_dir / "speakers_by_combined.jsonl", assignments)
    write_selection_csv(args.out_dir / "selection.csv", assignments)
    if args.emit_samples:
        samples = emit_samples(
            args.audio_scan, assignments, args.emit_samples, args.align_threshold)
        write_jsonl(args.out_dir / "samples.jsonl", samples)
        summary["samples"] = {
            "per_block_requested": args.emit_samples,
            "written": len(samples),
            "path": str(args.out_dir / "samples.jsonl"),
        }
        (args.out_dir / "summary.json").write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_jsonl(args.out_dir / "speakers_by_text_vote.jsonl", by_text)
    write_jsonl(args.out_dir / "speakers_by_audio_vote.jsonl", by_audio)

    print(json.dumps(summary, indent=2, ensure_ascii=False))
    print(f"wrote {len(speakers)} speaker row(s) to {args.out_dir}", file=sys.stderr)


if __name__ == "__main__":
    main()
