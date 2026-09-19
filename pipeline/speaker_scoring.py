"""Speaker-level dialect scoring and block carving.

One implementation, two consumers: the `speaker_select` pipeline stage and the
standalone `scripts/speaker_dialect_ranking.py` analysis CLI. Both read the same
thing -- the *audio* dialect scan's ``row_probabilities.jsonl``, which carries
both modalities per row -- so the scoring lives here rather than in either.

The unit is a speaker: a QASR ``recording_id`` or a MASC-C ``video_id``, parsed
out of the row ``uid``. Per speaker we track

  * mean p(LEV) from the text model and mean p(Levantine) from the audio model
  * hard votes: how often each model's *argmax* is the Levantine class
  * agreed rows: utterances where BOTH models clear ``align_threshold`` -- the
    two models landing on Levantine for the same clip, not merely averaging high
  * hours, split by whether the audio argmax is Levantine and whether both agree

and from those a set of selectable scores. The default,
``score_product_align``, is

    mean p(LEV) * mean p(Levantine) * WilsonLB(agreed rows / rows)

The Wilson term is what makes group size count: a speaker with one lucky
utterance scores near zero on it however high that utterance's probabilities
are, while agreement that repeats over dozens of rows survives the discount.
Ranking on it and cutting consecutive slices gives blocks that are
speaker-disjoint by construction -- ``verify_disjoint`` re-checks that rather
than trusting it.
"""
from __future__ import annotations

import math
import re
from collections import defaultdict
from typing import Any, Iterable

TEXT_LEV = "LEV"
AUDIO_LEV = "Levantine"

# The two uid shapes that coexist across the scan runs, per source.
QASR_LONG_RE = re.compile(r"uid=qasr:([0-9A-Fa-f-]+):")
QASR_SHORT_RE = re.compile(r"^qasr:([0-9A-Fa-f-]+):")
MASC_LONG_RE = re.compile(r"video_id=([^:]+)$")
MASC_SHORT_RE = re.compile(r"^masc_c:([^:]+):")

SCORE_FIELDS = (
    "score_sum", "score_product", "score_sum_align", "score_product_align",
    "combined_prob", "combined_vote_lb", "combined_prob_full_text",
)


def extract_group_key(source: str, uid: str) -> str | None:
    """recording_id for qasr, video_id for masc_c.

    These are the closest available speaker proxies: qasr exposes only the
    recording in its uid (a broadcast anchor recurs across recordings), and
    masc_c has no speaker id at all.
    """
    if source == "qasr":
        match = QASR_LONG_RE.search(uid) or QASR_SHORT_RE.search(uid)
        return match.group(1) if match else None
    if source == "masc_c":
        match = MASC_LONG_RE.search(uid) or MASC_SHORT_RE.search(uid)
        return match.group(1) if match else None
    return None


def wilson_lower_bound(successes: int, n: int, z: float = 1.96) -> float:
    """Lower end of the Wilson score interval for a rate.

    A raw rate has no notion of evidence: one row at 1.0 ties a hundred rows at
    1.0. This answers "how high is the true rate, pessimistically", so it rises
    with n and keeps one-utterance speakers out of the top of a ranking.
    """
    if n <= 0:
        return 0.0
    phat = successes / n
    denom = 1.0 + z * z / n
    centre = phat + z * z / (2 * n)
    margin = z * math.sqrt((phat * (1 - phat) + z * z / (4 * n)) / n)
    return max(0.0, (centre - margin) / denom)


def text_prob(record: dict[str, Any], prefix: str = "") -> float:
    """p(LEV): from a text-scan record, or the text half of an audio record."""
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


def new_pool_acc() -> dict[str, Any]:
    return {
        "n": 0, "hours": 0.0, "hours_audio_lev": 0.0,
        "text_sum": 0.0, "text_votes": 0,
        "audio_sum": 0.0, "audio_votes": 0, "audio_sq": 0.0,
        "both_agree": 0, "hours_both_agree": 0.0,
        "skipped_short": 0,
        "audio_label_counts": defaultdict(int),
    }


def accumulate_audio_row(acc: dict[str, Any], record: dict[str, Any], align_threshold: float) -> None:
    """Fold one dual-scored row into its speaker's accumulator."""
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
    if record.get("text_top_label") == TEXT_LEV:
        acc["text_votes"] += 1
    if audio_top == AUDIO_LEV:
        acc["audio_votes"] += 1
        acc["hours_audio_lev"] += duration / 3600.0
    if t_prob >= align_threshold and a_prob >= align_threshold:
        acc["both_agree"] += 1
        acc["hours_both_agree"] += duration / 3600.0


def build_speakers(
    pool: dict[tuple[str, str], dict],
    full_text: dict[tuple[str, str], dict] | None = None,
    threshold: float = 0.80,
    min_rows: int = 1,
    min_hours: float = 0.0,
) -> list[dict[str, Any]]:
    """Per-speaker record: soft and hard scores per modality, plus composites."""
    full_text = full_text or {}
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
            # text, over the dual-scored pool
            "text_mean_prob": text_mean,
            "text_soft_lev": text_mean >= threshold,
            "text_vote_frac": text_frac,
            "text_hard_lev": text_frac >= 0.5,
            # audio
            "audio_mean_prob": audio_mean,
            "audio_soft_lev": audio_mean >= threshold,
            "audio_vote_frac": audio_frac,
            "audio_hard_lev": audio_frac >= 0.5,
            "audio_label_counts": dict(acc["audio_label_counts"]),
            "audio_vote_lb": wilson_lower_bound(acc["audio_votes"], n),
            # per-sample agreement between the two models
            "rows_both_agree": acc["both_agree"],
            "align_frac": align_frac,
            "align_lb": align_lb,
            "audio_prob_sd": math.sqrt(variance),
            # composite scores
            "combined_prob": text_mean * audio_mean,
            "combined_vote_frac": text_frac * audio_frac,
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


def rank(speakers: list[dict[str, Any]], field: str, tiebreak: str = "rows_dual_scored") -> list[dict[str, Any]]:
    have = [s for s in speakers if field in s]
    return sorted(have, key=lambda s: (-s[field], -s.get(tiebreak, 0.0), s["source"], str(s["speaker_key"])))


def carve_blocks(
    ranked: list[dict[str, Any]],
    block_hours: Iterable[float],
    block_names: Iterable[str],
    hours_field: str = "hours",
    score_field: str = "score_product_align",
    block_min_rows: Iterable[int] | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Fill each block from the top of the ranking, in order.

    A speaker lands in at most one block, so the result is speaker-disjoint. The
    first block listed gets the highest-scoring speakers, so name them in the
    order of how much confidence each needs -- val and test before train.

    ``block_min_rows`` sets a per-block evidence floor. Eval blocks need enough
    utterances per speaker to trust the score at all, while a train block is
    happy to take short speakers; with a floor on the first blocks only, a
    two-utterance speaker skips val and test and lands in train instead of
    displacing a well-evidenced speaker from the eval set. A speaker that clears
    no block's floor stays unassigned.
    """
    names = list(block_names)
    hours = list(block_hours)
    floors = list(block_min_rows) if block_min_rows is not None else [0] * len(names)
    if not (len(names) == len(hours) == len(floors)):
        raise ValueError("block names, hours and min_rows must be the same length")

    blocks = [
        {"name": name, "target_hours": target, "min_rows": floor, "speakers": 0,
         "hours": 0.0, "rows": 0, "min_score": None, "max_score": None}
        for name, target, floor in zip(names, hours, floors)
    ]
    assignments: list[dict[str, Any]] = []

    for speaker in ranked:
        assigned = None
        for block in blocks:
            if block["hours"] >= block["target_hours"]:
                continue
            if speaker["rows_dual_scored"] < block["min_rows"]:
                continue
            block["speakers"] += 1
            block["hours"] += speaker[hours_field]
            block["rows"] += speaker["rows_dual_scored"]
            if block["max_score"] is None:
                block["max_score"] = speaker[score_field]
            block["min_score"] = speaker[score_field]
            assigned = block["name"]
            break
        assignments.append({**speaker, "block": assigned})

    for block in blocks:
        block["filled"] = block["hours"] >= block["target_hours"]
        block["hours"] = round(block["hours"], 3)
    return blocks, assignments


def verify_disjoint(assignments: list[dict[str, Any]]) -> dict[str, Any]:
    """Independent check that no speaker reaches two blocks.

    Carving guarantees this, but the guarantee is only as good as the carve, and
    a split that silently leaks is the exact failure this whole exercise exists
    to remove -- so it gets checked rather than assumed. Also flags a speaker key
    appearing under two sources, which would mean the key is not unique.
    """
    blocks_by_speaker: dict[tuple[str, str], set[str]] = defaultdict(set)
    keys_by_bare: dict[str, set[str]] = defaultdict(set)

    for row in assignments:
        block = row.get("block")
        if block is None:
            continue
        blocks_by_speaker[(row["source"], row["speaker_key"])].add(block)
        keys_by_bare[str(row["speaker_key"])].add(row["source"])

    straddlers = {
        f"{source}:{key}": sorted(blocks)
        for (source, key), blocks in blocks_by_speaker.items()
        if len(blocks) > 1
    }
    cross_source = {k: sorted(v) for k, v in keys_by_bare.items() if len(v) > 1}

    per_block: dict[str, int] = defaultdict(int)
    for (_source, _key), blocks in blocks_by_speaker.items():
        for block in blocks:
            per_block[block] += 1

    names = sorted(per_block)
    pairs = {
        f"{a} vs {b}": len({s for s, bl in blocks_by_speaker.items() if a in bl}
                           & {s for s, bl in blocks_by_speaker.items() if b in bl})
        for i, a in enumerate(names) for b in names[i + 1:]
    }

    return {
        "assigned_speakers": len(blocks_by_speaker),
        "speakers_per_block": dict(per_block),
        "pairwise_shared_speakers": pairs,
        "speakers_in_more_than_one_block": straddlers,
        "speaker_keys_seen_under_two_sources": cross_source,
        "disjoint": not straddlers and not any(pairs.values()),
    }


def cumulative_curve(
    ranked: list[dict[str, Any]],
    marks: Iterable[float],
    hours_field: str = "hours",
    score_field: str = "score_product_align",
) -> list[dict[str, Any]]:
    """Speakers/rows needed to reach each cumulative-hour mark."""
    out: list[dict[str, Any]] = []
    remaining = sorted(marks)
    cum_hours = 0.0
    cum_rows = 0
    for i, speaker in enumerate(ranked, start=1):
        cum_hours += speaker[hours_field]
        cum_rows += speaker["rows_dual_scored"]
        while remaining and cum_hours >= remaining[0]:
            out.append({
                "target_hours": remaining.pop(0),
                "speakers_needed": i,
                "hours_reached": round(cum_hours, 3),
                "rows": cum_rows,
                "score_at_cut": speaker[score_field],
            })
    for target in remaining:
        out.append({"target_hours": target, "speakers_needed": None, "reason": "pool exhausted"})
    return out
