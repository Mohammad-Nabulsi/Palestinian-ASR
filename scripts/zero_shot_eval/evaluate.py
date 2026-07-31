"""Evaluate API: WER/CER of hypotheses vs ground truth, using this repo's
standard Arabic normalization (pipeline.textnorm) so scoring matches how
manual_normalized_transcript was produced for every dataset in data/clean."""
from __future__ import annotations

import sys
from pathlib import Path

import jiwer

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from pipeline.textnorm import normalize_arabic_transcript  # noqa: E402


def evaluate(refs: list[str], hyps: list[str]) -> dict:
    """Corpus-level WER/CER plus per-utterance breakdown.

    Rows whose normalized reference is empty are excluded from scoring
    (jiwer is undefined on an empty reference) but counted in n_dropped.
    """
    if len(refs) != len(hyps):
        raise ValueError(f"refs/hyps length mismatch: {len(refs)} vs {len(hyps)}")

    norm_refs = [normalize_arabic_transcript(r) for r in refs]
    norm_hyps = [normalize_arabic_transcript(h) for h in hyps]

    pairs = [(r, h) for r, h in zip(norm_refs, norm_hyps) if r.strip()]
    dropped = len(norm_refs) - len(pairs)
    kept_refs = [r for r, _ in pairs]
    kept_hyps = [h for _, h in pairs]

    if not pairs:
        return {
            "n_total": len(refs),
            "n_scored": 0,
            "n_dropped_empty_ref": dropped,
            "wer": None,
            "cer": None,
            "per_utterance": [],
        }

    wer = jiwer.wer(kept_refs, kept_hyps)
    cer = jiwer.cer(kept_refs, kept_hyps)

    per_utt = [
        {"reference": r, "hypothesis": h, "wer": jiwer.wer([r], [h]), "cer": jiwer.cer([r], [h])}
        for r, h in pairs
    ]

    return {
        "n_total": len(refs),
        "n_scored": len(pairs),
        "n_dropped_empty_ref": dropped,
        "wer": wer,
        "cer": cer,
        "per_utterance": per_utt,
    }
