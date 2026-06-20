"""Model-agnostic ASR metric computation."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any


def load_prediction_records(path: str | Path) -> list[dict[str, Any]]:
    """Load JSONL prediction records from disk."""

    prediction_path = Path(path)
    with prediction_path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def evaluate_prediction_records(records: list[dict[str, Any]]) -> dict[str, Any]:
    """Compute CER, WER, and optional RTF from prediction records."""

    references = [_normalize_text(str(row.get("reference", ""))) for row in records]
    predictions = [_normalize_text(str(row.get("prediction", ""))) for row in records]
    cer = _error_rate("".join(references), "".join(predictions), unit="char")
    wer = _corpus_wer(references, predictions)
    total_audio_seconds = sum(float(row.get("audio_seconds") or 0.0) for row in records)
    total_inference_seconds = sum(float(row.get("inference_seconds") or 0.0) for row in records)
    rtf = total_inference_seconds / total_audio_seconds if total_audio_seconds > 0 else None
    return {
        "num_samples": len(records),
        "cer": cer,
        "wer": wer,
        "rtf": rtf,
        "total_audio_seconds": total_audio_seconds,
        "total_inference_seconds": total_inference_seconds,
    }


def _normalize_text(text: str) -> str:
    """Normalize whitespace before metric calculation."""

    return re.sub(r"\s+", " ", text.strip())


def _corpus_wer(references: list[str], predictions: list[str]) -> float:
    """Compute corpus WER using whitespace-token edit distance."""

    ref_words: list[str] = []
    pred_words: list[str] = []
    for reference, prediction in zip(references, predictions):
        ref_words.extend(reference.split())
        pred_words.extend(prediction.split())
    return _error_rate(ref_words, pred_words, unit="word")


def _error_rate(reference: str | list[str], prediction: str | list[str], unit: str) -> float:
    """Compute edit-distance error rate over characters or tokens."""

    if unit == "char":
        ref_units = list(reference) if isinstance(reference, str) else reference
        pred_units = list(prediction) if isinstance(prediction, str) else prediction
    else:
        ref_units = reference if isinstance(reference, list) else reference.split()
        pred_units = prediction if isinstance(prediction, list) else prediction.split()
    if not ref_units:
        return 0.0 if not pred_units else 1.0
    return _levenshtein(ref_units, pred_units) / len(ref_units)


def _levenshtein(reference: list[str], prediction: list[str]) -> int:
    """Return Levenshtein distance for token sequences."""

    previous = list(range(len(prediction) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, pred_token in enumerate(prediction, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (ref_token != pred_token)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]
