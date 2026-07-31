"""OmniPredict API: batched zero-shot Omnilingual ASR (omniASR_LLM_300M) transcription.

- Unlike Whisper's built-in sliding-window chunking, the Omnilingual pipeline hard-caps
  input audio at 40s and raises ValueError above that. Audio longer than a safety
  threshold is split here into non-overlapping <=30s chunks, transcribed, and the
  pieces joined with spaces.
- Batch size backs off (halves, never grows back) on CUDA OOM instead of crashing the
  run; a fresh `torch.cuda.empty_cache()` is issued on every backoff.
"""
from __future__ import annotations

import gc
import logging
import os
import time
from dataclasses import dataclass, field
from typing import Optional, Sequence

import numpy as np
import torch

# Reuse the checkpoint already downloaded for a prior run instead of re-fetching it.
os.environ.setdefault("FAIRSEQ2_CACHE_DIR", "/workspace/asr_env/models")

from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline  # noqa: E402

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass
class OmniPredictConfig:
    model_id: str = "omniASR_LLM_300M"
    device: str = field(default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = field(default_factory=lambda: torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    batch_size: int = 8
    min_batch_size: int = 1
    chunk_length_s: float = 30.0  # sub-chunk length for audio over the safety threshold
    safe_max_audio_s: float = 38.0  # below the model's hard 40s cap, with margin
    language: Optional[str] = "apc_Arab"  # default lang conditioning token; callers may override per row


class OmniPredictor:
    """Zero-shot Omnilingual ASR predictor. `predict()` takes decoded 16kHz mono
    float32 arrays (+ optional per-row language codes) and returns transcripts in the
    same order, chunking long audio and shrinking the batch on OOM as needed so a run
    never crashes."""

    def __init__(self, config: OmniPredictConfig | None = None):
        self.config = config or OmniPredictConfig()
        self._current_batch_size = self.config.batch_size
        self._load()

    def _load(self) -> None:
        c = self.config
        logger.info("Loading %s on %s (dtype=%s)", c.model_id, c.device, c.dtype)
        t0 = time.time()
        self.pipe = ASRInferencePipeline(model_card=c.model_id, device=c.device, dtype=c.dtype)
        logger.info("Model loaded in %.1fs", time.time() - t0)

    def predict(self, audios: Sequence[np.ndarray], langs: Sequence[Optional[str]] | None = None) -> list[str]:
        """audios: sequence of mono float32 np.ndarray @ 16kHz."""
        n = len(audios)
        if n == 0:
            return []
        row_langs = list(langs) if langs is not None else [self.config.language] * n

        results: list[Optional[str]] = [None] * n
        short_idx, long_idx = [], []
        for i, a in enumerate(audios):
            is_long = len(a) / SAMPLE_RATE > self.config.safe_max_audio_s
            (long_idx if is_long else short_idx).append(i)

        if short_idx:
            items = [{"raw": audios[i], "lang": row_langs[i]} for i in short_idx]
            for i, hyp in zip(short_idx, self._transcribe_with_backoff(items)):
                results[i] = hyp

        if long_idx:
            # Flatten every long row's chunks into one pool so the whole batch's GPU
            # parallelism is used, instead of transcribing one row's chunks at a time.
            chunk_len = int(self.config.chunk_length_s * SAMPLE_RATE)
            chunk_items: list[dict] = []
            chunk_owner: list[int] = []
            for i in long_idx:
                audio = audios[i]
                chunks = [audio[s : s + chunk_len] for s in range(0, len(audio), chunk_len)]
                for c in chunks:
                    if len(c) == 0:
                        continue
                    chunk_items.append({"raw": c, "lang": row_langs[i]})
                    chunk_owner.append(i)

            chunk_hyps = self._transcribe_with_backoff(chunk_items)
            parts_by_row: dict[int, list[str]] = {}
            for owner, hyp in zip(chunk_owner, chunk_hyps):
                parts_by_row.setdefault(owner, []).append(hyp)
            for i in long_idx:
                results[i] = " ".join(p for p in parts_by_row.get(i, []) if p).strip()

        return results  # type: ignore[return-value]

    def _transcribe_with_backoff(self, items: list[dict]) -> list[str]:
        if not items:
            return []
        bs = max(self.config.min_batch_size, min(self._current_batch_size, len(items)))
        inputs = [{"waveform": it["raw"], "sample_rate": SAMPLE_RATE} for it in items]
        langs = [it["lang"] for it in items]
        try:
            out = self.pipe.transcribe(inputs, lang=langs, batch_size=bs)
            return [str(t).strip() for t in out]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if len(items) == 1:
                logger.error("CUDA OOM on a single sample even at batch_size=1; skipping it")
                return [""]
            self._current_batch_size = max(self.config.min_batch_size, bs // 2)
            logger.warning(
                "CUDA OOM at batch_size=%d (n=%d); backing off to batch_size=%d and retrying",
                bs, len(items), self._current_batch_size,
            )
            mid = len(items) // 2
            return self._transcribe_with_backoff(items[:mid]) + self._transcribe_with_backoff(items[mid:])
