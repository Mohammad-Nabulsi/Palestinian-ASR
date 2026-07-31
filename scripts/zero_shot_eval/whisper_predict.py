"""WhisperPredict API: batched zero-shot Whisper-large-v3 transcription.

- Audio > 30s is split automatically by the HF ASR pipeline's built-in
  sliding-window chunking (chunk_length_s=30, stride_length_s=5) -- the
  well-tested mechanism transformers ships for exactly this, rather than a
  hand-rolled splitter.
- Batch size backs off (halves, never grows back) on CUDA OOM instead of
  crashing the run; a fresh `torch.cuda.empty_cache()` is issued on every
  backoff.
"""
from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import torch
from transformers import pipeline as hf_pipeline

logger = logging.getLogger(__name__)


@dataclass
class WhisperPredictConfig:
    model_id: str = "openai/whisper-large-v3"
    device: str = field(default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = field(default_factory=lambda: torch.float16 if torch.cuda.is_available() else torch.float32)
    batch_size: int = 8
    min_batch_size: int = 1
    chunk_length_s: float = 30.0
    stride_length_s: float = 5.0
    language: str = "arabic"
    task: str = "transcribe"
    max_new_tokens: int = 256


class WhisperPredictor:
    """Zero-shot Whisper predictor. `predict()` takes decoded 16kHz mono
    float32 arrays and returns transcripts in the same order, splitting long
    audio and shrinking the batch on OOM as needed so a run never crashes."""

    def __init__(self, config: WhisperPredictConfig | None = None):
        self.config = config or WhisperPredictConfig()
        self._current_batch_size = self.config.batch_size
        self._load()

    def _load(self) -> None:
        c = self.config
        logger.info("Loading %s on %s (dtype=%s)", c.model_id, c.device, c.dtype)
        t0 = time.time()
        self.pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=c.model_id,
            dtype=c.dtype,
            device=c.device,
            chunk_length_s=c.chunk_length_s,
            stride_length_s=c.stride_length_s,
            ignore_warning=True,
            generate_kwargs={
                "language": c.language,
                "task": c.task,
                "max_new_tokens": c.max_new_tokens,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.2,
            },
        )
        logger.info("Model loaded in %.1fs", time.time() - t0)

    def predict(self, audios: Sequence[Any]) -> list[str]:
        """audios: sequence of mono float32 np.ndarray @ 16kHz."""
        inputs = [{"raw": a, "sampling_rate": 16000} for a in audios]
        return self._predict_with_backoff(inputs)

    def _predict_with_backoff(self, inputs: list[dict]) -> list[str]:
        if not inputs:
            return []
        bs = max(self.config.min_batch_size, min(self._current_batch_size, len(inputs)))
        try:
            out = self.pipe(inputs, batch_size=bs)
            return [o["text"].strip() for o in out]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if len(inputs) == 1:
                logger.error("CUDA OOM on a single sample even at batch_size=1; skipping it")
                return [""]
            self._current_batch_size = max(self.config.min_batch_size, bs // 2)
            logger.warning(
                "CUDA OOM at batch_size=%d (n=%d); backing off to batch_size=%d and retrying",
                bs, len(inputs), self._current_batch_size,
            )
            mid = len(inputs) // 2
            return self._predict_with_backoff(inputs[:mid]) + self._predict_with_backoff(inputs[mid:])
