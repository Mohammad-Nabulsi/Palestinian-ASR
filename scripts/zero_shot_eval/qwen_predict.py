"""QwenPredict API: batched zero-shot Qwen/Qwen3-ASR-0.6B-hf transcription,
mirroring whisper_predict.py / cohere_predict.py's `predict(audios) -> list[str]`
contract so it plugs into the same run_eval resumable loop.

- Inference goes through `processor.apply_transcription_request(audio=..., language=...)`
  -> `model.generate` -> `processor.decode(..., return_format="transcription_only")`,
  exactly as Qwen3ASRAdapter.generate() does in asr_qwen3_finetune.ipynb. That call
  builds the chat-template prompt (system turn with the language hint + user turn
  holding the audio) internally, so no manual chat-template / prompt construction is
  needed for inference -- the notebook's `apply_chat_template(..., output_labels=True)`
  path is training-only.
- Verified empirically: going through the processor's `__call__` (as
  apply_transcription_request does) already pads each batch dynamically to that
  batch's own longest clip, not a fixed 30s -- including batches that mix a clip
  under Qwen3-ASR's native 30s chunk with one well over it. So length-grouped
  batching upstream (see run_eval_qwen.py) is what keeps padding waste low; no
  manual pre-padding is required here.
- Unlike Whisper (pipeline chunking) or Cohere (processor-internal chunk+reassemble),
  Qwen3-ASR has no >30s auto-chunking: a whole clip is one forward pass. max_new_tokens
  therefore scales with the batch's longest clip so long-clip datasets (omni, layla)
  don't get truncated transcripts.
- Batch size backs off (halves, never grows back) on CUDA OOM instead of crashing the
  run, same recursive-split policy as the other two predictors.
"""
from __future__ import annotations

import gc
import logging
import time
from dataclasses import dataclass, field
from typing import Any, Sequence

import numpy as np
import torch

logger = logging.getLogger(__name__)

MODEL_ID = "Qwen/Qwen3-ASR-0.6B-hf"


@dataclass
class QwenPredictConfig:
    model_id: str = MODEL_ID
    device: str = field(default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = field(default_factory=lambda: torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    language: str = "ar"
    tokens_per_sec: float = 8.0        # generous Arabic speech-rate bound for max_new_tokens sizing
    max_new_tokens_floor: int = 64
    max_new_tokens_cap: int = 1024
    min_batch_size: int = 1


class QwenPredictor:
    """Zero-shot Qwen3-ASR-0.6B predictor. `predict()` takes decoded 16kHz mono
    float32 arrays and returns transcripts in the same order, scaling max_new_tokens
    to the batch's longest clip and shrinking the batch on OOM as needed so a run
    never crashes."""

    def __init__(self, config: QwenPredictConfig | None = None):
        self.config = config or QwenPredictConfig()
        self._load()

    def _load(self) -> None:
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration

        c = self.config
        logger.info("Loading %s on %s (dtype=%s)", c.model_id, c.device, c.dtype)
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(c.model_id)
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
            c.model_id, dtype=c.dtype
        ).to(c.device)
        self.model.eval()
        self.model.config.use_cache = True
        logger.info("Model loaded in %.1fs", time.time() - t0)

    def predict(self, audios: Sequence[Any]) -> list[str]:
        """audios: sequence of mono float32 np.ndarray @ 16kHz."""
        if not audios:
            return []
        return self._predict_with_backoff(list(audios))

    @torch.no_grad()
    def _generate(self, audios: list[np.ndarray]) -> list[str]:
        c = self.config
        max_dur_s = max(len(a) for a in audios) / 16000.0
        max_new_tokens = int(min(
            c.max_new_tokens_cap,
            max(c.max_new_tokens_floor, round(max_dur_s * c.tokens_per_sec) + 32),
        ))
        req = self.processor.apply_transcription_request(
            audio=audios, language=[c.language] * len(audios)
        )
        req = req.to(c.device, c.dtype)
        out_ids = self.model.generate(**req, max_new_tokens=max_new_tokens)
        gen = out_ids[:, req["input_ids"].shape[1]:]
        texts = self.processor.decode(gen, return_format="transcription_only")
        return [str(t).strip() for t in texts]

    def _predict_with_backoff(self, audios: list[np.ndarray]) -> list[str]:
        try:
            return self._generate(audios)
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if len(audios) <= self.config.min_batch_size:
                logger.error("CUDA OOM on a single sample even at batch_size=1; skipping it")
                return [""] * len(audios)
            logger.warning(
                "CUDA OOM at batch_size=%d; splitting in half and retrying", len(audios)
            )
            mid = len(audios) // 2
            return self._predict_with_backoff(audios[:mid]) + self._predict_with_backoff(audios[mid:])
