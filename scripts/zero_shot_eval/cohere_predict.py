"""CoherePredict API: batched zero-shot CohereLabs/cohere-transcribe-arabic-07-2026
transcription, mirroring whisper_predict.py's WhisperPredictor shape (same
`predict(audios) -> list[str]` contract) so it plugs into the same run_eval
resumable loop.

- Uses the model's real predict path (processor -> generate -> decode), as in
  the CohereAsrAdapter.generate() in asr_cohere_transcribe_finetune.ipynb.
- Audio > processor.feature_extractor.max_audio_clip_s (35s) is split into
  chunks internally by the processor and reassembled with
  processor._reassemble_chunk_texts, exactly like the notebook.
- Batch size backs off (halves, never grows back) on CUDA OOM instead of
  crashing the run, same recursive-split policy as WhisperPredictor.
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

MODEL_ID = "CohereLabs/cohere-transcribe-arabic-07-2026"


@dataclass
class CoherePredictConfig:
    model_id: str = MODEL_ID
    device: str = field(default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu")
    dtype: torch.dtype = field(default_factory=lambda: torch.bfloat16 if torch.cuda.is_available() else torch.float32)
    language: str = "ar"
    max_new_tokens: int = 256
    min_batch_size: int = 1


class CoherePredictor:
    """Zero-shot CohereAsr predictor. `predict()` takes decoded 16kHz mono
    float32 arrays and returns transcripts in the same order, splitting long
    audio (processor-internal chunking) and shrinking the batch on OOM as
    needed so a run never crashes."""

    def __init__(self, config: CoherePredictConfig | None = None):
        self.config = config or CoherePredictConfig()
        self._load()

    def _load(self) -> None:
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration

        c = self.config
        logger.info("Loading %s on %s (dtype=%s)", c.model_id, c.device, c.dtype)
        t0 = time.time()
        self.processor = AutoProcessor.from_pretrained(c.model_id)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
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
        enc = self.processor(audio=audios, language=c.language, sampling_rate=16000)
        chunk_index = enc.pop("audio_chunk_index", None)
        enc = enc.to(c.device)
        req = {
            k: (v.to(c.dtype) if torch.is_tensor(v) and v.is_floating_point() else v)
            for k, v in enc.items()
        }
        out_ids = self.model.generate(**req, max_new_tokens=c.max_new_tokens)
        gen = out_ids[:, req["decoder_input_ids"].shape[1]:]
        texts = self.processor.tokenizer.batch_decode(gen, skip_special_tokens=True)
        if chunk_index is not None and any(ci[1] is not None for ci in chunk_index):
            texts = self.processor._reassemble_chunk_texts(texts, chunk_index, " ")
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
