"""ConformerPredict API: batched zero-shot FastConformer-Hybrid (CTC) transcription.

- The NeMo `ASRModel.transcribe()` API takes file paths, not raw arrays, so each batch
  is written to short-lived WAV files under a temp dir and cleaned up immediately after
  (mirrors the collate-time pattern already used for training in
  ConformerCTCAdapter.generate() in asr_conformer_ctc_finetune.ipynb).
- Batch size backs off (halves, never grows back) on CUDA OOM instead of crashing the
  run; a fresh `torch.cuda.empty_cache()` is issued on every backoff.
"""
from __future__ import annotations

import gc
import logging
import os
import tempfile
import time
from dataclasses import dataclass, field
from typing import Sequence

import numpy as np
import soundfile as sf
import torch

logger = logging.getLogger(__name__)

SAMPLE_RATE = 16000


@dataclass
class ConformerPredictConfig:
    model_id: str = "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0"
    device: str = field(default_factory=lambda: "cuda:0" if torch.cuda.is_available() else "cpu")
    batch_size: int = 8
    min_batch_size: int = 1


class ConformerPredictor:
    """Zero-shot FastConformer-CTC predictor. `predict()` takes decoded 16kHz mono
    float32 arrays and returns transcripts in the same order, shrinking the batch on
    OOM as needed so a run never crashes."""

    def __init__(self, config: ConformerPredictConfig | None = None):
        self.config = config or ConformerPredictConfig()
        self._current_batch_size = self.config.batch_size
        self._load()

    def _load(self) -> None:
        import nemo.collections.asr as nemo_asr

        c = self.config
        logger.info("Loading %s on %s", c.model_id, c.device)
        t0 = time.time()
        self.model = nemo_asr.models.ASRModel.from_pretrained(c.model_id, map_location=c.device)
        self.model.change_decoding_strategy(decoder_type="ctc")  # drive the CTC head (matches training notebook)
        self.model.eval()
        logger.info("Model loaded in %.1fs", time.time() - t0)

    def predict(self, audios: Sequence[np.ndarray]) -> list[str]:
        """audios: sequence of mono float32 np.ndarray @ 16kHz."""
        if len(audios) == 0:
            return []
        return self._transcribe_with_backoff(list(audios))

    def _transcribe_with_backoff(self, audios: list[np.ndarray]) -> list[str]:
        if not audios:
            return []
        bs = max(self.config.min_batch_size, min(self._current_batch_size, len(audios)))
        tmp = tempfile.mkdtemp(prefix="conformer_zeroshot_")
        paths = []
        try:
            for i, a in enumerate(audios):
                p = os.path.join(tmp, f"{i}.wav")
                sf.write(p, np.asarray(a, dtype=np.float32), SAMPLE_RATE)
                paths.append(p)
            with torch.no_grad():
                hyps = self.model.transcribe(paths, batch_size=bs, verbose=False)
            return [(h.text if hasattr(h, "text") else str(h)).strip() for h in hyps]
        except torch.cuda.OutOfMemoryError:
            torch.cuda.empty_cache()
            gc.collect()
            if len(audios) == 1:
                logger.error("CUDA OOM on a single sample even at batch_size=1; skipping it")
                return [""]
            self._current_batch_size = max(self.config.min_batch_size, bs // 2)
            logger.warning(
                "CUDA OOM at batch_size=%d (n=%d); backing off to batch_size=%d and retrying",
                bs, len(audios), self._current_batch_size,
            )
            mid = len(audios) // 2
            return self._transcribe_with_backoff(audios[:mid]) + self._transcribe_with_backoff(audios[mid:])
        finally:
            for p in paths:
                try:
                    os.unlink(p)
                except OSError:
                    pass
            try:
                os.rmdir(tmp)
            except OSError:
                pass
