"""Decode the HF-style {'bytes', 'path'} audio cells stored in data/clean/*.parquet."""
from __future__ import annotations

import io

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000


def decode_audio_cell(audio_cell: dict) -> np.ndarray:
    """Return mono float32 PCM at TARGET_SR from a parquet 'audio' cell."""
    data, sr = sf.read(io.BytesIO(audio_cell["bytes"]), dtype="float32", always_2d=False)
    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
    return np.ascontiguousarray(data, dtype=np.float32)
