"""Decode the HF-style {'bytes', 'path'} audio cells stored in data/clean/*.parquet."""
from __future__ import annotations

import io

import librosa
import numpy as np
import soundfile as sf

TARGET_SR = 16000


def decode_audio_cell(audio_cell, sampling_rate: int | None = None) -> np.ndarray:
    """Return mono float32 PCM at TARGET_SR from a parquet audio cell.

    Three shapes occur across the eval sets: an HF-style ``{bytes, path}``
    struct holding WAV (Casablanca, Layla) or FLAC (omni), and bare raw PCM16
    bytes with the rate in a sibling column (the custom MASC/QASR test block,
    which stores QASR's raw int16 exactly as QASR ships it). Raw PCM has no
    header to sniff, so it is only treated as such when `sampling_rate` is
    passed -- guessing would silently mangle a headered file.
    """
    raw = audio_cell["bytes"] if isinstance(audio_cell, dict) else bytes(audio_cell)

    if sampling_rate is not None and raw[:4] not in (b"RIFF", b"fLaC", b"OggS"):
        data = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        sr = int(sampling_rate)
    else:
        data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)

    if data.ndim > 1:
        data = data.mean(axis=1)
    if sr != TARGET_SR:
        data = librosa.resample(data, orig_sr=sr, target_sr=TARGET_SR)
    return np.ascontiguousarray(data, dtype=np.float32)
