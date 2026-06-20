"""Inference helpers for baseline and tuned Whisper predictions."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any

from Runs.utils.data import audio_duration_seconds, resolve_manifest_records

from .config import WhisperLargeRunConfig
from .modeling import WhisperBundle, load_whisper_bundle


def generate_predictions(
    config: WhisperLargeRunConfig,
    manifest_path: str | Path,
    output_path: str | Path,
    bundle: WhisperBundle | None = None,
    adapter_path: str | Path | None = None,
) -> Path:
    """Generate JSONL predictions for a manifest and return the output path."""

    records = resolve_manifest_records(manifest_path, split="test")
    prediction_path = Path(output_path)
    prediction_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"[predictions] Output file: {prediction_path}")
    print(f"[predictions] Number of samples: {len(records)}")
    active_bundle = bundle or load_whisper_bundle(config, adapter_path=adapter_path)

    with prediction_path.open("w", encoding="utf-8") as handle:
        for record in _progress(records, desc="[predictions] Generating", unit="sample"):
            started = time.perf_counter()
            prediction = _predict_one(config, active_bundle, record.audio_filepath, record.text)
            inference_seconds = time.perf_counter() - started
            row = {
                "sample_id": record.sample_id,
                "audio_filepath": str(record.audio_filepath),
                "reference": record.text,
                "prediction": prediction,
                "audio_seconds": audio_duration_seconds(record.audio_filepath),
                "inference_seconds": inference_seconds,
            }
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    print(f"[predictions] Finished writing predictions: {prediction_path}")
    return prediction_path


def _progress(records: list[Any], desc: str, unit: str) -> Any:
    """Return a tqdm progress iterator when tqdm is installed."""

    try:
        from tqdm import tqdm

        return tqdm(records, desc=desc, unit=unit, ascii=True)
    except ImportError:
        print("[predictions] tqdm is not installed; continuing without a progress bar.")
        return records


def _predict_one(config: WhisperLargeRunConfig, bundle: WhisperBundle, audio_path: Path, reference: str) -> str:
    """Predict one transcript using either the smoke or Transformers backend."""

    if bundle.backend == "smoke":
        return bundle.model.predict(reference)

    import torch
    import torchaudio

    waveform, sample_rate = torchaudio.load(str(audio_path))
    if waveform.shape[0] > 1:
        waveform = waveform.mean(dim=0, keepdim=True)
    if sample_rate != 16_000:
        waveform = torchaudio.transforms.Resample(sample_rate, 16_000)(waveform)
    inputs = bundle.processor.feature_extractor(
        waveform.squeeze(0).numpy(),
        sampling_rate=16_000,
        return_tensors="pt",
    )
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = bundle.model.to(device)
    input_features = inputs.input_features.to(device)
    with torch.no_grad():
        predicted_ids = model.generate(input_features, max_new_tokens=config.generation_max_new_tokens)
    return bundle.processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
