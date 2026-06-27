#!/usr/bin/env python3
"""Evaluate the trained Whisper checkpoint on a prepared split."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from asr_universal_trainer import compute_wer_cer, load_config, load_prepared_splits


def to_mono_float32(audio: Any) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        if arr.shape[0] < arr.shape[1]:
            return arr.mean(axis=0).astype(np.float32)
        return arr.mean(axis=1).astype(np.float32)
    return arr.reshape(-1).astype(np.float32)


def load_audio(path: str | Path, target_sample_rate: int) -> tuple[np.ndarray, int]:
    try:
        import soundfile as sf  # type: ignore
    except Exception:
        sf = None
    try:
        import librosa  # type: ignore
    except Exception:
        librosa = None

    if sf is not None:
        audio, sample_rate = sf.read(str(path), dtype="float32", always_2d=False)
        audio = to_mono_float32(audio)
    elif librosa is not None:
        audio, sample_rate = librosa.load(str(path), sr=None, mono=True)
        audio = to_mono_float32(audio)
    else:
        raise RuntimeError("Install soundfile or librosa for Whisper evaluation.")

    if sample_rate != int(target_sample_rate):
        if librosa is None:
            raise RuntimeError("librosa is required for resampling when sample rates differ.")
        audio = librosa.resample(audio, orig_sr=int(sample_rate), target_sr=int(target_sample_rate))
        sample_rate = int(target_sample_rate)
    return to_mono_float32(audio), int(sample_rate)


def resolve_checkpoint(work_dir: Path) -> str:
    result_path = work_dir / "run_result.json"
    if not result_path.exists():
        raise FileNotFoundError(f"Missing run result file: {result_path}")
    result = json.loads(result_path.read_text(encoding="utf-8"))
    checkpoint = result.get("train", {}).get("best_checkpoint")
    if not checkpoint:
        raise RuntimeError(f"Could not find best checkpoint in {result_path}")
    return str(checkpoint)


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate a trained Whisper checkpoint on a prepared split.")
    parser.add_argument("--config", required=True, help="Config YAML/JSON path")
    parser.add_argument("--split", default="test", choices=["train", "validation", "test"], help="Prepared split to evaluate")
    args = parser.parse_args()

    cfg = load_config(args.config)
    work_dir = Path(cfg.get("output", {}).get("work_dir", "asr_run")).resolve()
    sample_rate = int(cfg.get("model", {}).get("sample_rate", 16000))
    language = cfg.get("model", {}).get("language", "arabic")
    task = cfg.get("model", {}).get("task", "transcribe")
    max_new_tokens = int(cfg.get("evaluation", {}).get("max_new_tokens", 256))
    metric_normalizer = cfg.get("evaluation", {}).get("metric_normalizer", "arabic_basic")

    rows = load_prepared_splits(work_dir)[args.split]
    if not rows:
        raise ValueError(f"Prepared split {args.split} is empty in {work_dir}")

    checkpoint_path = resolve_checkpoint(work_dir)
    processor = AutoProcessor.from_pretrained(cfg.get("model", {}).get("model_id", "openai/whisper-large-v3"))
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        checkpoint_path,
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    try:
        model.generation_config.language = language
        model.generation_config.task = task
        model.generation_config.forced_decoder_ids = None
    except Exception:
        pass

    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()

    predictions: list[str] = []
    references: list[str] = []
    records: list[dict[str, Any]] = []
    for row in rows:
        audio, audio_sr = load_audio(row["audio_path"], sample_rate)
        inputs = processor(audio, sampling_rate=audio_sr, return_tensors="pt")
        input_features = inputs.input_features.to(device=device, dtype=getattr(model, "dtype", inputs.input_features.dtype))
        with torch.no_grad():
            generated_ids = model.generate(
                input_features=input_features,
                max_new_tokens=max_new_tokens,
                language=language,
                task=task,
            )
        prediction = processor.batch_decode(
            generated_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )[0]
        reference = str(row.get("text") or "")
        predictions.append(prediction)
        references.append(reference)
        records.append(
            {
                "uid": row.get("uid"),
                "audio_path": row.get("audio_path"),
                "split": args.split,
                "prediction": prediction,
                "reference": reference,
            }
        )

    metrics = compute_wer_cer(predictions, references, metric_normalizer)
    prediction_path = work_dir / f"whisper_{args.split}_predictions.json"
    metrics_path = work_dir / f"whisper_{args.split}_metrics.json"
    prediction_path.write_text(json.dumps(records, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    metrics_payload = {
        "split": args.split,
        "n": len(rows),
        "prediction_path": str(prediction_path),
        **metrics,
    }
    metrics_path.write_text(json.dumps(metrics_payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(metrics_payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
