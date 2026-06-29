#!/usr/bin/env python3
"""Generate ASR predictions separately from metric scoring."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np

from asr_universal_trainer import infer_model_spec, load_config, load_prepared_splits, load_run_result, slugify


def to_mono_float32(audio: Any) -> np.ndarray:
    arr = np.asarray(audio, dtype=np.float32)
    if arr.ndim == 0:
        return arr.reshape(1)
    if arr.ndim == 1:
        return arr
    if arr.ndim == 2:
        axis = 0 if arr.shape[0] < arr.shape[1] else 1
        return arr.mean(axis=axis).astype(np.float32)
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
        raise RuntimeError("Install soundfile or librosa for prediction generation.")

    if sample_rate != int(target_sample_rate):
        if librosa is None:
            raise RuntimeError("librosa is required for resampling when sample rates differ.")
        audio = librosa.resample(audio, orig_sr=int(sample_rate), target_sr=int(target_sample_rate))
        sample_rate = int(target_sample_rate)
    return to_mono_float32(audio), int(sample_rate)


def resolve_model_artifact(work_dir: Path, cfg: dict[str, Any], spec_model_id: str) -> str:
    model_override = cfg.get("model", {}).get("prediction_model_path")
    if model_override:
        return str(model_override)
    result = load_run_result(work_dir)
    train = result.get("train", {}) if isinstance(result, dict) else {}
    for key in ["best_model", "best_checkpoint", "runner"]:
        value = train.get(key)
        if value and Path(value).exists():
            return str(value)
    for candidate in [work_dir / "best_model", work_dir / "best_checkpoint"]:
        if candidate.exists():
            return str(candidate)
    return spec_model_id


def make_record(row: dict[str, Any], prediction: str, model_id: str, backend: str, split: str) -> dict[str, Any]:
    return {
        "uid": row.get("uid"),
        "audio_path": row.get("audio_path"),
        "split": split,
        "model_id": model_id,
        "backend": backend,
        "prediction": prediction,
        "reference": str(row.get("text") or ""),
    }


def write_prediction_outputs(work_dir: Path, model_id: str, split: str, records: list[dict[str, Any]], metadata: dict[str, Any]) -> tuple[Path, Path]:
    pred_dir = work_dir / "predictions"
    pred_dir.mkdir(parents=True, exist_ok=True)
    stem = f"{split}_{slugify(model_id)}"
    pred_path = pred_dir / f"{stem}.jsonl"
    meta_path = pred_dir / f"{stem}_metadata.json"
    with pred_path.open("w", encoding="utf-8") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
    meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return pred_path, meta_path


def predict_mock(rows: list[dict[str, Any]], model_id: str, split: str) -> list[dict[str, Any]]:
    return [make_record(row, str(row.get("text") or ""), model_id, "mock", split) for row in rows]


def predict_whisper(rows: list[dict[str, Any]], cfg: dict[str, Any], model_id: str, model_path: str, split: str) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

    sample_rate = int(cfg.get("model", {}).get("sample_rate", 16000))
    language = cfg.get("model", {}).get("language", "arabic")
    task = cfg.get("model", {}).get("task", "transcribe")
    max_new_tokens = int(cfg.get("evaluation", {}).get("max_new_tokens", 256))

    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        model_path,
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
    records = []
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
        prediction = processor.batch_decode(generated_ids, skip_special_tokens=True, clean_up_tokenization_spaces=True)[0]
        records.append(make_record(row, prediction, model_id, "hf_whisper_seq2seq", split))
    return records


def predict_ctc(rows: list[dict[str, Any]], cfg: dict[str, Any], model_id: str, model_path: str, split: str) -> list[dict[str, Any]]:
    import torch
    from transformers import AutoModelForCTC, AutoProcessor

    sample_rate = int(cfg.get("model", {}).get("sample_rate", 16000))
    processor = AutoProcessor.from_pretrained(model_path)
    model = AutoModelForCTC.from_pretrained(model_path)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    model.eval()
    records = []
    for row in rows:
        audio, audio_sr = load_audio(row["audio_path"], sample_rate)
        inputs = processor(audio, sampling_rate=audio_sr, return_tensors="pt")
        input_values = inputs.input_values.to(device)
        with torch.no_grad():
            logits = model(input_values=input_values).logits
        pred_ids = torch.argmax(logits, dim=-1)
        prediction = processor.batch_decode(pred_ids)[0]
        records.append(make_record(row, prediction, model_id, "hf_ctc", split))
    return records


def predict_qwen(rows: list[dict[str, Any]], cfg: dict[str, Any], model_id: str, model_path: str, split: str) -> list[dict[str, Any]]:
    import torch
    from qwen_asr import Qwen3ASRModel  # type: ignore

    model = Qwen3ASRModel.from_pretrained(
        model_path,
        dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
        device_map=cfg.get("model", {}).get("device_map", "cuda:0" if torch.cuda.is_available() else "cpu"),
        max_new_tokens=int(cfg.get("evaluation", {}).get("max_new_tokens", 256)),
    )
    language = cfg.get("model", {}).get("qwen_language", "Arabic")
    records = []
    for row in rows:
        out = model.transcribe(audio=row["audio_path"], language=language)
        prediction = out[0].text if out else ""
        records.append(make_record(row, prediction, model_id, "qwen_chat_asr", split))
    return records


def predict_cohere(rows: list[dict[str, Any]], model_id: str, model_path: str, split: str) -> list[dict[str, Any]]:
    from transformers import pipeline  # type: ignore

    pipe = pipeline("automatic-speech-recognition", model=model_path, trust_remote_code=True)
    records = []
    for row in rows:
        out = pipe(row["audio_path"])
        prediction = out.get("text", "") if isinstance(out, dict) else str(out)
        records.append(make_record(row, prediction, model_id, "cohere_eval_only", split))
    return records


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate predictions without scoring them.")
    parser.add_argument("--config", required=True, help="Config YAML/JSON path")
    parser.add_argument("--split", default=None, choices=[None, "train", "validation", "test"], help="Prepared split to predict")
    parser.add_argument("--output", default=None, help="Optional output JSONL path")
    args = parser.parse_args()

    cfg = load_config(args.config)
    model_id = cfg.get("model", {}).get("model_id", "mock")
    spec = infer_model_spec(model_id, cfg.get("model", {}).get("spec_overrides", {}))
    work_dir = Path(cfg.get("output", {}).get("work_dir", "asr_run")).resolve()
    split = args.split or cfg.get("evaluation", {}).get("split", "test")
    rows = load_prepared_splits(work_dir)[split]
    if not rows:
        raise ValueError(f"Prepared split {split} is empty in {work_dir}")

    model_path = resolve_model_artifact(work_dir, cfg, spec.model_id)
    if spec.backend == "mock":
        records = predict_mock(rows, spec.model_id, split)
    elif spec.backend == "hf_whisper_seq2seq":
        records = predict_whisper(rows, cfg, spec.model_id, model_path, split)
    elif spec.backend == "hf_ctc":
        records = predict_ctc(rows, cfg, spec.model_id, model_path, split)
    elif spec.backend == "qwen_chat_asr":
        records = predict_qwen(rows, cfg, spec.model_id, model_path, split)
    elif spec.backend == "cohere_eval_only":
        records = predict_cohere(rows, spec.model_id, model_path, split)
    else:
        raise RuntimeError(f"Prediction generation is not implemented for backend={spec.backend}: {spec.notes}")

    metadata = {
        "split": split,
        "n": len(records),
        "model_id": spec.model_id,
        "backend": spec.backend,
        "model_path": model_path,
        "work_dir": str(work_dir),
    }
    if args.output:
        pred_path = Path(args.output).resolve()
        pred_path.parent.mkdir(parents=True, exist_ok=True)
        with pred_path.open("w", encoding="utf-8") as handle:
            for record in records:
                handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        meta_path = pred_path.with_name(pred_path.stem + "_metadata.json")
        meta_path.write_text(json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    else:
        pred_path, meta_path = write_prediction_outputs(work_dir, spec.model_id, split, records, metadata)

    payload = {**metadata, "prediction_path": str(pred_path), "metadata_path": str(meta_path)}
    print(json.dumps(payload, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
