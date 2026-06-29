#!/usr/bin/env python3
"""
Universal ASR plug-and-play trainer/evaluator.

Goal:
  One common dataset schema + model-family adapters.

Supported backends:
  - mock: offline smoke test, no heavy ML dependencies.
  - hf_whisper_seq2seq: Whisper-style encoder-decoder ASR via Transformers Seq2SeqTrainer.
  - hf_ctc: wav2vec2 / HuBERT / WavLM / generic AutoModelForCTC via Transformers Trainer.
  - qwen_chat_asr: Qwen3-ASR data formatting + inference/eval via qwen-asr; experimental training path.
  - omni_llm / omni_ctc: official local omnilingual-asr recipe + inference pipeline when the repo is available.
  - nemo_fastconformer: exports NeMo manifest JSONL and runner command for NVIDIA NeMo.
  - cohere_transcribe: inference/eval path; training intentionally disabled unless Cohere publishes a supported fine-tuning API.

The important design choice: the dataset is universal; the collator/trainer is not.
"""
from __future__ import annotations

import argparse
import csv
import dataclasses
import json
import logging
import math
import os
import random
import re
import shutil
import subprocess
import sys
import time
import traceback
import wave
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from omnilingual_backend import eval_omnilingual, prepare_omnilingual_recipe_dataset, train_omnilingual

try:
    import inspect
    from accelerate import Accelerator
except Exception:  # pragma: no cover - optional at import time
    Accelerator = None
else:
    # Transformers 4.57 passes keep_torch_compile; older Accelerate releases ignore it.
    if "keep_torch_compile" not in inspect.signature(Accelerator.unwrap_model).parameters:
        _orig_unwrap_model = Accelerator.unwrap_model

        def _compat_unwrap_model(self, model, keep_fp32_wrapper: bool = True, keep_torch_compile: bool = False):
            return _orig_unwrap_model(self, model, keep_fp32_wrapper=keep_fp32_wrapper)

        Accelerator.unwrap_model = _compat_unwrap_model

# -----------------------------
# Logging
# -----------------------------

LOGGER = logging.getLogger("asr_universal")


def setup_logging(work_dir: str | Path, level: str = "INFO") -> Path:
    Path(work_dir).mkdir(parents=True, exist_ok=True)
    log_path = Path(work_dir) / "run.log"
    LOGGER.setLevel(getattr(logging, level.upper(), logging.INFO))
    LOGGER.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    stream = logging.StreamHandler(sys.stdout)
    stream.setFormatter(fmt)
    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(fmt)
    LOGGER.addHandler(stream)
    LOGGER.addHandler(file_handler)
    LOGGER.info("Logging to %s", log_path)
    return log_path


# -----------------------------
# Config IO
# -----------------------------


def load_config(path: str | Path) -> Dict[str, Any]:
    path = Path(path).resolve()
    text = path.read_text(encoding="utf-8")
    if path.suffix.lower() in {".json"}:
        cfg = json.loads(text)
    else:
        try:
            import yaml  # type: ignore
            cfg = yaml.safe_load(text)
        except Exception:
            # very small fallback: allow JSON-in-YAML files
            cfg = json.loads(text)

    if isinstance(cfg, dict):
        meta = dict(cfg.get("_meta", {}) or {})
        meta["config_path"] = str(path)
        meta["config_dir"] = str(path.parent)
        cfg["_meta"] = meta
        return cfg
    raise TypeError(f"Config at {path} must parse to a mapping/dict, got {type(cfg).__name__}")


def dump_json(path: str | Path, obj: Any) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def slugify(value: str) -> str:
    text = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip())
    text = text.strip("._-")
    return text or "item"


def _resolve_path_like(value: str | Path, base_dir: Path) -> str:
    path = Path(value)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def resolve_data_conf(data_conf: Dict[str, Any], base_dir: str | Path | None = None) -> Dict[str, Any]:
    data_conf = dict(data_conf or {})
    base_path = Path(base_dir).resolve() if base_dir else None
    dataset_name = str(data_conf.get("dataset_name") or "").strip()
    if data_conf.get("path") and base_path is not None:
        data_conf["path"] = _resolve_path_like(data_conf["path"], base_path)
    if data_conf.get("split_paths") and base_path is not None:
        data_conf["split_paths"] = {
            split: _resolve_path_like(path, base_path)
            for split, path in dict(data_conf["split_paths"]).items()
        }
    # When a direct path/split-paths format is supplied we keep the config as-is.
    # A named dataset is treated as a local registry entry for the notebook/smoke workflow.
    if not dataset_name or data_conf.get("format") == "hf" or data_conf.get("path") or data_conf.get("split_paths"):
        return data_conf

    registry_value = data_conf.get("dataset_registry_path")
    if registry_value:
        registry_path = Path(registry_value)
        if not registry_path.is_absolute() and base_path is not None:
            registry_path = (base_path / registry_path).resolve()
        else:
            registry_path = registry_path.resolve()
    else:
        registry_path = (Path(__file__).resolve().parent / "datasets" / "registry.json").resolve()
    registry_obj = load_config(registry_path)
    registry = registry_obj.get("datasets", registry_obj)
    if dataset_name not in registry:
        raise KeyError(f"Dataset name '{dataset_name}' not found in registry: {registry_path}")

    entry = dict(registry[dataset_name] or {})
    resolved = dict(entry)
    for key, value in data_conf.items():
        if key in {"dataset_name", "dataset_registry_path", "columns"}:
            continue
        resolved[key] = value

    entry_cols = dict(entry.get("columns", {}) or {})
    override_cols = dict(data_conf.get("columns", {}) or {})
    if entry_cols or override_cols:
        entry_cols.update(override_cols)
        resolved["columns"] = entry_cols

    if resolved.get("path"):
        resolved["path"] = _resolve_path_like(resolved["path"], registry_path.parent)
    if resolved.get("split_paths"):
        resolved["split_paths"] = {
            split: _resolve_path_like(path, registry_path.parent)
            for split, path in dict(resolved["split_paths"]).items()
        }
    resolved["dataset_name"] = dataset_name
    resolved["dataset_registry_path"] = str(registry_path)
    return resolved


def read_jsonl(path: str | Path) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def write_jsonl(path: str | Path, rows: Iterable[Dict[str, Any]]) -> int:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    n = 0
    with open(path, "w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
            n += 1
    return n


# -----------------------------
# Model registry
# -----------------------------


@dataclass
class ModelSpec:
    model_id: str
    family: str
    backend: str
    max_train_seconds: Optional[float]
    max_eval_chunk_seconds: Optional[float]
    sample_rate: int = 16000
    requires_chat_style: bool = False
    train_supported_in_this_script: bool = True
    eval_supported_in_this_script: bool = True
    notes: str = ""


def infer_model_spec(model_id: str, overrides: Optional[Dict[str, Any]] = None) -> ModelSpec:
    m = model_id.lower()
    spec: ModelSpec
    if model_id == "mock":
        spec = ModelSpec(
            model_id=model_id,
            family="mock",
            backend="mock",
            max_train_seconds=30,
            max_eval_chunk_seconds=30,
            notes="No-dependency smoke-test backend.",
        )
    elif "whisper" in m:
        spec = ModelSpec(
            model_id=model_id,
            family="whisper_seq2seq",
            backend="hf_whisper_seq2seq",
            max_train_seconds=30,
            max_eval_chunk_seconds=30,
            notes="Whisper has a 30s receptive field; train/eval should use <=30s segments or chunk long audio.",
        )
    elif "qwen" in m and "asr" in m:
        spec = ModelSpec(
            model_id=model_id,
            family="qwen_chat_asr",
            backend="qwen_chat_asr",
            max_train_seconds=None,
            max_eval_chunk_seconds=None,
            requires_chat_style=True,
            train_supported_in_this_script=False,
            eval_supported_in_this_script=True,
            notes=(
                "Qwen3-ASR uses qwen-asr/vLLM inference with audio_url/chat-like content. "
                "This script prepares chat-style JSONL and can evaluate through qwen-asr if installed. "
                "Training is marked experimental because the official card documents inference, not a stable public HF Trainer recipe."
            ),
        )
    elif "omnilingual" in m or "omniasr" in m:
        is_ctc = "ctc" in m
        spec = ModelSpec(
            model_id=model_id,
            family="omni_ctc" if is_ctc else "omni_llm",
            backend="omnilingual_recipe",
            max_train_seconds=30,
            max_eval_chunk_seconds=30,
            train_supported_in_this_script=True,
            eval_supported_in_this_script=True,
            notes=(
                "Omnilingual ASR uses the local third_party/omnilingual-asr recipe. "
                "This trainer converts the prepared dataset into the recipe's parquet/card format and runs the official workflow."
            ),
        )
    elif "cohere-transcribe" in m:
        spec = ModelSpec(
            model_id=model_id,
            family="cohere_transcribe",
            backend="cohere_eval_only",
            max_train_seconds=None,
            max_eval_chunk_seconds=None,
            train_supported_in_this_script=False,
            eval_supported_in_this_script=True,
            notes=(
                "Cohere Transcribe is exposed for Transformers/vLLM inference. Public fine-tuning API/recipe is not assumed."
            ),
        )
    elif "nvidia/stt_" in m or "fastconformer" in m or "nemo" in m:
        spec = ModelSpec(
            model_id=model_id,
            family="nemo_fastconformer_hybrid",
            backend="nemo_fastconformer",
            max_train_seconds=None,
            max_eval_chunk_seconds=None,
            train_supported_in_this_script=True,
            eval_supported_in_this_script=True,
            notes="NVIDIA FastConformer hybrid RNNT/CTC is trained/evaluated via NeMo manifest JSONL and NeMo scripts.",
        )
    else:
        spec = ModelSpec(
            model_id=model_id,
            family="hf_ctc",
            backend="hf_ctc",
            max_train_seconds=None,
            max_eval_chunk_seconds=None,
            notes="Defaulting to Transformers AutoModelForCTC. Override model.family/backend if wrong.",
        )

    overrides = overrides or {}
    for k, v in overrides.items():
        if hasattr(spec, k) and v is not None:
            setattr(spec, k, v)
    return spec


# -----------------------------
# Text normalization + metrics
# -----------------------------

_AR_DIACRITICS = re.compile(r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]")
_PUNCT = re.compile(r"[\.,!\?;:\"'`،؛؟\(\)\[\]\{\}\-–—_/\\]+")
_SPACE = re.compile(r"\s+")


def normalize_arabic_basic(text: str, remove_diacritics: bool = True, remove_punct: bool = True) -> str:
    text = str(text or "").strip()
    if remove_diacritics:
        text = _AR_DIACRITICS.sub("", text)
    # Common Arabic character normalization. Keep dialectal spelling as much as possible.
    text = text.replace("إ", "ا").replace("أ", "ا").replace("آ", "ا")
    text = text.replace("ى", "ي").replace("ؤ", "و").replace("ئ", "ي")
    if remove_punct:
        text = _PUNCT.sub(" ", text)
    text = _SPACE.sub(" ", text).strip()
    return text


def normalize_for_metric(text: str, mode: str = "arabic_basic") -> str:
    if mode in {"arabic_basic", "arabic_no_punct"}:
        return normalize_arabic_basic(text, remove_diacritics=True, remove_punct=True)
    if mode == "none":
        return str(text or "")
    return normalize_arabic_basic(text)


def edit_distance(a: Sequence[Any], b: Sequence[Any]) -> int:
    # O(min(n,m)) memory Levenshtein.
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            ins = curr[j - 1] + 1
            delete = prev[j] + 1
            sub = prev[j - 1] + (ca != cb)
            curr.append(min(ins, delete, sub))
        prev = curr
    return prev[-1]


def compute_wer_cer(preds: Sequence[str], refs: Sequence[str], normalizer: str = "arabic_basic") -> Dict[str, float]:
    total_word_edits = 0
    total_words = 0
    total_char_edits = 0
    total_chars = 0
    for p, r in zip(preds, refs):
        pn = normalize_for_metric(p, normalizer)
        rn = normalize_for_metric(r, normalizer)
        pw, rw = pn.split(), rn.split()
        total_word_edits += edit_distance(pw, rw)
        total_words += max(1, len(rw))
        total_char_edits += edit_distance(list(pn), list(rn))
        total_chars += max(1, len(rn))
    return {
        "wer": total_word_edits / total_words if total_words else 0.0,
        "cer": total_char_edits / total_chars if total_chars else 0.0,
    }


# -----------------------------
# Audio helpers
# -----------------------------


def wav_duration(path: str | Path) -> Optional[float]:
    try:
        with wave.open(str(path), "rb") as w:
            return w.getnframes() / float(w.getframerate())
    except Exception:
        return None


def make_tiny_wav(path: str | Path, seconds: float = 0.25, sample_rate: int = 16000, freq: float = 440.0) -> None:
    import struct
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    n = int(seconds * sample_rate)
    with wave.open(str(path), "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(sample_rate)
        frames = bytearray()
        for i in range(n):
            sample = int(0.15 * 32767 * math.sin(2 * math.pi * freq * i / sample_rate))
            frames.extend(struct.pack("<h", sample))
        w.writeframes(bytes(frames))


def copy_first_chunk_wav(src: str | Path, dst: str | Path, max_seconds: float) -> bool:
    try:
        with wave.open(str(src), "rb") as r:
            fr = r.getframerate()
            n = min(r.getnframes(), int(max_seconds * fr))
            params = r.getparams()
            data = r.readframes(n)
        Path(dst).parent.mkdir(parents=True, exist_ok=True)
        with wave.open(str(dst), "wb") as w:
            w.setparams(params)
            w.writeframes(data)
        return True
    except Exception as e:
        LOGGER.warning("Could not first-chunk %s: %s", src, e)
        return False


# -----------------------------
# Dataset preparation
# -----------------------------


def _get_nested(row: Dict[str, Any], key: str, default: Any = None) -> Any:
    if not key:
        return default
    cur: Any = row
    for part in key.split("."):
        if isinstance(cur, dict) and part in cur:
            cur = cur[part]
        else:
            return default
    return cur


def _resolve_audio_path(value: Any) -> Optional[str]:
    if value is None:
        return None
    if isinstance(value, str):
        return value
    if isinstance(value, dict):
        for k in ("path", "audio_filepath", "file", "filename"):
            if value.get(k):
                return str(value[k])
    return None


def read_source_rows(data_conf: Dict[str, Any]) -> List[Dict[str, Any]]:
    fmt = data_conf.get("format", "jsonl")
    rows: List[Dict[str, Any]] = []
    if "split_paths" in data_conf:
        for split, p in data_conf["split_paths"].items():
            sub_conf = dict(data_conf)
            sub_conf.pop("split_paths", None)
            sub_conf["path"] = p
            sub_rows = read_source_rows(sub_conf)
            for r in sub_rows:
                r.setdefault("split", split)
            rows.extend(sub_rows)
        return rows

    if fmt == "hf":
        from datasets import load_dataset  # type: ignore
        name = data_conf.get("hf_dataset_name") or data_conf["dataset_name"]
        subset = data_conf.get("subset")
        split = data_conf.get("split", "train")
        LOGGER.info("Loading HF dataset %s subset=%s split=%s", name, subset, split)
        ds = load_dataset(name, subset, split=split) if subset else load_dataset(name, split=split)
        max_rows = data_conf.get("max_rows")
        if max_rows:
            ds = ds.select(range(min(int(max_rows), len(ds))))
        return [dict(x) for x in ds]

    path = Path(data_conf["path"])
    LOGGER.info("Reading %s dataset: %s", fmt, path)
    if fmt in {"jsonl", "manifest"}:
        rows = read_jsonl(path)
    elif fmt == "json":
        obj = json.loads(path.read_text(encoding="utf-8"))
        rows = obj if isinstance(obj, list) else obj.get("data", [])
    elif fmt == "csv":
        with open(path, newline="", encoding="utf-8") as f:
            rows = [dict(r) for r in csv.DictReader(f)]
    elif fmt == "parquet":
        import pandas as pd  # type: ignore
        df = pd.read_parquet(path)
        rows = df.to_dict("records")
    else:
        raise ValueError(f"Unsupported data.format={fmt}")
    return rows


def normalize_rows_to_common(rows: List[Dict[str, Any]], cfg: Dict[str, Any], spec: ModelSpec) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    data_conf = cfg.get("data", {})
    out_conf = cfg.get("output", {})
    work_dir = Path(out_conf.get("work_dir", "asr_run"))
    col = data_conf.get("columns", {})
    audio_col = col.get("audio", data_conf.get("audio_column", "audio_path"))
    text_col = col.get("text", data_conf.get("text_column", "text"))
    dur_col = col.get("duration", data_conf.get("duration_column", "duration"))
    split_col = col.get("split", data_conf.get("split_column", "split"))
    uid_col = col.get("uid", data_conf.get("uid_column", "uid"))
    speaker_col = col.get("speaker_id", data_conf.get("speaker_column", "speaker_id"))
    language_col = col.get("language", data_conf.get("language_column", "language"))

    sample_rate = int(cfg.get("model", {}).get("sample_rate", spec.sample_rate))
    min_seconds = float(data_conf.get("min_seconds", 0.05))
    max_train_seconds = cfg.get("model", {}).get("max_train_seconds", spec.max_train_seconds)
    if max_train_seconds is not None:
        max_train_seconds = float(max_train_seconds)
    long_policy = data_conf.get("long_audio_policy", "drop")  # drop | keep | first_chunk | error
    metric_norm = cfg.get("evaluation", {}).get("metric_normalizer", "arabic_basic")
    prompt_template = cfg.get("model", {}).get("prompt_template", "Transcribe this audio in Arabic.")

    common: List[Dict[str, Any]] = []
    stats = {
        "input_rows": len(rows),
        "kept_rows": 0,
        "dropped_empty_text": 0,
        "dropped_missing_audio_path": 0,
        "dropped_too_short": 0,
        "dropped_too_long_train": 0,
        "first_chunked": 0,
        "duration_missing": 0,
    }

    for i, r in enumerate(rows):
        split = str(_get_nested(r, split_col, "") or "").lower().strip()
        text_raw = _get_nested(r, text_col, None)
        text = str(text_raw or "").strip()
        if not text:
            stats["dropped_empty_text"] += 1
            continue
        audio_value = _get_nested(r, audio_col, None)
        audio_path = _resolve_audio_path(audio_value)
        if not audio_path:
            # HF audio dict may not have path if loaded as array. Keep it only for real HF pipelines.
            stats["dropped_missing_audio_path"] += 1
            continue
        duration_v = _get_nested(r, dur_col, None)
        try:
            duration = float(duration_v) if duration_v is not None and duration_v != "" else None
        except Exception:
            duration = None
        if duration is None:
            duration = wav_duration(audio_path)
            if duration is None:
                stats["duration_missing"] += 1
                duration = -1.0
        if duration >= 0 and duration < min_seconds:
            stats["dropped_too_short"] += 1
            continue
        if not split:
            split = "train"
        uid = str(_get_nested(r, uid_col, None) or f"utt_{i:09d}")
        row = {
            "uid": uid,
            "audio_path": str(audio_path),
            "text": text,
            "text_norm": normalize_for_metric(text, metric_norm),
            "duration": float(duration),
            "split": split,
            "language": str(_get_nested(r, language_col, data_conf.get("language", "ar")) or "ar"),
            "speaker_id": str(_get_nested(r, speaker_col, "") or ""),
            "prompt": prompt_template,
            "metadata": {"source_index": i},
        }
        if spec.requires_chat_style:
            row["messages"] = [
                {
                    "role": "user",
                    "content": [
                        {"type": "audio_url", "audio_url": {"url": str(audio_path)}},
                        {"type": "text", "text": prompt_template},
                    ],
                },
                {"role": "assistant", "content": text},
            ]
        if split == "train" and max_train_seconds is not None and duration > max_train_seconds:
            if long_policy == "drop":
                stats["dropped_too_long_train"] += 1
                continue
            if long_policy == "error":
                raise ValueError(f"Train row {uid} duration={duration:.2f}s exceeds max_train_seconds={max_train_seconds}")
            if long_policy == "first_chunk":
                chunk_path = work_dir / "first_chunks" / f"{Path(audio_path).stem}_{uid}_first{int(max_train_seconds)}s.wav"
                if copy_first_chunk_wav(audio_path, chunk_path, max_train_seconds):
                    row["audio_path"] = str(chunk_path)
                    row["duration"] = max_train_seconds
                    row["metadata"]["first_chunk_from"] = str(audio_path)
                    stats["first_chunked"] += 1
                else:
                    stats["dropped_too_long_train"] += 1
                    continue
            # keep means do nothing.
        common.append(row)

    # Split rows if only train provided.
    if not any(r["split"] in {"validation", "val", "dev", "test"} for r in common):
        seed = int(data_conf.get("seed", 42))
        rng = random.Random(seed)
        rng.shuffle(common)
        val_ratio = float(data_conf.get("val_ratio", 0.1))
        test_ratio = float(data_conf.get("test_ratio", 0.1))
        n = len(common)
        n_test = max(1, int(n * test_ratio)) if n >= 3 and test_ratio > 0 else 0
        n_val = max(1, int(n * val_ratio)) if n >= 2 and val_ratio > 0 else 0
        for idx, row in enumerate(common):
            if idx < n_test:
                row["split"] = "test"
            elif idx < n_test + n_val:
                row["split"] = "validation"
            else:
                row["split"] = "train"
    # canonical split names
    for r in common:
        if r["split"] in {"val", "dev"}:
            r["split"] = "validation"
    stats["kept_rows"] = len(common)
    stats["splits"] = {s: sum(1 for r in common if r["split"] == s) for s in sorted({r["split"] for r in common})}
    return common, stats


def prepare_dataset(cfg: Dict[str, Any], spec: ModelSpec) -> Dict[str, str]:
    out_conf = cfg.get("output", {})
    work_dir = Path(out_conf.get("work_dir", "asr_run"))
    prepared_dir = work_dir / "prepared"
    prepared_dir.mkdir(parents=True, exist_ok=True)
    config_dir = cfg.get("_meta", {}).get("config_dir")
    resolved_data_conf = resolve_data_conf(cfg.get("data", {}), base_dir=config_dir)
    cfg = dict(cfg)
    cfg["data"] = resolved_data_conf
    rows = read_source_rows(resolved_data_conf)
    common, stats = normalize_rows_to_common(rows, cfg, spec)
    LOGGER.info("Prepared %d/%d rows. Splits=%s", stats["kept_rows"], stats["input_rows"], stats.get("splits"))
    paths: Dict[str, str] = {}
    for split in ["train", "validation", "test"]:
        split_rows = [r for r in common if r["split"] == split]
        path = prepared_dir / f"{split}.jsonl"
        write_jsonl(path, split_rows)
        paths[split] = str(path)
        LOGGER.info("Wrote %s rows: %s", len(split_rows), path)

    # NeMo manifests are useful even if using a different backend.
    nemo_dir = prepared_dir / "nemo_manifests"
    nemo_paths = {}
    for split in ["train", "validation", "test"]:
        split_rows = [r for r in common if r["split"] == split]
        manifest_rows = [
            {"audio_filepath": str(Path(r["audio_path"]).absolute()), "text": r["text"], "duration": r.get("duration", -1)}
            for r in split_rows
        ]
        mp = nemo_dir / f"{split}.json"
        write_jsonl(mp, manifest_rows)
        nemo_paths[split] = str(mp)
    paths["nemo_train_manifest"] = nemo_paths["train"]
    paths["nemo_val_manifest"] = nemo_paths["validation"]
    paths["nemo_test_manifest"] = nemo_paths["test"]

    dump_json(prepared_dir / "stats.json", stats)
    dump_json(prepared_dir / "model_spec.json", dataclasses.asdict(spec))
    dump_json(prepared_dir / "dataset_resolved.json", resolved_data_conf)
    paths["stats"] = str(prepared_dir / "stats.json")
    paths["model_spec"] = str(prepared_dir / "model_spec.json")
    paths["dataset_resolved"] = str(prepared_dir / "dataset_resolved.json")

    if spec.requires_chat_style:
        chat_path = prepared_dir / "qwen_chat_train.jsonl"
        write_jsonl(chat_path, [r for r in common if r["split"] == "train"])
        paths["qwen_chat_train"] = str(chat_path)
    return paths


# -----------------------------
# Trainer utils
# -----------------------------


def load_prepared_splits(work_dir: str | Path) -> Dict[str, List[Dict[str, Any]]]:
    prepared_dir = Path(work_dir) / "prepared"
    return {
        "train": read_jsonl(prepared_dir / "train.jsonl"),
        "validation": read_jsonl(prepared_dir / "validation.jsonl"),
        "test": read_jsonl(prepared_dir / "test.jsonl"),
    }


class MetricsCSVLogger:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if not self.path.exists():
            with open(self.path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "wer", "cer", "is_best", "time"])
                writer.writeheader()

    def append(self, row: Dict[str, Any]) -> None:
        with open(self.path, "a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["epoch", "train_loss", "val_loss", "wer", "cer", "is_best", "time"])
            writer.writerow({k: row.get(k, "") for k in writer.fieldnames})


class MockTrainer:
    def __init__(self, cfg: Dict[str, Any], spec: ModelSpec):
        self.cfg = cfg
        self.spec = spec
        self.work_dir = Path(cfg.get("output", {}).get("work_dir", "asr_run"))
        self.metrics_logger = MetricsCSVLogger(self.work_dir / "metrics.csv")

    def train(self) -> Dict[str, Any]:
        splits = load_prepared_splits(self.work_dir)
        epochs = int(self.cfg.get("training", {}).get("num_train_epochs", 1))
        patience = int(self.cfg.get("training", {}).get("early_stopping_patience", 5))
        best_wer = float("inf")
        bad_epochs = 0
        best_epoch = 0
        LOGGER.info("[mock] training rows=%d val rows=%d", len(splits["train"]), len(splits["validation"]))
        for epoch in range(1, epochs + 1):
            # Fake deterministic improvement for smoke run.
            train_loss = max(0.05, 1.0 / epoch)
            val_loss = max(0.05, 1.1 / epoch)
            refs = [r["text"] for r in splits["validation"]]
            preds = refs[:]  # perfect fake predictions
            metrics = compute_wer_cer(preds, refs, self.cfg.get("evaluation", {}).get("metric_normalizer", "arabic_basic"))
            wer = metrics["wer"]
            is_best = wer < best_wer
            if is_best:
                best_wer = wer
                best_epoch = epoch
                bad_epochs = 0
                best_dir = self.work_dir / "best_checkpoint"
                best_dir.mkdir(exist_ok=True)
                (best_dir / "MOCK_BEST.txt").write_text(f"epoch={epoch}\nwer={wer}\n", encoding="utf-8")
            else:
                bad_epochs += 1
            LOGGER.info(
                "EPOCH %d | train_loss=%.4f | val_loss=%.4f | WER=%.4f | CER=%.4f | best=%s",
                epoch,
                train_loss,
                val_loss,
                wer,
                metrics["cer"],
                is_best,
            )
            self.metrics_logger.append(
                {"epoch": epoch, "train_loss": train_loss, "val_loss": val_loss, "wer": wer, "cer": metrics["cer"], "is_best": is_best, "time": time.time()}
            )
            if bad_epochs >= patience:
                LOGGER.info("Early stopping after %d bad epochs (patience=%d)", bad_epochs, patience)
                break
        return {"best_wer": best_wer, "best_epoch": best_epoch, "best_checkpoint": str(self.work_dir / "best_checkpoint")}

    def evaluate(self, split: str = "test") -> Dict[str, Any]:
        splits = load_prepared_splits(self.work_dir)
        refs = [r["text"] for r in splits[split]]
        preds = refs[:]
        metrics = compute_wer_cer(preds, refs, self.cfg.get("evaluation", {}).get("metric_normalizer", "arabic_basic"))
        out = {"split": split, **metrics, "n": len(refs)}
        dump_json(self.work_dir / f"mock_{split}_metrics.json", out)
        LOGGER.info("[mock] %s metrics: %s", split, out)
        return out


# -----------------------------
# Hugging Face Whisper seq2seq
# -----------------------------


def _make_eval_strategy_args(args_cls: Any, args: Dict[str, Any]) -> Any:
    try:
        return args_cls(evaluation_strategy="epoch", **args)
    except TypeError:
        return args_cls(eval_strategy="epoch", **args)


def train_hf_whisper_seq2seq(cfg: Dict[str, Any], spec: ModelSpec) -> Dict[str, Any]:
    try:
        import numpy as np  # noqa: F401
        import torch
        from datasets import Audio, Dataset
        from transformers import (
            AutoModelForSpeechSeq2Seq,
            AutoProcessor,
            EarlyStoppingCallback,
            Seq2SeqTrainer,
            Seq2SeqTrainingArguments,
        )
    except Exception as e:
        raise RuntimeError("Install requirements for Whisper training: transformers datasets evaluate accelerate soundfile librosa torch") from e

    work_dir = Path(cfg.get("output", {}).get("work_dir", "asr_run"))
    splits = load_prepared_splits(work_dir)
    if not splits["train"] or not splits["validation"]:
        raise ValueError("Need non-empty train and validation splits.")

    processor = AutoProcessor.from_pretrained(spec.model_id)
    torch_dtype = torch.bfloat16 if cfg.get("training", {}).get("bf16", True) and torch.cuda.is_available() else torch.float32
    model = AutoModelForSpeechSeq2Seq.from_pretrained(
        spec.model_id,
        torch_dtype=torch_dtype,
        low_cpu_mem_usage=True,
        use_safetensors=True,
    )
    if cfg.get("model", {}).get("gradient_checkpointing", True):
        model.gradient_checkpointing_enable()
        if hasattr(model.config, "use_cache"):
            model.config.use_cache = False

    language = cfg.get("model", {}).get("language", "arabic")
    task = cfg.get("model", {}).get("task", "transcribe")
    try:
        model.generation_config.language = language
        model.generation_config.task = task
        model.generation_config.forced_decoder_ids = None
    except Exception:
        pass

    def to_ds(rows: List[Dict[str, Any]]) -> Any:
        ds = Dataset.from_list(rows)
        ds = ds.rename_column("audio_path", "audio")
        return ds.cast_column("audio", Audio(sampling_rate=int(cfg.get("model", {}).get("sample_rate", spec.sample_rate))))

    train_ds = to_ds(splits["train"])
    val_ds = to_ds(splits["validation"])
    metric_norm = cfg.get("evaluation", {}).get("metric_normalizer", "arabic_basic")
    max_label_length = int(getattr(model.config, "max_target_positions", 448))

    def prepare(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio = batch["audio"]
        batch["input_features"] = processor.feature_extractor(audio["array"], sampling_rate=audio["sampling_rate"]).input_features[0]
        labels = processor.tokenizer(batch["text"]).input_ids
        if len(labels) > max_label_length:
            labels = labels[:max_label_length]
        batch["labels"] = labels
        return batch

    train_ds = train_ds.map(prepare, remove_columns=train_ds.column_names, desc="Whisper preprocess train")
    val_ds = val_ds.map(prepare, remove_columns=val_ds.column_names, desc="Whisper preprocess val")

    class DataCollatorSpeechSeq2SeqWithPadding:
        def __init__(self, processor: Any, decoder_start_token_id: int):
            self.processor = processor
            self.decoder_start_token_id = decoder_start_token_id

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            input_features = [{"input_features": f["input_features"]} for f in features]
            batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
            label_features = [{"input_ids": f["labels"]} for f in features]
            labels_batch = self.processor.tokenizer.pad(label_features, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            if labels.shape[1] > 0 and (labels[:, 0] == self.decoder_start_token_id).all().cpu().item():
                labels = labels[:, 1:]
            batch["labels"] = labels
            return batch

    data_collator = DataCollatorSpeechSeq2SeqWithPadding(processor, model.config.decoder_start_token_id)

    def compute_metrics(pred: Any) -> Dict[str, float]:
        pred_ids = pred.predictions
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        pred_str = processor.batch_decode(pred_ids, skip_special_tokens=True)
        label_str = processor.batch_decode(label_ids, skip_special_tokens=True)
        return compute_wer_cer(pred_str, label_str, metric_norm)

    training = cfg.get("training", {})
    args = {
        "output_dir": str(work_dir / "checkpoints"),
        "per_device_train_batch_size": int(training.get("per_device_train_batch_size", 4)),
        "per_device_eval_batch_size": int(training.get("per_device_eval_batch_size", 4)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 4)),
        "learning_rate": float(training.get("learning_rate", 1e-5)),
        "warmup_ratio": float(training.get("warmup_ratio", 0.05)),
        "num_train_epochs": float(training.get("num_train_epochs", 10)),
        "gradient_checkpointing": bool(training.get("gradient_checkpointing", cfg.get("model", {}).get("gradient_checkpointing", True))),
        "fp16": bool(training.get("fp16", False)),
        "bf16": bool(training.get("bf16", True) and torch.cuda.is_available()),
        "predict_with_generate": True,
        "generation_max_length": int(training.get("generation_max_length", 256)),
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": int(training.get("logging_steps", 10)),
        "report_to": training.get("report_to", "none"),
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "save_total_limit": int(training.get("save_total_limit", 3)),
        "remove_unused_columns": False,
    }
    train_args = _make_eval_strategy_args(Seq2SeqTrainingArguments, args)
    trainer = Seq2SeqTrainer(
        args=train_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=data_collator,
        compute_metrics=compute_metrics,
        tokenizer=processor.feature_extractor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(training.get("early_stopping_patience", 5)))],
    )
    LOGGER.info("Starting Whisper/HF Seq2Seq training. Best model selected by lowest WER.")
    result = trainer.train()
    trainer.save_model(str(work_dir / "best_model"))
    processor.save_pretrained(str(work_dir / "best_model"))
    metrics = trainer.evaluate()
    LOGGER.info("Final validation metrics: %s", metrics)
    dump_json(work_dir / "train_result.json", {"train": str(result), "eval": metrics})
    return {"best_model": str(work_dir / "best_model"), "eval": metrics}


# -----------------------------
# Hugging Face CTC
# -----------------------------


def train_hf_ctc(cfg: Dict[str, Any], spec: ModelSpec) -> Dict[str, Any]:
    try:
        import numpy as np
        import torch
        from datasets import Audio, Dataset
        from transformers import AutoModelForCTC, AutoProcessor, EarlyStoppingCallback, Trainer, TrainingArguments
    except Exception as e:
        raise RuntimeError("Install requirements for CTC training: transformers datasets accelerate soundfile librosa torch") from e

    work_dir = Path(cfg.get("output", {}).get("work_dir", "asr_run"))
    splits = load_prepared_splits(work_dir)
    processor = AutoProcessor.from_pretrained(spec.model_id)
    model = AutoModelForCTC.from_pretrained(
        spec.model_id,
        ctc_loss_reduction=cfg.get("model", {}).get("ctc_loss_reduction", "mean"),
        pad_token_id=getattr(processor.tokenizer, "pad_token_id", None),
    )
    if cfg.get("model", {}).get("freeze_feature_encoder", True) and hasattr(model, "freeze_feature_encoder"):
        model.freeze_feature_encoder()

    def to_ds(rows: List[Dict[str, Any]]) -> Any:
        ds = Dataset.from_list(rows)
        ds = ds.rename_column("audio_path", "audio")
        return ds.cast_column("audio", Audio(sampling_rate=int(cfg.get("model", {}).get("sample_rate", spec.sample_rate))))

    train_ds = to_ds(splits["train"])
    val_ds = to_ds(splits["validation"])
    metric_norm = cfg.get("evaluation", {}).get("metric_normalizer", "arabic_basic")

    def prepare(batch: Dict[str, Any]) -> Dict[str, Any]:
        audio = batch["audio"]
        batch["input_values"] = processor(audio["array"], sampling_rate=audio["sampling_rate"]).input_values[0]
        labels = processor(text=batch["text"]).input_ids
        batch["labels"] = labels
        return batch

    train_ds = train_ds.map(prepare, remove_columns=train_ds.column_names, desc="CTC preprocess train")
    val_ds = val_ds.map(prepare, remove_columns=val_ds.column_names, desc="CTC preprocess val")

    class DataCollatorCTCWithPadding:
        def __init__(self, processor: Any, padding: str = "longest"):
            self.processor = processor
            self.padding = padding

        def __call__(self, features: List[Dict[str, Any]]) -> Dict[str, Any]:
            input_features = [{"input_values": f["input_values"]} for f in features]
            label_features = [{"input_ids": f["labels"]} for f in features]
            batch = self.processor.pad(input_features, padding=self.padding, return_tensors="pt")
            labels_batch = self.processor.pad(labels=label_features, padding=self.padding, return_tensors="pt")
            labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
            batch["labels"] = labels
            return batch

    def compute_metrics(pred: Any) -> Dict[str, float]:
        pred_logits = pred.predictions
        pred_ids = np.argmax(pred_logits, axis=-1)
        pred_str = processor.batch_decode(pred_ids)
        label_ids = pred.label_ids
        label_ids[label_ids == -100] = processor.tokenizer.pad_token_id
        label_str = processor.batch_decode(label_ids, group_tokens=False)
        return compute_wer_cer(pred_str, label_str, metric_norm)

    training = cfg.get("training", {})
    args = {
        "output_dir": str(work_dir / "checkpoints"),
        "per_device_train_batch_size": int(training.get("per_device_train_batch_size", 8)),
        "per_device_eval_batch_size": int(training.get("per_device_eval_batch_size", 8)),
        "gradient_accumulation_steps": int(training.get("gradient_accumulation_steps", 2)),
        "learning_rate": float(training.get("learning_rate", 3e-4)),
        "warmup_ratio": float(training.get("warmup_ratio", 0.05)),
        "num_train_epochs": float(training.get("num_train_epochs", 10)),
        "fp16": bool(training.get("fp16", False)),
        "bf16": bool(training.get("bf16", True) and torch.cuda.is_available()),
        "save_strategy": "epoch",
        "logging_strategy": "steps",
        "logging_steps": int(training.get("logging_steps", 10)),
        "report_to": training.get("report_to", "none"),
        "load_best_model_at_end": True,
        "metric_for_best_model": "wer",
        "greater_is_better": False,
        "save_total_limit": int(training.get("save_total_limit", 3)),
    }
    train_args = _make_eval_strategy_args(TrainingArguments, args)
    trainer = Trainer(
        args=train_args,
        model=model,
        train_dataset=train_ds,
        eval_dataset=val_ds,
        data_collator=DataCollatorCTCWithPadding(processor),
        compute_metrics=compute_metrics,
        tokenizer=processor,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=int(training.get("early_stopping_patience", 5)))],
    )
    LOGGER.info("Starting HF CTC training. Best model selected by lowest WER.")
    result = trainer.train()
    trainer.save_model(str(work_dir / "best_model"))
    processor.save_pretrained(str(work_dir / "best_model"))
    metrics = trainer.evaluate()
    dump_json(work_dir / "train_result.json", {"train": str(result), "eval": metrics})
    return {"best_model": str(work_dir / "best_model"), "eval": metrics}


# -----------------------------
# Qwen / Cohere eval helpers
# -----------------------------


def eval_qwen_chat_asr(cfg: Dict[str, Any], spec: ModelSpec, split: str = "test") -> Dict[str, Any]:
    try:
        import torch
        from qwen_asr import Qwen3ASRModel  # type: ignore
    except Exception as e:
        raise RuntimeError("Install qwen-asr for Qwen3-ASR eval: pip install -U qwen-asr") from e
    work_dir = Path(cfg.get("output", {}).get("work_dir", "asr_run"))
    rows = load_prepared_splits(work_dir)[split]
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    model = Qwen3ASRModel.from_pretrained(
        spec.model_id,
        dtype=dtype,
        device_map=cfg.get("model", {}).get("device_map", "cuda:0" if torch.cuda.is_available() else "cpu"),
        max_new_tokens=int(cfg.get("evaluation", {}).get("max_new_tokens", 256)),
    )
    preds, refs = [], []
    language = cfg.get("model", {}).get("qwen_language", "Arabic")
    for i, r in enumerate(rows, start=1):
        LOGGER.info("Qwen eval %d/%d: %s", i, len(rows), r["uid"])
        out = model.transcribe(audio=r["audio_path"], language=language)
        text = out[0].text if out else ""
        preds.append(text)
        refs.append(r["text"])
    metrics = compute_wer_cer(preds, refs, cfg.get("evaluation", {}).get("metric_normalizer", "arabic_basic"))
    dump_json(work_dir / f"qwen_{split}_predictions.json", [{"pred": p, "ref": r} for p, r in zip(preds, refs)])
    LOGGER.info("Qwen %s metrics: %s", split, metrics)
    return {"split": split, **metrics, "n": len(rows)}


def eval_cohere_transcribe(cfg: Dict[str, Any], spec: ModelSpec, split: str = "test") -> Dict[str, Any]:
    try:
        from transformers import pipeline  # type: ignore
    except Exception as e:
        raise RuntimeError("Install transformers>=5.4.0 and audio deps for Cohere Transcribe eval.") from e
    work_dir = Path(cfg.get("output", {}).get("work_dir", "asr_run"))
    rows = load_prepared_splits(work_dir)[split]
    pipe = pipeline("automatic-speech-recognition", model=spec.model_id, trust_remote_code=True)
    preds, refs = [], []
    for i, r in enumerate(rows, start=1):
        LOGGER.info("Cohere eval %d/%d: %s", i, len(rows), r["uid"])
        out = pipe(r["audio_path"])
        preds.append(out.get("text", "") if isinstance(out, dict) else str(out))
        refs.append(r["text"])
    metrics = compute_wer_cer(preds, refs, cfg.get("evaluation", {}).get("metric_normalizer", "arabic_basic"))
    dump_json(work_dir / f"cohere_{split}_predictions.json", [{"pred": p, "ref": r} for p, r in zip(preds, refs)])
    LOGGER.info("Cohere %s metrics: %s", split, metrics)
    return {"split": split, **metrics, "n": len(rows)}


# -----------------------------
# NeMo runner/export
# -----------------------------


def write_nemo_run_script(cfg: Dict[str, Any], spec: ModelSpec) -> Path:
    work_dir = Path(cfg.get("output", {}).get("work_dir", "asr_run"))
    prepared = work_dir / "prepared" / "nemo_manifests"
    run_path = work_dir / "run_nemo_fastconformer.sh"
    epochs = int(cfg.get("training", {}).get("num_train_epochs", 10))
    lr = float(cfg.get("training", {}).get("learning_rate", 1e-4))
    patience = int(cfg.get("training", {}).get("early_stopping_patience", 5))
    script = f"""#!/usr/bin/env bash
set -euo pipefail
# Generated by asr_universal_trainer.py
# 1) Install NeMo in your GPU env, for example:
#    pip install 'nemo_toolkit[asr]'
# 2) Clone NeMo if you want to use their example scripts:
#    git clone https://github.com/NVIDIA/NeMo.git
# 3) Set NEMO_ROOT to the clone path:
: "${{NEMO_ROOT:=/path/to/NeMo}}"

python "$NEMO_ROOT/examples/asr/transcribe_speech.py" --help >/dev/null || true

# Fine-tuning command template. Adjust tokenizer settings if you train a new tokenizer.
python "$NEMO_ROOT/examples/asr/asr_hybrid_transducer_ctc/speech_to_text_hybrid_rnnt_ctc_bpe.py" \
  +init_from_pretrained_model="{spec.model_id}" \
  model.train_ds.manifest_filepath="{prepared / 'train.json'}" \
  model.validation_ds.manifest_filepath="{prepared / 'validation.json'}" \
  model.test_ds.manifest_filepath="{prepared / 'test.json'}" \
  trainer.max_epochs={epochs} \
  model.optim.lr={lr} \
  exp_manager.checkpoint_callback_params.monitor="val_wer" \
  exp_manager.checkpoint_callback_params.mode="min" \
  +trainer.callbacks.early_stop.monitor="val_wer" \
  +trainer.callbacks.early_stop.patience={patience}
"""
    run_path.write_text(script, encoding="utf-8")
    run_path.chmod(0o755)
    LOGGER.info("Wrote NeMo runner template: %s", run_path)
    return run_path


# -----------------------------
# Main orchestration
# -----------------------------


def create_smoke_dataset(base: str | Path) -> Path:
    base = Path(base)
    data_dir = base / "smoke_audio"
    rows = []
    examples = [
        ("train", "مرحبا هذا اختبار تدريب", 440.0),
        ("validation", "مرحبا هذا اختبار تحقق", 550.0),
        ("test", "مرحبا هذا اختبار نهائي", 660.0),
    ]
    for idx, (split, text, freq) in enumerate(examples):
        wav_path = data_dir / f"{split}_{idx}.wav"
        make_tiny_wav(wav_path, seconds=0.25, freq=freq)
        rows.append({"uid": f"smoke_{split}", "audio_path": str(wav_path), "text": text, "duration": 0.25, "split": split, "language": "ar"})
    jsonl = base / "smoke_dataset.jsonl"
    write_jsonl(jsonl, rows)
    return jsonl


def build_smoke_config(work_dir: str | Path) -> Dict[str, Any]:
    jsonl = create_smoke_dataset(Path(work_dir))
    return {
        "model": {"model_id": "mock", "sample_rate": 16000},
        "data": {
            "format": "jsonl",
            "path": str(jsonl),
            "columns": {"audio": "audio_path", "text": "text", "duration": "duration", "split": "split", "uid": "uid"},
            "language": "ar",
            "min_seconds": 0.01,
            "long_audio_policy": "drop",
        },
        "training": {"num_train_epochs": 1, "early_stopping_patience": 5, "logging_steps": 1},
        "evaluation": {"metric_normalizer": "arabic_basic"},
        "output": {"work_dir": str(work_dir)},
    }


def load_run_result(work_dir: str | Path) -> Dict[str, Any]:
    path = Path(work_dir) / "run_result.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        LOGGER.warning("Could not parse existing run result file: %s", path)
        return {}


def save_run_result(work_dir: str | Path, stage: str, results: Dict[str, Any]) -> Dict[str, Any]:
    work_dir = Path(work_dir)
    merged = load_run_result(work_dir)
    merged.update(results)
    merged["last_stage"] = stage
    history = list(merged.get("stage_history", []))
    history.append({"stage": stage, "timestamp": time.time()})
    merged["stage_history"] = history
    dump_json(work_dir / "run_result.json", merged)
    return merged


def run_stage(cfg: Dict[str, Any], stage: str) -> Dict[str, Any]:
    model_conf = cfg.get("model", {})
    spec = infer_model_spec(model_conf.get("model_id", "mock"), model_conf.get("spec_overrides", {}))
    setup_logging(cfg.get("output", {}).get("work_dir", "asr_run"), cfg.get("output", {}).get("log_level", "INFO"))
    LOGGER.info("Model spec: %s", dataclasses.asdict(spec))
    results: Dict[str, Any] = load_run_result(cfg.get("output", {}).get("work_dir", "asr_run"))
    results["model_spec"] = dataclasses.asdict(spec)

    if stage in {"prepare", "all", "train", "eval"}:
        # prepare if prepared files are missing, or explicitly prepare/all.
        prepared_train = Path(cfg.get("output", {}).get("work_dir", "asr_run")) / "prepared" / "train.jsonl"
        if stage in {"prepare", "all"} or not prepared_train.exists():
            results["prepared_paths"] = prepare_dataset(cfg, spec)
            if spec.backend == "nemo_fastconformer":
                results["nemo_runner"] = str(write_nemo_run_script(cfg, spec))
            elif spec.backend == "omnilingual_recipe":
                results["omnilingual_recipe"] = prepare_omnilingual_recipe_dataset(cfg, spec.model_id)
        if stage == "prepare":
            return save_run_result(cfg.get("output", {}).get("work_dir", "asr_run"), stage, results)

    if stage in {"train", "all"}:
        if spec.backend == "mock":
            results["train"] = MockTrainer(cfg, spec).train()
        elif spec.backend == "hf_whisper_seq2seq":
            results["train"] = train_hf_whisper_seq2seq(cfg, spec)
        elif spec.backend == "hf_ctc":
            results["train"] = train_hf_ctc(cfg, spec)
        elif spec.backend == "omnilingual_recipe":
            results["train"] = train_omnilingual(cfg, spec.model_id)
        elif spec.backend == "nemo_fastconformer":
            runner = write_nemo_run_script(cfg, spec)
            LOGGER.info("NeMo backend selected. Dataset manifests are ready. Run: bash %s", runner)
            results["train"] = {"status": "external_nemo_command_written", "runner": str(runner)}
        else:
            LOGGER.warning("Training is not implemented safely for backend=%s. %s", spec.backend, spec.notes)
            results["train"] = {"status": "not_supported_in_this_script", "reason": spec.notes}

    if stage in {"eval", "all"}:
        split = cfg.get("evaluation", {}).get("split", "test")
        if spec.backend == "mock":
            results["eval"] = MockTrainer(cfg, spec).evaluate(split)
        elif spec.backend == "qwen_chat_asr":
            results["eval"] = eval_qwen_chat_asr(cfg, spec, split)
        elif spec.backend == "cohere_eval_only":
            results["eval"] = eval_cohere_transcribe(cfg, spec, split)
        elif spec.backend == "omnilingual_recipe":
            results["eval"] = eval_omnilingual(cfg, spec.model_id, split)
        elif spec.backend in {"hf_whisper_seq2seq", "hf_ctc"} and stage == "eval":
            LOGGER.info("For HF eval-only, use Trainer prediction from best_model or run stage=all after training. Placeholder complete.")
            results["eval"] = {"status": "hf_eval_after_training_available_in_train_stage"}
        elif spec.backend == "nemo_fastconformer":
            runner = write_nemo_run_script(cfg, spec)
            results["eval"] = {"status": "external_nemo_command_written", "runner": str(runner)}
        else:
            results["eval"] = {"status": "not_supported_in_this_script", "reason": spec.notes}

    return save_run_result(cfg.get("output", {}).get("work_dir", "asr_run"), stage, results)


def main() -> None:
    parser = argparse.ArgumentParser(description="Universal ASR plug-and-play trainer/evaluator")
    parser.add_argument("--config", type=str, default=None, help="YAML/JSON config path")
    parser.add_argument("--stage", type=str, default="all", choices=["prepare", "train", "eval", "all"], help="Pipeline stage")
    parser.add_argument("--smoke-test", action="store_true", help="Create a 1 train / 1 val / 1 test mock run and execute it")
    parser.add_argument("--work-dir", type=str, default="/mnt/data/asr_smoke_run", help="Work dir for smoke-test if --config omitted")
    args = parser.parse_args()

    if args.smoke_test or not args.config:
        cfg = build_smoke_config(args.work_dir)
    else:
        cfg = load_config(args.config)

    try:
        result = run_stage(cfg, args.stage)
        print(json.dumps(result, ensure_ascii=False, indent=2))
    except Exception as e:
        LOGGER.error("Pipeline failed: %s", e)
        LOGGER.error(traceback.format_exc())
        raise


if __name__ == "__main__":
    main()
