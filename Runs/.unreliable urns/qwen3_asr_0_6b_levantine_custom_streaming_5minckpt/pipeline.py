"""Shared helpers for the Qwen3-ASR 0.6B Levantine custom run notebook."""

from __future__ import annotations

import collections
import contextlib
import hashlib
import inspect
import io
import json
import logging
import math
import statistics
import time
import unicodedata
import re
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


REPO_ROOT = Path("/home/MohammadNabulsi/whisper")
RUN_DIR = REPO_ROOT / "Runs" / "qwen3_asr_0_6b_levantine_custom_streaming_5minckpt"
NOTEBOOK_PATH = RUN_DIR / "qwen3_asr_0_6b_lora_levantine_custom_streaming_5minckpt_run.ipynb"
SMOKE_NOTEBOOK_PATH = RUN_DIR / "qwen3_asr_0_6b_lora_levantine_custom_streaming_5minckpt_smoketest.ipynb"
EXECUTED_SMOKE_NOTEBOOK_PATH = RUN_DIR / "qwen3_asr_0_6b_lora_levantine_custom_streaming_5minckpt_smoketest_executed.ipynb"
LOG_DIR = RUN_DIR / "logs"
OUTPUT_DIR = RUN_DIR / "checkpoints"
BEST_MODEL_DIR = RUN_DIR / "best"
FINAL_MODEL_DIR = RUN_DIR / "final_adapter"
MANIFEST_DIR = RUN_DIR / "manifests"
PREDICTION_DIR = RUN_DIR / "eval_predictions"

TRAIN_PARQUET_FILES = (
    REPO_ROOT / "casablanca" / "levant" / "Palestine" / "validation-00001-of-00002.parquet",
    REPO_ROOT / "casablanca" / "levant" / "Palestine" / "validation-00000-of-00002.parquet",
    REPO_ROOT / "casablanca" / "levant" / "Jordan" / "validation-00000-of-00001.parquet",
)
EVAL_PARQUET_FILES = (
    REPO_ROOT / "casablanca" / "levant" / "Palestine" / "test-00001-of-00002.parquet",
    REPO_ROOT / "casablanca" / "levant" / "Palestine" / "test-00000-of-00002.parquet",
    REPO_ROOT / "casablanca" / "levant" / "Jordan" / "test-00000-of-00001.parquet",
)
TRAIN_ARROW_FILES = (
    REPO_ROOT / "omnilingual_selected" / "apc_north_levantine_all_splits" / "data-00001-of-00003.arrow",
    REPO_ROOT / "omnilingual_selected" / "apc_north_levantine_all_splits" / "data-00000-of-00003.arrow",
)
EVAL_ARROW_FILES = (
    REPO_ROOT / "omnilingual_selected" / "apc_north_levantine_all_splits" / "data-00002-of-00003.arrow",
)

LOCAL_QWEN_CACHE_DIR = Path("/home/MohammadNabulsi/.cache/huggingface/hub/models--Qwen--Qwen3-ASR-0.6B/snapshots/5eb144179a02acc5e5ba31e748d22b0cf3e303b0")

AR_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
CONTROL_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
SPACE_RE = re.compile(r"\s+")
ASR_TAG_RE = re.compile(r"<[^>]+>|\[[^\]]+\]")
AR_PUNCT_SPACING_RE = re.compile(r"\s*([،؛؟,.!?;:])\s*")
REPEATED_PUNCT_RE = re.compile(r"([،؛؟,.!?;:])\1+")
PUNCT_RE = re.compile(r"[\.,!\?;:،؛؟\-\(\)\[\]{}\"'“”‘’«»]")

LOGGER = logging.getLogger("qwen3_asr_levantine_run")


@dataclass
class RunConfig:
    model_name: str = "Qwen/Qwen3-ASR-0.6B"
    run_id: str = "qwen3_asr_0_6b_levantine_custom_streaming_5minckpt"
    sample_rate: int = 16_000
    min_audio_seconds: float = 0.3
    drop_audio_at_or_above_seconds: float | None = None
    train_batch_size: int = 4
    eval_batch_size: int = 16
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_train_epochs: int = 50
    logging_steps: int = 0
    checkpoint_every_minutes: int = 5
    save_total_limit: int = 6
    generation_max_new_tokens: int = 256
    split_seed: int = 42
    train_dataloader_num_workers: int = 4
    language: str = "Arabic"
    smoke_mode: bool = False
    run_baseline_before_train: bool = True
    run_post_train_eval: bool = True
    force_rebuild_manifests: bool = False
    resume_from_checkpoint: bool = False
    use_bf16: bool = True
    use_fp16: bool = False
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    qwen_system_prompt: str = "Transcribe this audio into Arabic."


def make_config(
    *,
    smoke_mode: bool = False,
    num_train_epochs: int = 50,
    run_baseline_before_train: bool = True,
    run_post_train_eval: bool = True,
) -> RunConfig:
    config = RunConfig(
        smoke_mode=smoke_mode,
        num_train_epochs=num_train_epochs,
        run_baseline_before_train=run_baseline_before_train,
        run_post_train_eval=run_post_train_eval,
    )
    if smoke_mode:
        config.train_batch_size = 1
        config.eval_batch_size = 1
        config.gradient_accumulation_steps = 1
        config.train_dataloader_num_workers = 0
    return config


def ensure_run_layout() -> None:
    for path in [RUN_DIR, LOG_DIR, OUTPUT_DIR, BEST_MODEL_DIR, FINAL_MODEL_DIR, MANIFEST_DIR, PREDICTION_DIR]:
        path.mkdir(parents=True, exist_ok=True)
    for split in ["train", "val", "test"]:
        (MANIFEST_DIR / split).mkdir(parents=True, exist_ok=True)


def setup_logging() -> Path:
    ensure_run_layout()
    log_path = LOG_DIR / "qwen3_asr_levantine_run.log"
    if not LOGGER.handlers:
        LOGGER.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)
        LOGGER.addHandler(stream_handler)
        LOGGER.propagate = False
    return log_path


def resolve_model_source(config: RunConfig) -> str:
    if LOCAL_QWEN_CACHE_DIR.exists() and (LOCAL_QWEN_CACHE_DIR / "config.json").exists():
        return str(LOCAL_QWEN_CACHE_DIR)
    return config.model_name


def config_snapshot(config: RunConfig) -> dict[str, Any]:
    data = asdict(config)
    data.update(
        {
            "run_dir": str(RUN_DIR),
            "notebook_path": str(NOTEBOOK_PATH),
            "smoke_notebook_path": str(SMOKE_NOTEBOOK_PATH),
            "output_dir": str(OUTPUT_DIR),
            "best_model_dir": str(BEST_MODEL_DIR),
            "final_model_dir": str(FINAL_MODEL_DIR),
            "manifest_dir": str(MANIFEST_DIR),
            "prediction_dir": str(PREDICTION_DIR),
            "resolved_model_source": resolve_model_source(config),
            "train_parquet_files": [str(path) for path in TRAIN_PARQUET_FILES],
            "eval_parquet_files": [str(path) for path in EVAL_PARQUET_FILES],
            "train_arrow_files": [str(path) for path in TRAIN_ARROW_FILES],
            "eval_arrow_files": [str(path) for path in EVAL_ARROW_FILES],
        }
    )
    return data


def jsonl_read(path: Path) -> list[dict[str, Any]]:
    with path.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def jsonl_write(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")


def normalize_arabic_text(
    text: Any,
    *,
    remove_punctuation: bool = False,
    normalize_alef: bool = False,
    normalize_yaa: bool = False,
    normalize_taa_marbuta: bool = False,
) -> str:
    if text is None:
        return ""
    text = unicodedata.normalize("NFKC", str(text))
    text = CONTROL_RE.sub(" ", text)
    text = ASR_TAG_RE.sub(" ", text)
    text = text.replace("ـ", "")
    text = AR_DIACRITICS_RE.sub("", text)
    if normalize_alef:
        text = re.sub("[إأآٱ]", "ا", text)
    if normalize_yaa:
        text = text.replace("ى", "ي")
    if normalize_taa_marbuta:
        text = text.replace("ة", "ه")
    text = AR_PUNCT_SPACING_RE.sub(r" \1 ", text)
    text = REPEATED_PUNCT_RE.sub(r"\1", text)
    if remove_punctuation:
        text = PUNCT_RE.sub(" ", text)
    text = SPACE_RE.sub(" ", text).strip()
    return text


def normalize_metric_text(text: Any, *, loose: bool = False, punctuation_insensitive: bool = False) -> str:
    normalized = normalize_arabic_text(text, remove_punctuation=punctuation_insensitive)
    if loose:
        normalized = re.sub("[إأآٱ]", "ا", normalized)
        normalized = normalized.replace("ى", "ي")
    return normalized


def char_error_rate(reference: str, prediction: str, *, loose: bool = False, punctuation_insensitive: bool = False) -> float:
    ref = list(normalize_metric_text(reference, loose=loose, punctuation_insensitive=punctuation_insensitive))
    pred = list(normalize_metric_text(prediction, loose=loose, punctuation_insensitive=punctuation_insensitive))
    return _error_rate(ref, pred)


def word_error_rate(reference: str, prediction: str, *, loose: bool = False, punctuation_insensitive: bool = False) -> float:
    ref = normalize_metric_text(reference, loose=loose, punctuation_insensitive=punctuation_insensitive).split()
    pred = normalize_metric_text(prediction, loose=loose, punctuation_insensitive=punctuation_insensitive).split()
    return _error_rate(ref, pred)


def _error_rate(reference: list[str], prediction: list[str]) -> float:
    if not reference:
        return 0.0 if not prediction else 1.0
    return _levenshtein(reference, prediction) / float(len(reference))


def _levenshtein(reference: list[str], prediction: list[str]) -> int:
    previous = list(range(len(prediction) + 1))
    for i, ref_token in enumerate(reference, start=1):
        current = [i]
        for j, pred_token in enumerate(prediction, start=1):
            insertion = current[j - 1] + 1
            deletion = previous[j] + 1
            substitution = previous[j - 1] + (ref_token != pred_token)
            current.append(min(insertion, deletion, substitution))
        previous = current
    return previous[-1]


def valid_duration(duration: Any, config: RunConfig) -> bool:
    if duration is None:
        return False
    try:
        seconds = float(duration)
    except Exception:
        return False
    if seconds < config.min_audio_seconds:
        return False
    if config.drop_audio_at_or_above_seconds is not None and seconds >= config.drop_audio_at_or_above_seconds:
        return False
    return True


def _stable_hash(text: str, seed: int) -> int:
    payload = f"{seed}|{text}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16)


def _stable_row_order(rows: list[dict[str, Any]], *, seed: int, salt: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _stable_hash(f"{salt}|{row.get('uid')}", seed))


def _read_and_filter_rows(rows: list[dict[str, Any]], split: str, config: RunConfig) -> list[dict[str, Any]]:
    filtered: list[dict[str, Any]] = []
    dropped_long = 0
    dropped_empty = 0
    for row in rows:
        text = normalize_arabic_text(row.get("text", ""))
        if not text:
            dropped_empty += 1
            continue
        if not valid_duration(row.get("duration"), config):
            dropped_long += 1
            continue
        clean = dict(row)
        clean["text"] = text
        clean["split"] = split
        filtered.append(clean)
    LOGGER.info(
        "prepared split=%s rows=%d kept=%d dropped_empty=%d dropped_duration=%d",
        split,
        len(rows),
        len(filtered),
        dropped_empty,
        dropped_long,
    )
    return filtered


def _row_uid(prefix: str, path: Path, row_idx: int) -> str:
    digest = hashlib.sha1(f"{prefix}:{path}:{row_idx}".encode("utf-8")).hexdigest()
    return f"{prefix}-{digest[:16]}"


def _ensure_audio_tools() -> tuple[Any, Any, Any, Any, Any]:
    try:
        import soundfile as sf
    except ImportError:
        sf = None
    try:
        import librosa
    except ImportError:
        librosa = None
    try:
        import pyarrow as pa
        import pyarrow.ipc as pa_ipc
        import pyarrow.parquet as pq
    except ImportError as exc:
        raise RuntimeError("pyarrow is required for the shared manifest audio loader.") from exc
    if sf is None and librosa is None:
        raise RuntimeError("Install soundfile or librosa for audio decoding.")
    return sf, librosa, pa, pa_ipc, pq


def _build_parquet_rows(paths: tuple[Path, ...], *, original_split: str, source_group: str, source_label: str) -> list[dict[str, Any]]:
    _, _, _, _, pq = _ensure_audio_tools()
    rows: list[dict[str, Any]] = []
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        row_idx = 0
        for batch in parquet_file.iter_batches(batch_size=1024, columns=["seg_id", "transcription", "gender", "duration"]):
            for item in batch.to_pylist():
                rows.append(
                    {
                        "uid": _row_uid("parquet", path, row_idx),
                        "source": "casablanca",
                        "source_group": source_group,
                        "source_root": str(path.parent),
                        "original_split": original_split,
                        "parquet_file": str(path),
                        "arrow_file": None,
                        "row_idx": row_idx,
                        "audio_kind": "bytes",
                        "audio_path": None,
                        "audio_bytes_ref": None,
                        "text": item.get("transcription", ""),
                        "duration": item.get("duration"),
                        "segment_id": item.get("seg_id"),
                        "speaker_id": None,
                        "gender": item.get("gender"),
                        "language": "Arabic",
                        "start": None,
                        "end": None,
                        "metadata": {
                            "source_label": source_label,
                            "country": path.parent.name,
                            "filename": path.name,
                        },
                    }
                )
                row_idx += 1
    return rows


def _build_arrow_rows(paths: tuple[Path, ...], *, original_split: str, source_group: str) -> list[dict[str, Any]]:
    _, _, pa, pa_ipc, _ = _ensure_audio_tools()
    rows: list[dict[str, Any]] = []
    columns = [
        "language",
        "speaker_id",
        "prompt_id",
        "prompt",
        "segment_id",
        "duration",
        "raw_text",
        "iso_639_3",
        "glottocode",
        "iso_15924",
        "config",
        "original_split",
    ]
    for path in paths:
        row_idx = 0
        with pa.memory_map(str(path), "r") as source:
            reader = pa_ipc.open_stream(source)
            for batch in reader:
                names = set(batch.schema.names)
                use_cols = [column for column in columns if column in names]
                payload = {column: batch.column(batch.schema.get_field_index(column)).to_pylist() for column in use_cols}
                for offset in range(batch.num_rows):
                    lang = payload.get("language", ["Arabic"])[offset] or "Arabic"
                    metadata = {
                        "config": payload.get("config", [None])[offset],
                        "glottocode": payload.get("glottocode", [None])[offset],
                        "iso_639_3": payload.get("iso_639_3", [None])[offset],
                        "iso_15924": payload.get("iso_15924", [None])[offset],
                        "prompt": payload.get("prompt", [None])[offset],
                        "prompt_id": payload.get("prompt_id", [None])[offset],
                        "filename": path.name,
                    }
                    rows.append(
                        {
                            "uid": _row_uid("arrow", path, row_idx),
                            "source": "omnilingual",
                            "source_group": source_group,
                            "source_root": str(path.parent),
                            "original_split": payload.get("original_split", [original_split])[offset] or original_split,
                            "parquet_file": None,
                            "arrow_file": str(path),
                            "row_idx": row_idx,
                            "audio_kind": "hf_audio",
                            "audio_path": None,
                            "audio_bytes_ref": None,
                            "text": payload.get("raw_text", [""])[offset],
                            "duration": payload.get("duration", [None])[offset],
                            "segment_id": payload.get("segment_id", [None])[offset],
                            "speaker_id": payload.get("speaker_id", [None])[offset],
                            "gender": None,
                            "language": lang,
                            "start": None,
                            "end": None,
                            "metadata": metadata,
                        }
                    )
                    row_idx += 1
    return rows


def _split_rows_half(rows: list[dict[str, Any]], *, seed: int, salt: str) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    ordered = _stable_row_order(rows, seed=seed, salt=salt)
    midpoint = len(ordered) // 2
    val_rows = [dict(row, split="val") for row in ordered[:midpoint]]
    test_rows = [dict(row, split="test") for row in ordered[midpoint:]]
    return val_rows, test_rows


def _rows_by_source_group(rows: list[dict[str, Any]]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    groups = sorted({str(row.get("source_group")) for row in rows})
    for group in groups:
        group_rows = [row for row in rows if str(row.get("source_group")) == group]
        out[group] = {
            "rows": len(group_rows),
            "hours": sum(float(row.get("duration") or 0.0) for row in group_rows) / 3600.0,
        }
    return out


def prepare_manifests(config: RunConfig) -> dict[str, Any]:
    ensure_run_layout()

    train_rows_raw = _build_parquet_rows(
        TRAIN_PARQUET_FILES,
        original_split="validation",
        source_group="casablanca_levantine_train",
        source_label="casablanca_levantine_validation_as_train",
    ) + _build_arrow_rows(
        TRAIN_ARROW_FILES,
        original_split="train",
        source_group="omnilingual_apc_north_levantine_train",
    )

    eval_parquet_raw = _build_parquet_rows(
        EVAL_PARQUET_FILES,
        original_split="test",
        source_group="casablanca_levantine_eval_pool",
        source_label="casablanca_levantine_test_split_50_50",
    )
    eval_arrow_raw = _build_arrow_rows(
        EVAL_ARROW_FILES,
        original_split="train",
        source_group="omnilingual_apc_north_levantine_eval_pool",
    )

    filtered_train = _read_and_filter_rows(train_rows_raw, "train", config)
    filtered_eval_parquet = _read_and_filter_rows(eval_parquet_raw, "eval_pool", config)
    filtered_eval_arrow = _read_and_filter_rows(eval_arrow_raw, "eval_pool", config)

    parquet_val_rows, parquet_test_rows = _split_rows_half(filtered_eval_parquet, seed=config.split_seed, salt="custom-casablanca-heldout")
    arrow_val_rows, arrow_test_rows = _split_rows_half(filtered_eval_arrow, seed=config.split_seed, salt="custom-omnilingual-heldout")

    train_rows = [dict(row, split="train") for row in _stable_row_order(filtered_train, seed=config.split_seed, salt="custom-train")]
    val_rows = _stable_row_order(parquet_val_rows + arrow_val_rows, seed=config.split_seed, salt="custom-val")
    test_rows = _stable_row_order(parquet_test_rows + arrow_test_rows, seed=config.split_seed, salt="custom-test")

    if config.smoke_mode:
        train_rows = train_rows[:1]
        val_rows = val_rows[:1]
        test_rows = test_rows[:1]

    train_path = MANIFEST_DIR / "train" / "manifest_train_custom_levantine.jsonl"
    val_path = MANIFEST_DIR / "val" / "manifest_val_custom_levantine.jsonl"
    test_path = MANIFEST_DIR / "test" / "manifest_test_custom_levantine.jsonl"
    all_path = MANIFEST_DIR / "manifest_all.jsonl"
    jsonl_write(train_path, train_rows)
    jsonl_write(val_path, val_rows)
    jsonl_write(test_path, test_rows)
    jsonl_write(all_path, train_rows + val_rows + test_rows)

    selection_summary = {
        "run_id": config.run_id,
        "drop_segments_at_or_above_seconds": config.drop_audio_at_or_above_seconds,
        "full_counts_after_filter": {
            "train": len(filtered_train),
            "eval_parquet_pool": len(filtered_eval_parquet),
            "eval_arrow_pool": len(filtered_eval_arrow),
        },
        "selected_counts": {
            "train": len(train_rows),
            "val": len(val_rows),
            "test": len(test_rows),
        },
        "selected_hours": {
            "train": sum(float(row.get("duration") or 0.0) for row in train_rows) / 3600.0,
            "val": sum(float(row.get("duration") or 0.0) for row in val_rows) / 3600.0,
            "test": sum(float(row.get("duration") or 0.0) for row in test_rows) / 3600.0,
        },
        "train_by_source_group": _rows_by_source_group(train_rows),
        "val_by_source_group": _rows_by_source_group(val_rows),
        "test_by_source_group": _rows_by_source_group(test_rows),
        "data_sources": {
            "train_parquet_files": [str(path) for path in TRAIN_PARQUET_FILES],
            "eval_parquet_files": [str(path) for path in EVAL_PARQUET_FILES],
            "train_arrow_files": [str(path) for path in TRAIN_ARROW_FILES],
            "eval_arrow_files": [str(path) for path in EVAL_ARROW_FILES],
        },
        "manifest_paths": {
            "train": str(train_path),
            "val": str(val_path),
            "test": str(test_path),
            "all": str(all_path),
        },
    }

    summary_path = MANIFEST_DIR / "custom_selection_summary.json"
    summary_path.write_text(json.dumps(selection_summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return {
        "train_rows": train_rows,
        "val_rows": val_rows,
        "test_rows": test_rows,
        "train_manifest_path": train_path,
        "val_manifest_path": val_path,
        "test_manifest_path": test_path,
        "selection_summary": selection_summary,
    }


def _to_mono_float32(audio: Any) -> np.ndarray:
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


def _read_audio_bytes(payload: bytes) -> tuple[np.ndarray, int]:
    sf, librosa, *_ = _ensure_audio_tools()
    if sf is not None:
        audio, sample_rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
        return _to_mono_float32(audio), int(sample_rate)
    import tempfile
    with tempfile.NamedTemporaryFile(suffix=".wav") as handle:
        handle.write(payload)
        handle.flush()
        audio, sample_rate = librosa.load(handle.name, sr=None, mono=True)
    return _to_mono_float32(audio), int(sample_rate)


def _read_audio_path(path: str) -> tuple[np.ndarray, int]:
    sf, librosa, *_ = _ensure_audio_tools()
    if sf is not None:
        audio, sample_rate = sf.read(path, dtype="float32", always_2d=False)
        return _to_mono_float32(audio), int(sample_rate)
    audio, sample_rate = librosa.load(path, sr=None, mono=True)
    return _to_mono_float32(audio), int(sample_rate)


def _resample_audio(audio: np.ndarray, sample_rate: int, target_sample_rate: int) -> np.ndarray:
    if sample_rate == target_sample_rate:
        return _to_mono_float32(audio)
    _, librosa, *_ = _ensure_audio_tools()
    if librosa is None:
        raise RuntimeError("librosa is required for resampling.")
    return librosa.resample(_to_mono_float32(audio), orig_sr=int(sample_rate), target_sr=int(target_sample_rate)).astype(np.float32)


def _read_arrow_row(path: str, row_idx: int, columns: Optional[list[str]] = None) -> dict[str, Any]:
    _, _, pa, pa_ipc, _ = _ensure_audio_tools()
    index = int(row_idx)
    seen = 0
    with pa.memory_map(path, "r") as source:
        reader = pa_ipc.open_stream(source)
        for batch in reader:
            if seen + batch.num_rows <= index:
                seen += batch.num_rows
                continue
            local = index - seen
            names = set(batch.schema.names)
            use_cols = [column for column in (columns or batch.schema.names) if column in names]
            return {column: batch.column(batch.schema.get_field_index(column)).to_pylist()[local] for column in use_cols}
    raise IndexError(f"row_idx={row_idx} out of range for {path}")


def _load_arrow_audio(row: dict[str, Any]) -> tuple[np.ndarray, int]:
    example = _read_arrow_row(str(row["arrow_file"]), int(row["row_idx"]), columns=["audio"])
    audio = example.get("audio")
    if isinstance(audio, dict):
        if audio.get("array") is not None:
            sample_rate = audio.get("sampling_rate") or 16_000
            return _to_mono_float32(audio["array"]), int(sample_rate)
        if audio.get("bytes") is not None:
            return _read_audio_bytes(audio["bytes"])
        if audio.get("path"):
            candidate = str(audio["path"])
            if not Path(candidate).is_absolute():
                candidate = str(Path(str(row["arrow_file"])).parent / candidate)
            return _read_audio_path(candidate)
    raise RuntimeError(f"Unsupported Arrow audio payload for uid={row.get('uid')}")


def _read_parquet_row(path: str, row_idx: int) -> dict[str, Any]:
    *_, pq = _ensure_audio_tools()
    parquet_file = pq.ParquetFile(path)
    seen = 0
    for batch in parquet_file.iter_batches(batch_size=1024):
        py_rows = batch.to_pylist()
        if seen + len(py_rows) > row_idx:
            return py_rows[row_idx - seen]
        seen += len(py_rows)
    raise IndexError(f"row_idx={row_idx} out of range for {path}")


def _load_parquet_audio(row: dict[str, Any]) -> tuple[np.ndarray, int]:
    example = _read_parquet_row(str(row["parquet_file"]), int(row["row_idx"]))
    audio = example.get("audio") or {}
    if isinstance(audio, dict):
        if audio.get("bytes") is not None:
            return _read_audio_bytes(audio["bytes"])
        if audio.get("path"):
            candidate = str(audio["path"])
            if not Path(candidate).is_absolute():
                candidate = str(Path(str(row["parquet_file"])).parent / candidate)
            return _read_audio_path(candidate)
    raise RuntimeError(f"Unsupported Parquet audio payload for uid={row.get('uid')}")


def load_audio_for_row(row: dict[str, Any], config: RunConfig) -> tuple[np.ndarray, int]:
    if row.get("audio_kind") == "hf_audio":
        audio, sample_rate = _load_arrow_audio(row)
    elif row.get("audio_kind") == "bytes":
        audio, sample_rate = _load_parquet_audio(row)
    elif row.get("audio_kind") == "path" and row.get("audio_path"):
        audio, sample_rate = _read_audio_path(str(row["audio_path"]))
    else:
        raise RuntimeError(f"Unsupported audio_kind={row.get('audio_kind')} uid={row.get('uid')}")
    return _resample_audio(audio, int(sample_rate), config.sample_rate), config.sample_rate


@dataclass
class QwenBundle:
    model: Any
    processor: Any
    backend: str
    asr_wrapper: Any = None
    adapter_path: str | None = None


class SmokeProcessor:
    tokenizer = None
    feature_extractor = None


class SmokeQwenModel:
    def predict(self, reference: str) -> str:
        return reference


def _import_torch_and_training() -> tuple[Any, Any, Any, Any, Any, Any, Any, Any, Any]:
    import torch
    from torch.utils.data import DataLoader, Dataset
    from transformers import GenerationConfig, Trainer, TrainerCallback, TrainingArguments
    from peft import LoraConfig, PeftModel, TaskType, get_peft_model
    return torch, DataLoader, Dataset, GenerationConfig, Trainer, TrainerCallback, TrainingArguments, LoraConfig, PeftModel, TaskType, get_peft_model


def patch_outer_forward(model: Any) -> None:
    cls = model.__class__
    if getattr(cls, "_forward_patched", False):
        return
    if not hasattr(model, "thinker") or not hasattr(model.thinker, "forward"):
        raise RuntimeError("Cannot patch forward: Qwen3-ASR model has no .thinker.forward.")

    def forward(self, input_ids=None, attention_mask=None, input_features=None, feature_attention_mask=None, labels=None, **kwargs):
        return self.thinker.forward(
            input_ids=input_ids,
            attention_mask=attention_mask,
            input_features=input_features,
            feature_attention_mask=feature_attention_mask,
            labels=labels,
            **kwargs,
        )

    cls.forward = forward
    cls._forward_patched = True


def load_qwen_bundle(config: RunConfig, *, adapter_path: str | Path | None = None) -> QwenBundle:
    if config.smoke_mode:
        return QwenBundle(model=SmokeQwenModel(), processor=SmokeProcessor(), backend="smoke")

    import torch
    from transformers import GenerationConfig
    from peft import PeftModel
    from qwen_asr import Qwen3ASRModel

    source = resolve_model_source(config)
    LOGGER.info("Loading Qwen3-ASR from %s", source)
    model_dtype = torch.bfloat16 if torch.cuda.is_available() and config.use_bf16 and torch.cuda.is_bf16_supported() else (torch.float16 if torch.cuda.is_available() and config.use_fp16 else torch.float32)
    asr_wrapper = Qwen3ASRModel.from_pretrained(
        source,
        dtype=model_dtype,
        device_map=None,
        max_new_tokens=config.generation_max_new_tokens,
        max_inference_batch_size=max(1, config.eval_batch_size),
    )
    model = asr_wrapper.model
    processor = asr_wrapper.processor
    patch_outer_forward(model)
    model.generation_config = GenerationConfig.from_model_config(model.config)
    if adapter_path is not None:
        model = PeftModel.from_pretrained(model, str(adapter_path), is_trainable=False)
    if torch.cuda.is_available():
        model.to("cuda")
    model.eval()
    LOGGER.info("model=%s processor=%s adapter=%s", model.__class__.__name__, processor.__class__.__name__, adapter_path)
    return QwenBundle(model=model, processor=processor, backend="qwen_asr", asr_wrapper=asr_wrapper, adapter_path=str(adapter_path) if adapter_path else None)


def infer_lora_target_modules(model: Any) -> list[str]:
    import torch

    preferred = ["q_proj", "k_proj", "v_proj", "o_proj"]
    present = set()
    linear_suffixes = collections.Counter()
    for name, module in model.named_modules():
        leaf = name.split(".")[-1]
        if isinstance(module, torch.nn.Linear):
            linear_suffixes[leaf] += 1
            if leaf in preferred:
                present.add(leaf)
    if present:
        return [name for name in preferred if name in present]
    fallback = [name for name, _count in linear_suffixes.most_common() if any(tok in name for tok in ["proj", "linear", "fc"])]
    if fallback:
        return fallback[:8]
    raise RuntimeError("Could not infer LoRA target modules for Qwen3-ASR.")


def attach_lora(config: RunConfig, model: Any) -> tuple[Any, Any]:
    if config.smoke_mode:
        return model, None
    from peft import LoraConfig, TaskType, get_peft_model

    lora_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=infer_lora_target_modules(model),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type=TaskType.CAUSAL_LM,
    )
    peft_model = get_peft_model(model, lora_config)
    if hasattr(peft_model, "enable_input_require_grads"):
        peft_model.enable_input_require_grads()
    return peft_model, lora_config


def qwen_language_prefix(row: dict[str, Any], config: RunConfig) -> str:
    lang = row.get("language") or config.language or "Arabic"
    if str(lang).lower() in {"ar", "ara", "arabic", "apc_arab"}:
        lang = "Arabic"
    return f"language {lang}<asr_text>"


def build_prefix_messages(config: RunConfig, row: dict[str, Any], audio_array: Any) -> list[dict[str, Any]]:
    prompt = row.get("prompt") or row.get("metadata", {}).get("prompt") or config.qwen_system_prompt
    return [
        {"role": "system", "content": prompt},
        {"role": "user", "content": [{"type": "audio", "audio": audio_array}]},
    ]


def build_prefix_text(processor: Any, config: RunConfig, row: dict[str, Any]) -> str:
    messages = build_prefix_messages(config, row, None)
    try:
        templated = processor.apply_chat_template([messages], add_generation_prompt=True, tokenize=False)
        return templated[0] if isinstance(templated, list) else templated
    except TypeError:
        return processor.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)


def _squeeze_batch_dim(batch: dict[str, Any]) -> dict[str, Any]:
    import torch

    out = {}
    for key, value in batch.items():
        if torch.is_tensor(value) and value.ndim > 0 and value.shape[0] == 1:
            out[key] = value[0].detach().cpu()
        else:
            out[key] = value.detach().cpu() if torch.is_tensor(value) else value
    return out


def process_manifest_row(row: dict[str, Any], processor: Any, config: RunConfig) -> dict[str, Any]:
    audio, sample_rate = load_audio_for_row(row, config)
    text = normalize_arabic_text(row.get("text", ""))
    prefix_text = build_prefix_text(processor, config, row)
    target = qwen_language_prefix(row, config) + text
    eos = getattr(processor.tokenizer, "eos_token", "") or ""
    full_text = prefix_text + target + eos
    full_inputs = processor(text=[full_text], audio=[audio], return_tensors="pt", padding=True, truncation=False)
    prefix_inputs = processor(text=[prefix_text], audio=[audio], return_tensors="pt", padding=True, truncation=False)
    labels = full_inputs["input_ids"].clone()
    prefix_len = int(prefix_inputs["attention_mask"].sum(dim=1).item())
    labels[0, :prefix_len] = -100
    pad_id = getattr(processor.tokenizer, "pad_token_id", None)
    if pad_id is not None:
        labels[labels == pad_id] = -100
    full_inputs["labels"] = labels
    sample = _squeeze_batch_dim(full_inputs)
    sample.update(
        {
            "uid": row["uid"],
            "source": row.get("source"),
            "source_group": row.get("source_group"),
            "split": row.get("split"),
            "duration": row.get("duration"),
            "text": text,
        }
    )
    return sample


class QwenManifestDataset:
    def __init__(self, rows: list[dict[str, Any]], processor: Any, config: RunConfig) -> None:
        self.rows = rows
        self.processor = processor
        self.config = config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        return process_manifest_row(row, self.processor, self.config)


def _pad_tensor_list(values: list[Any], pad_value: int | float = 0) -> Any:
    import torch

    values = [value if torch.is_tensor(value) else torch.tensor(value) for value in values]
    if not values:
        return torch.tensor([])
    max_ndim = max(value.ndim for value in values)
    values = [value.reshape((1,) if value.ndim == 0 else value.shape) for value in values]
    if max_ndim <= 1:
        max_len = max(value.shape[0] for value in values)
        out = values[0].new_full((len(values), max_len), pad_value)
        for idx, value in enumerate(values):
            out[idx, : value.shape[0]] = value
        return out
    if max_ndim == 2:
        dim0 = max(value.shape[0] for value in values)
        dim1 = max(value.shape[1] for value in values)
        out = values[0].new_full((len(values), dim0, dim1), pad_value)
        for idx, value in enumerate(values):
            out[idx, : value.shape[0], : value.shape[1]] = value
        return out
    raise ValueError(f"Cannot pad tensors with ndim={max_ndim}")


class DataCollatorForQwen3ASRLoRA:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        batch: dict[str, Any] = {}
        pad_id = getattr(self.processor.tokenizer, "pad_token_id", 0) or 0
        for key in ["input_ids", "attention_mask", "input_features", "feature_attention_mask", "labels"]:
            values = [feature[key] for feature in features if key in feature and feature[key] is not None]
            if not values:
                continue
            if key == "labels":
                batch[key] = _pad_tensor_list(values, -100).long()
            elif key == "input_ids":
                batch[key] = _pad_tensor_list(values, pad_id).long()
            elif key.endswith("attention_mask"):
                batch[key] = _pad_tensor_list(values, 0).long()
            else:
                batch[key] = _pad_tensor_list(values, 0.0).float()
        batch["uids"] = [feature.get("uid") for feature in features]
        batch["source_groups"] = [feature.get("source_group") for feature in features]
        return batch


def _device() -> str:
    import torch
    return "cuda" if torch.cuda.is_available() else "cpu"


def _extract_generation_sequences(out: Any) -> Any:
    import torch

    if isinstance(out, str):
        return out
    if hasattr(out, "sequences"):
        return out.sequences
    if isinstance(out, dict) and "sequences" in out:
        return out["sequences"]
    if torch.is_tensor(out):
        return out
    if isinstance(out, (list, tuple)):
        if not out:
            return out
        if isinstance(out[0], str):
            return out[0]
        if torch.is_tensor(out[0]):
            return out[0]
    return out


def decode_qwen_generate_output(bundle: QwenBundle, generated: Any, inputs: Optional[dict[str, Any]] = None) -> list[str]:
    import torch

    seq = _extract_generation_sequences(generated)
    if isinstance(seq, str):
        return [seq]
    if not torch.is_tensor(seq):
        return [str(seq)]
    if seq.ndim == 1:
        seq = seq.unsqueeze(0)

    decode_sequences = []
    input_ids = inputs.get("input_ids") if inputs is not None else None
    for idx in range(seq.shape[0]):
        row_seq = seq[idx]
        if torch.is_tensor(input_ids) and idx < input_ids.shape[0]:
            input_len = int(input_ids[idx].shape[-1])
            if row_seq.shape[-1] > input_len:
                row_seq = row_seq[input_len:]
        decode_sequences.append(row_seq)

    decoded = bundle.processor.tokenizer.batch_decode(
        decode_sequences,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )
    return list(decoded)


def clean_qwen_asr_decoded_text(decoded: str) -> str:
    decoded = str(decoded or "")
    if "<asr_text>" in decoded:
        decoded = decoded.split("<asr_text>")[-1]
    if "</asr_text>" in decoded:
        decoded = decoded.split("</asr_text>")[0]
    for token in ["<|im_start|>", "<|im_end|>", "<|endoftext|>", "assistant", "user", "system"]:
        decoded = decoded.replace(token, " ")
    return normalize_arabic_text(decoded)


def transcribe_rows(rows: list[dict[str, Any]], config: RunConfig, bundle: QwenBundle) -> list[tuple[str, float]]:
    if bundle.backend == "smoke":
        return [(normalize_arabic_text(row.get("text", "")), 0.0) for row in rows]

    import torch

    if not rows:
        return []

    audios = []
    prefix_texts = []
    for row in rows:
        audio, sample_rate = load_audio_for_row(row, config)
        audios.append(audio)
        prefix_texts.append(build_prefix_text(bundle.processor, config, row))

    model = bundle.model.to(_device())
    model.eval()
    started = time.perf_counter()

    inputs = bundle.processor(text=prefix_texts, audio=audios, return_tensors="pt", padding=True)
    for key, value in list(inputs.items()):
        if torch.is_tensor(value):
            inputs[key] = value.to(_device())
            if inputs[key].is_floating_point() and getattr(model, "dtype", None) is not None:
                inputs[key] = inputs[key].to(dtype=getattr(model, "dtype"))

    generation_kwargs = dict(
        max_new_tokens=config.generation_max_new_tokens,
        return_dict_in_generate=True,
        pad_token_id=getattr(bundle.processor.tokenizer, "pad_token_id", None) or getattr(bundle.processor.tokenizer, "eos_token_id", None),
        eos_token_id=getattr(bundle.processor.tokenizer, "eos_token_id", None),
    )

    try:
        with torch.no_grad():
            generated = model.generate(**inputs, **generation_kwargs)
        total_inference_seconds = time.perf_counter() - started
        decoded = decode_qwen_generate_output(bundle, generated, inputs=inputs)
        cleaned = [clean_qwen_asr_decoded_text(text) for text in decoded]
        avg_seconds = total_inference_seconds / max(1, len(rows))
        return [(text, avg_seconds) for text in cleaned]
    except Exception:
        if bundle.asr_wrapper is not None:
            results = []
            for row, audio in zip(rows, audios):
                item_started = time.perf_counter()
                with contextlib.suppress(Exception):
                    wrapper_results = bundle.asr_wrapper.transcribe(
                        audio=(audio, config.sample_rate),
                        language=config.language,
                        return_time_stamps=False,
                    )
                    text = wrapper_results[0].text if wrapper_results else ""
                    results.append((normalize_arabic_text(text), time.perf_counter() - item_started))
                    continue
                results.append(("", time.perf_counter() - item_started))
            return results
        raise


def prediction_record(row: dict[str, Any], prediction: str, inference_seconds: float) -> dict[str, Any]:
    reference = normalize_arabic_text(row.get("text", ""))
    return {
        "uid": row["uid"],
        "source": row.get("source"),
        "source_group": row.get("source_group"),
        "split": row.get("split"),
        "duration": row.get("duration"),
        "reference": reference,
        "prediction": prediction,
        "normalized_reference": normalize_metric_text(reference),
        "normalized_prediction": normalize_metric_text(prediction),
        "wer": word_error_rate(reference, prediction),
        "cer": char_error_rate(reference, prediction),
        "wer_loose": word_error_rate(reference, prediction, loose=True),
        "cer_loose": char_error_rate(reference, prediction, loose=True),
        "wer_no_punct": word_error_rate(reference, prediction, punctuation_insensitive=True),
        "cer_no_punct": char_error_rate(reference, prediction, punctuation_insensitive=True),
        "inference_seconds": inference_seconds,
    }


def summarize_prediction_records(records: list[dict[str, Any]], prediction_path: Path) -> dict[str, Any]:
    if not records:
        return {"num_predictions": 0, "prediction_path": str(prediction_path)}
    keys = ["wer", "cer", "wer_loose", "cer_loose", "wer_no_punct", "cer_no_punct"]
    out: dict[str, Any] = {"num_predictions": len(records), "prediction_path": str(prediction_path)}
    for key in keys:
        values = [float(record[key]) for record in records if record.get(key) is not None and math.isfinite(float(record[key]))]
        out[key] = statistics.mean(values) if values else None
    out["total_hours"] = sum(float(record.get("duration") or 0.0) for record in records) / 3600.0
    out["by_source_group"] = {}
    for group in sorted({str(record.get("source_group")) for record in records}):
        group_records = [record for record in records if str(record.get("source_group")) == group]
        out["by_source_group"][group] = {
            "rows": len(group_records),
            "wer": statistics.mean(float(record["wer"]) for record in group_records),
            "cer": statistics.mean(float(record["cer"]) for record in group_records),
        }
    object_dump = [
        record.get("uid")
        for record in records
        if "GenerateDecoderOnlyOutput" in str(record.get("prediction", ""))
        or "GenerateEncoderDecoderOutput" in str(record.get("prediction", ""))
        or "sequences=tensor" in str(record.get("prediction", ""))
    ]
    out["object_dump_predictions"] = len(object_dump)
    out["object_dump_prediction_uids"] = object_dump[:20]
    return out


def run_predictions(rows: list[dict[str, Any]], config: RunConfig, bundle: QwenBundle, *, name: str) -> dict[str, Any]:
    path = PREDICTION_DIR / f"{name}.jsonl"
    records: list[dict[str, Any]] = []
    selected_rows = list(rows)
    batch_size = max(1, int(config.eval_batch_size))
    for start in range(0, len(selected_rows), batch_size):
        batch_rows = selected_rows[start : start + batch_size]
        batch_outputs = transcribe_rows(batch_rows, config, bundle)
        for row, (prediction, inference_seconds) in zip(batch_rows, batch_outputs):
            records.append(prediction_record(row, prediction, inference_seconds))
        LOGGER.info("prediction progress name=%s done=%d total=%d", name, min(start + len(batch_rows), len(selected_rows)), len(selected_rows))
    jsonl_write(path, records)
    return summarize_prediction_records(records, path)


class TimedSaveCallback:
    def __init__(self, save_every_minutes: int) -> None:
        self.interval_seconds = max(60, int(save_every_minutes) * 60)
        self.last_save = time.monotonic()

    def on_step_end(self, args: Any, state: Any, control: Any, **_: Any) -> Any:
        now = time.monotonic()
        if now - self.last_save >= self.interval_seconds:
            control.should_save = True
            self.last_save = now
        return control


def _save_qwen_lora_bundle(model: Any, processor: Any, output_dir: Path) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(output_dir)
    with contextlib.suppress(Exception):
        processor.save_pretrained(output_dir)


def train_model(config: RunConfig, manifest_state: dict[str, Any]) -> dict[str, Any]:
    if config.smoke_mode:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        marker = {
            "smoke_mode": True,
            "message": "Smoke checkpoint placeholder; switch RUN_SMOKE_TEST to False for a real Qwen3-ASR 0.6B LoRA run.",
            "completed_at": timestamp,
        }
        for path in [BEST_MODEL_DIR / "adapter_config.json", FINAL_MODEL_DIR / "adapter_config.json"]:
            path.write_text(json.dumps(marker, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        summary = {
            "backend": "smoke",
            "best_checkpoint": str(BEST_MODEL_DIR),
            "best_metric": None,
            "completed_at": timestamp,
        }
        (RUN_DIR / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
        return summary

    import os
    import torch
    from transformers import Trainer, TrainerCallback, TrainingArguments

    class _QwenTrainer(Trainer):
        def _prepare_inputs(self, inputs: dict[str, Any]) -> dict[str, Any]:
            meta = {key: inputs.pop(key) for key in list(inputs.keys()) if key in {"uids", "source_groups"}}
            inputs = super()._prepare_inputs(inputs)
            model_dtype = getattr(self.model, "dtype", None)
            if model_dtype is not None:
                for key, value in list(inputs.items()):
                    if torch.is_tensor(value) and value.is_floating_point():
                        inputs[key] = value.to(dtype=model_dtype)
            inputs.update(meta)
            return inputs

        def compute_loss(self, model: Any, inputs: dict[str, Any], return_outputs: bool = False, **kwargs: Any) -> Any:
            inputs = {key: value for key, value in inputs.items() if key not in {"uids", "source_groups"}}
            return super().compute_loss(model, inputs, return_outputs=return_outputs, **kwargs)

    class _EpochMetricsCallback(TrainerCallback):
        def __init__(self, processor: Any, base_bundle: QwenBundle, val_rows: list[dict[str, Any]]) -> None:
            self.processor = processor
            self.base_bundle = base_bundle
            self.val_rows = val_rows
            self.best_wer = math.inf
            self.best_payload: dict[str, Any] | None = None
            self.epoch_metrics_path = RUN_DIR / "epoch_metrics.jsonl"
            if self.epoch_metrics_path.exists():
                self.epoch_metrics_path.unlink()

        def on_evaluate(self, args: Any, state: Any, control: Any, metrics: Optional[dict[str, Any]] = None, model: Any = None, **kwargs: Any) -> Any:
            metrics = dict(metrics or {})
            current_bundle = QwenBundle(
                model=model,
                processor=self.processor,
                backend="qwen_asr",
                asr_wrapper=self.base_bundle.asr_wrapper,
                adapter_path="__in_memory__",
            )
            val_generation_metrics = run_predictions(
                self.val_rows,
                config,
                current_bundle,
                name=f"epoch_{state.epoch:.2f}_val_predictions",
            )
            train_loss = None
            for item in reversed(state.log_history):
                if "loss" in item:
                    train_loss = item["loss"]
                    break
            summary = {
                "epoch": float(state.epoch or 0.0),
                "global_step": int(state.global_step),
                "train_loss": train_loss,
                "val_loss": metrics.get("eval_loss"),
                "wer": val_generation_metrics.get("wer"),
                "cer": val_generation_metrics.get("cer"),
                "prediction_path": val_generation_metrics.get("prediction_path"),
            }
            with self.epoch_metrics_path.open("a", encoding="utf-8") as handle:
                handle.write(json.dumps(summary, ensure_ascii=False, sort_keys=True) + "\n")
            LOGGER.info(
                "epoch_summary epoch=%.2f train_loss=%s val_loss=%s wer=%s cer=%s",
                summary["epoch"],
                summary["train_loss"],
                summary["val_loss"],
                summary["wer"],
                summary["cer"],
            )
            print(json.dumps(summary, ensure_ascii=False, indent=2))
            wer_value = summary["wer"]
            if wer_value is not None and float(wer_value) < float(self.best_wer):
                self.best_wer = float(wer_value)
                _save_qwen_lora_bundle(model, self.processor, BEST_MODEL_DIR)
                self.best_payload = {
                    "best_wer": self.best_wer,
                    "best_cer": summary["cer"],
                    "best_val_loss": summary["val_loss"],
                    "epoch": summary["epoch"],
                    "global_step": summary["global_step"],
                    "prediction_path": summary["prediction_path"],
                }
                (BEST_MODEL_DIR / "best_config.json").write_text(
                    json.dumps(self.best_payload, ensure_ascii=False, indent=2),
                    encoding="utf-8",
                )
                LOGGER.info("best model updated by WER: wer=%s epoch=%.2f", self.best_wer, summary["epoch"])
            return control

    bundle = load_qwen_bundle(config)
    model, lora_config = attach_lora(config, bundle.model)
    processor = bundle.processor
    train_dataset = QwenManifestDataset(manifest_state["train_rows"], processor, config)
    eval_dataset = QwenManifestDataset(manifest_state["val_rows"], processor, config)
    collator = DataCollatorForQwen3ASRLoRA(processor)

    if hasattr(model, "gradient_checkpointing_enable"):
        model.gradient_checkpointing_enable()
    if hasattr(model, "config") and hasattr(model.config, "use_cache"):
        model.config.use_cache = False

    os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")
    torch.backends.cuda.matmul.allow_tf32 = True
    if hasattr(torch.backends, "cudnn"):
        torch.backends.cudnn.allow_tf32 = True

    args = TrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        num_train_epochs=config.num_train_epochs,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        logging_strategy="epoch",
        logging_steps=1,
        save_strategy="epoch",
        save_total_limit=config.save_total_limit,
        do_eval=True,
        eval_strategy="epoch",
        prediction_loss_only=True,
        bf16=bool(torch.cuda.is_available() and config.use_bf16 and torch.cuda.is_bf16_supported()),
        fp16=bool(torch.cuda.is_available() and (config.use_fp16 or not (config.use_bf16 and torch.cuda.is_bf16_supported()))),
        dataloader_num_workers=config.train_dataloader_num_workers,
        dataloader_pin_memory=torch.cuda.is_available(),
        remove_unused_columns=False,
        report_to="none",
        save_safetensors=True,
        load_best_model_at_end=False,
    )
    epoch_callback = _EpochMetricsCallback(processor, bundle, manifest_state["val_rows"])
    trainer = _QwenTrainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        callbacks=[epoch_callback],
        tokenizer=processor.tokenizer,
    )
    trainer.train(resume_from_checkpoint=config.resume_from_checkpoint or None)
    _save_qwen_lora_bundle(model, processor, FINAL_MODEL_DIR)
    best_checkpoint = str(BEST_MODEL_DIR) if (BEST_MODEL_DIR / "adapter_config.json").exists() else str(FINAL_MODEL_DIR)
    summary = {
        "backend": "qwen_asr",
        "best_checkpoint": best_checkpoint,
        "best_metric": epoch_callback.best_wer if epoch_callback.best_wer < math.inf else None,
        "best_metric_name": "wer",
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_rows": len(manifest_state["train_rows"]),
        "val_rows": len(manifest_state["val_rows"]),
        "resolved_model_source": resolve_model_source(config),
        "lora_target_modules": infer_lora_target_modules(bundle.model),
        "forward_signature": str(inspect.signature(bundle.model.forward)),
        "epoch_metrics_path": str(RUN_DIR / "epoch_metrics.jsonl"),
    }
    if epoch_callback.best_payload is not None:
        summary["best_payload"] = epoch_callback.best_payload
    (RUN_DIR / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def prediction_file_health(path: Path) -> dict[str, Any]:
    records = jsonl_read(path)
    empty = [record.get("uid") for record in records if not str(record.get("prediction", "")).strip()]
    object_dump = [
        record.get("uid")
        for record in records
        if "GenerateDecoderOnlyOutput" in str(record.get("prediction", ""))
        or "GenerateEncoderDecoderOutput" in str(record.get("prediction", ""))
        or "sequences=tensor" in str(record.get("prediction", ""))
    ]
    return {
        "path": str(path),
        "rows": len(records),
        "empty_predictions": len(empty),
        "empty_prediction_uids": empty[:20],
        "object_dump_predictions": len(object_dump),
        "object_dump_prediction_uids": object_dump[:20],
        "hours": sum(float(record.get("duration") or 0.0) for record in records) / 3600.0,
    }


def write_summary_report(
    config: RunConfig,
    selection_summary: dict[str, Any],
    baseline_metrics: Optional[dict[str, Any]],
    val_metrics: Optional[dict[str, Any]],
    test_metrics: Optional[dict[str, Any]],
    training_summary: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "run_id": config.run_id,
        "notebook_path": str(NOTEBOOK_PATH),
        "run_dir": str(RUN_DIR),
        "selection_summary": selection_summary,
        "baseline_test_metrics": baseline_metrics,
        "val_prediction_metrics": val_metrics,
        "test_prediction_metrics": test_metrics,
        "training_summary": training_summary,
        "final_model_dir": str(FINAL_MODEL_DIR),
        "resolved_model_source": resolve_model_source(config),
    }
    summary_path = RUN_DIR / "summary_report.json"
    summary_path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_integrity_report(
    config: RunConfig,
    selection_summary: dict[str, Any],
    baseline_metrics: Optional[dict[str, Any]],
    val_metrics: Optional[dict[str, Any]],
    test_metrics: Optional[dict[str, Any]],
    training_summary: dict[str, Any],
) -> dict[str, Any]:
    prediction_health = {}
    if test_metrics and test_metrics.get("prediction_path"):
        prediction_health["test_predictions"] = prediction_file_health(Path(test_metrics["prediction_path"]))
    if val_metrics and val_metrics.get("prediction_path"):
        prediction_health["val_predictions"] = prediction_file_health(Path(val_metrics["prediction_path"]))
    results = {
        "run_id": config.run_id,
        "notebook_path": str(NOTEBOOK_PATH),
        "run_dir": str(RUN_DIR),
        "selection_summary": selection_summary,
        "baseline_test_metrics": baseline_metrics,
        "val_prediction_metrics": val_metrics,
        "test_prediction_metrics": test_metrics,
        "prediction_health": prediction_health,
        "training_summary": training_summary,
        "final_adapter_dir": str(FINAL_MODEL_DIR),
        "summary_report": str(RUN_DIR / "summary_report.json"),
        "decode_guard": "Qwen predictions are decoded from generated token ids, prompt tokens are stripped, and object dumps are flagged in integrity checks.",
        "drop_segments_at_or_above_seconds": config.drop_audio_at_or_above_seconds,
        "resolved_model_source": resolve_model_source(config),
        "qwen_system_prompt": config.qwen_system_prompt,
    }
    results_path = RUN_DIR / "mini_50h_100eval_streaming_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results
