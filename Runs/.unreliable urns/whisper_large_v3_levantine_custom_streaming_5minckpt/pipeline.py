"""Shared helpers for the Whisper Medium mini-run notebook."""

from __future__ import annotations

import contextlib
import io
import json
import logging
import math
import re
import statistics
import time
import unicodedata
import hashlib
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Optional

import numpy as np


REPO_ROOT = Path("/home/MohammadNabulsi/whisper")
RUN_DIR = REPO_ROOT / "Runs" / "whisper_large_v3_levantine_custom_streaming_5minckpt"
NOTEBOOK_PATH = RUN_DIR / "whisper_large_v3_lora_levantine_custom_streaming_5minckpt_run.ipynb"
SMOKE_NOTEBOOK_PATH = RUN_DIR / "whisper_large_v3_lora_levantine_custom_streaming_5minckpt_smoketest.ipynb"
EXECUTED_SMOKE_NOTEBOOK_PATH = RUN_DIR / "whisper_large_v3_lora_levantine_custom_streaming_5minckpt_smoketest_executed.ipynb"
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

AR_DIACRITICS_RE = re.compile(r"[\u0610-\u061a\u064b-\u065f\u0670\u06d6-\u06ed]")
CONTROL_RE = re.compile(r"[\u200b-\u200f\u202a-\u202e\u2066-\u2069\ufeff]")
SPACE_RE = re.compile(r"\s+")
ASR_TAG_RE = re.compile(r"<[^>]+>|\[[^\]]+\]")
AR_PUNCT_SPACING_RE = re.compile(r"\s*([،؛؟,.!?;:])\s*")
REPEATED_PUNCT_RE = re.compile(r"([،؛؟,.!?;:])\1+")
PUNCT_RE = re.compile(r"[\.,!\?;:،؛؟\-\(\)\[\]{}\"'“”‘’«»]")

LOGGER = logging.getLogger("whisper_large_v3_run")


@dataclass
class RunConfig:
    model_name: str = "openai/whisper-large-v3"
    run_id: str = "whisper_large_v3_levantine_custom_streaming_5minckpt"
    sample_rate: int = 16_000
    min_audio_seconds: float = 0.3
    drop_audio_at_or_above_seconds: float = 30.0
    train_hours_cap: float = 50.0
    eval_sample_cap: int = 100
    train_batch_size: int = 4
    eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.01
    max_grad_norm: float = 1.0
    num_train_epochs: int = 50
    logging_steps: int = 5
    checkpoint_every_minutes: int = 5
    save_total_limit: int = 6
    generation_max_new_tokens: int = 256
    early_stopping_patience: int = 3
    split_seed: int = 42
    train_dataloader_num_workers: int = 4
    language: str = "ar"
    task: str = "transcribe"
    smoke_mode: bool = False
    run_baseline_before_train: bool = True
    run_post_train_eval: bool = True
    force_rebuild_manifests: bool = False
    resume_from_checkpoint: Optional[str] = None
    use_fp16: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    lora_target_modules: tuple[str, ...] = ("q_proj", "v_proj")


def make_config(
    *,
    smoke_mode: bool = False,
    train_hours_cap: float = 50.0,
    eval_sample_cap: int = 100,
    num_train_epochs: int = 50,
    run_baseline_before_train: bool = True,
    run_post_train_eval: bool = True,
    resume_from_checkpoint: Optional[str] = None,
) -> RunConfig:
    config = RunConfig(
        smoke_mode=smoke_mode,
        train_hours_cap=train_hours_cap,
        eval_sample_cap=eval_sample_cap,
        num_train_epochs=num_train_epochs,
        run_baseline_before_train=run_baseline_before_train,
        run_post_train_eval=run_post_train_eval,
        resume_from_checkpoint=resume_from_checkpoint,
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
    log_path = LOG_DIR / "whisper_large_v3_run.log"
    if not LOGGER.handlers:
        LOGGER.setLevel(logging.INFO)
        formatter = logging.Formatter("%(asctime)s | %(levelname)s | %(name)s | %(message)s")
        file_handler = logging.FileHandler(log_path, encoding="utf-8")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        LOGGER.addHandler(file_handler)
        LOGGER.addHandler(stream_handler)
    return log_path


def resolve_resume_checkpoint(resume_from_checkpoint: Optional[str]) -> Optional[str]:
    if not resume_from_checkpoint:
        return None
    checkpoint_path = Path(resume_from_checkpoint)
    if not checkpoint_path.is_absolute():
        checkpoint_path = (REPO_ROOT / checkpoint_path).resolve()
    if not checkpoint_path.exists():
        raise FileNotFoundError(f"Resume checkpoint not found: {checkpoint_path}")
    return str(checkpoint_path)


def epoch_metrics_from_log_history(log_history: list[dict[str, Any]]) -> list[dict[str, Any]]:
    summaries: list[dict[str, Any]] = []
    latest_train_loss: Optional[float] = None
    for entry in log_history:
        if "loss" in entry:
            latest_train_loss = float(entry["loss"])
        if "eval_loss" not in entry:
            continue
        summaries.append(
            {
                "epoch": float(entry["epoch"]) if entry.get("epoch") is not None else None,
                "step": int(entry["step"]) if entry.get("step") is not None else None,
                "train_loss": latest_train_loss,
                "eval_loss": float(entry["eval_loss"]) if entry.get("eval_loss") is not None else None,
                "eval_wer": float(entry["eval_wer"]) if entry.get("eval_wer") is not None else None,
                "eval_cer": float(entry["eval_cer"]) if entry.get("eval_cer") is not None else None,
            }
        )
    return summaries


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
        normalized = normalized.replace("ا", "أ")
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
    if seconds >= config.drop_audio_at_or_above_seconds:
        return False
    return True


def _stable_hash(text: str, seed: int) -> int:
    import hashlib

    payload = f"{seed}|{text}".encode("utf-8")
    return int(hashlib.sha256(payload).hexdigest(), 16)


def _stable_row_order(rows: list[dict[str, Any]], *, seed: int, salt: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: _stable_hash(f"{salt}|{row.get('uid')}", seed))


def _duration_desc_order(rows: list[dict[str, Any]], *, seed: int, salt: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (-float(row.get("duration") or 0.0), _stable_hash(f"{salt}|{row.get('uid')}", seed)))


def _duration_asc_order(rows: list[dict[str, Any]], *, seed: int, salt: str) -> list[dict[str, Any]]:
    return sorted(rows, key=lambda row: (float(row.get("duration") or 0.0), _stable_hash(f"{salt}|{row.get('uid')}", seed)))


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


def _build_parquet_rows(paths: tuple[Path, ...], *, original_split: str, source_group: str, source_label: str) -> list[dict[str, Any]]:
    _, _, _, _, pq = _ensure_audio_tools()
    rows: list[dict[str, Any]] = []
    for path in paths:
        parquet_file = pq.ParquetFile(path)
        row_idx = 0
        for batch in parquet_file.iter_batches(
            batch_size=1024,
            columns=["seg_id", "transcription", "gender", "duration"],
        ):
            payload = batch.to_pylist()
            for item in payload:
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
                        "language": "apc_Arab",
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
                payload = {
                    column: batch.column(batch.schema.get_field_index(column)).to_pylist()
                    for column in use_cols
                }
                for offset in range(batch.num_rows):
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
                            "language": payload.get("language", ["apc_Arab"])[offset],
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

    parquet_val_rows, parquet_test_rows = _split_rows_half(
        filtered_eval_parquet,
        seed=config.split_seed,
        salt="custom-casablanca-heldout",
    )
    arrow_val_rows, arrow_test_rows = _split_rows_half(
        filtered_eval_arrow,
        seed=config.split_seed,
        salt="custom-omnilingual-heldout",
    )

    train_rows = [dict(row, split="train") for row in _stable_row_order(filtered_train, seed=config.split_seed, salt="custom-train")]
    val_rows = _stable_row_order(parquet_val_rows + arrow_val_rows, seed=config.split_seed, salt="custom-val")
    test_rows = _stable_row_order(parquet_test_rows + arrow_test_rows, seed=config.split_seed, salt="custom-test")

    if config.smoke_mode:
        train_rows = train_rows[:1]
        val_rows = val_rows[:1]
        test_rows = test_rows[:1]

    train_path = MANIFEST_DIR / "train" / "manifest_train_custom_levantine_lt30s.jsonl"
    val_path = MANIFEST_DIR / "val" / "manifest_val_custom_levantine_lt30s.jsonl"
    test_path = MANIFEST_DIR / "test" / "manifest_test_custom_levantine_lt30s.jsonl"
    all_path = MANIFEST_DIR / "manifest_all_lt30s.jsonl"
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


def _ensure_audio_tools() -> tuple[Any, Any, Any, Any]:
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
class WhisperBundle:
    model: Any
    processor: Any
    backend: str


class SmokeProcessor:
    tokenizer = None
    feature_extractor = None


class SmokeWhisperModel:
    def predict(self, reference: str) -> str:
        return reference


def load_whisper_bundle(config: RunConfig, *, adapter_path: str | Path | None = None) -> WhisperBundle:
    if config.smoke_mode:
        return WhisperBundle(model=SmokeWhisperModel(), processor=SmokeProcessor(), backend="smoke")

    from peft import LoraConfig, inject_adapter_in_model, set_peft_model_state_dict
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    model_root = RUN_DIR / "models" / "openai_whisper-large-v3"
    source = str(model_root) if (model_root / "config.json").exists() else config.model_name
    processor = WhisperProcessor.from_pretrained(source, language=config.language, task=config.task)
    model = WhisperForConditionalGeneration.from_pretrained(source)
    if source == config.model_name:
        model_root.mkdir(parents=True, exist_ok=True)
        processor.save_pretrained(model_root)
        model.save_pretrained(model_root)
    if adapter_path is not None:
        adapter_path = Path(adapter_path)
        peft_config = LoraConfig.from_pretrained(adapter_path)
        model = inject_adapter_in_model(peft_config, model)
        adapter_weights = adapter_path / "adapter_model.bin"
        if not adapter_weights.exists():
            raise FileNotFoundError(f"Missing adapter weights: {adapter_weights}")
        import torch
        state_dict = torch.load(adapter_weights, map_location="cpu")
        set_peft_model_state_dict(model, state_dict)
    prompt_ids = processor.get_decoder_prompt_ids(language=config.language, task=config.task)
    model.config.forced_decoder_ids = prompt_ids
    model.config.suppress_tokens = []
    if getattr(model, "generation_config", None) is not None:
        # Trainer eval uses generation_config during predict_with_generate, so keep it
        # aligned with the runtime language/task settings instead of the stale defaults
        # saved with the upstream checkpoint.
        model.generation_config.forced_decoder_ids = prompt_ids
        model.generation_config.language = config.language
        model.generation_config.task = config.task
        model.generation_config.suppress_tokens = []
    return WhisperBundle(model=model, processor=processor, backend="transformers")


def attach_lora(config: RunConfig, model: Any) -> Any:
    if config.smoke_mode:
        return model
    from peft import LoraConfig, inject_adapter_in_model

    peft_config = LoraConfig(
        r=config.lora_r,
        lora_alpha=config.lora_alpha,
        target_modules=list(config.lora_target_modules),
        lora_dropout=config.lora_dropout,
        bias="none",
        task_type="SEQ_2_SEQ_LM",
    )
    return inject_adapter_in_model(peft_config, model)


class WhisperManifestDataset:
    def __init__(self, rows: list[dict[str, Any]], processor: Any, config: RunConfig) -> None:
        self.rows = rows
        self.processor = processor
        self.config = config

    def __len__(self) -> int:
        return len(self.rows)

    def __getitem__(self, index: int) -> dict[str, Any]:
        row = self.rows[index]
        audio, sample_rate = load_audio_for_row(row, self.config)
        features = self.processor.feature_extractor(
            audio,
            sampling_rate=sample_rate,
            return_attention_mask=True,
        )
        input_features = features.input_features[0]
        feature_attention_mask = None
        if hasattr(features, "attention_mask") and features.attention_mask is not None:
            feature_attention_mask = features.attention_mask[0]
        sample = {
            "input_features": input_features,
            "text": normalize_arabic_text(row.get("text", "")),
            "uid": row["uid"],
            "source_group": row.get("source_group"),
        }
        if feature_attention_mask is not None:
            sample["attention_mask"] = feature_attention_mask
        return sample


def save_lora_artifacts(model: Any, processor: Any, output_dir: Path, training_config: RunConfig) -> None:
    from peft import LoraConfig, get_peft_model_state_dict
    import torch

    output_dir.mkdir(parents=True, exist_ok=True)
    peft_config = LoraConfig(
        r=training_config.lora_r,
        lora_alpha=training_config.lora_alpha,
        target_modules=list(training_config.lora_target_modules),
        lora_dropout=training_config.lora_dropout,
        bias="none",
    )
    peft_config.save_pretrained(output_dir)
    adapter_state = get_peft_model_state_dict(model)
    torch.save(adapter_state, output_dir / "adapter_model.bin")
    with contextlib.suppress(Exception):
        processor.save_pretrained(output_dir)


class DataCollatorSpeechSeq2SeqWithPadding:
    def __init__(self, processor: Any) -> None:
        self.processor = processor

    def __call__(self, features: list[dict[str, Any]]) -> dict[str, Any]:
        input_features = []
        for feature in features:
            item = {"input_features": feature["input_features"]}
            if "attention_mask" in feature:
                item["attention_mask"] = feature["attention_mask"]
            input_features.append(item)
        batch = self.processor.feature_extractor.pad(input_features, return_tensors="pt")
        label_texts = [feature["text"] for feature in features]
        labels_batch = self.processor.tokenizer(
            label_texts,
            return_tensors="pt",
            padding=True,
        )
        labels = labels_batch["input_ids"].masked_fill(labels_batch.attention_mask.ne(1), -100)
        if labels.shape[1] > 0 and (labels[:, 0] == self.processor.tokenizer.bos_token_id).all().item():
            labels = labels[:, 1:]
        batch["labels"] = labels
        return batch


def _device() -> str:
    import torch

    return "cuda" if torch.cuda.is_available() else "cpu"


def decode_generated_ids(bundle: WhisperBundle, generated: Any) -> str:
    if bundle.backend == "smoke":
        return ""
    sequences = generated
    if hasattr(generated, "sequences"):
        sequences = generated.sequences
    if isinstance(sequences, (list, tuple)) and sequences:
        sequences = sequences[0]
    if hasattr(sequences, "ndim") and sequences.ndim == 1:
        sequences = sequences[None, :]
    return bundle.processor.batch_decode(
        sequences,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=True,
    )[0]


def transcribe_row(row: dict[str, Any], config: RunConfig, bundle: WhisperBundle) -> tuple[str, float]:
    if bundle.backend == "smoke":
        return normalize_arabic_text(row.get("text", "")), 0.0

    import torch

    audio, sample_rate = load_audio_for_row(row, config)
    inputs = bundle.processor.feature_extractor(
        audio,
        sampling_rate=sample_rate,
        return_tensors="pt",
        return_attention_mask=True,
    )
    device = _device()
    model = bundle.model.to(device)
    input_features = inputs.input_features.to(device=device, dtype=getattr(model, "dtype", inputs.input_features.dtype))
    attention_mask = inputs.attention_mask.to(device) if hasattr(inputs, "attention_mask") and inputs.attention_mask is not None else None
    started = time.perf_counter()
    generate_kwargs = {
        "input_features": input_features,
        "max_new_tokens": config.generation_max_new_tokens,
        "return_dict_in_generate": True,
        "language": config.language,
        "task": config.task,
    }
    if attention_mask is not None:
        generate_kwargs["attention_mask"] = attention_mask
    with torch.no_grad():
        generated = model.generate(**generate_kwargs)
    inference_seconds = time.perf_counter() - started
    decoded = decode_generated_ids(bundle, generated)
    prediction = normalize_arabic_text(decoded)
    return prediction, inference_seconds


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
        if "GenerateEncoderDecoderOutput" in str(record.get("prediction", ""))
        or "sequences=tensor" in str(record.get("prediction", ""))
    ]
    out["object_dump_predictions"] = len(object_dump)
    out["object_dump_prediction_uids"] = object_dump[:20]
    return out


def run_predictions(
    rows: list[dict[str, Any]],
    config: RunConfig,
    bundle: WhisperBundle,
    *,
    name: str,
) -> dict[str, Any]:
    path = PREDICTION_DIR / f"{name}.jsonl"
    records: list[dict[str, Any]] = []
    for row in rows:
        prediction, inference_seconds = transcribe_row(row, config, bundle)
        records.append(prediction_record(row, prediction, inference_seconds))
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


def train_model(config: RunConfig, manifest_state: dict[str, Any]) -> dict[str, Any]:
    if config.smoke_mode:
        timestamp = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
        BEST_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        FINAL_MODEL_DIR.mkdir(parents=True, exist_ok=True)
        marker = {
            "smoke_mode": True,
            "message": "Smoke checkpoint placeholder; switch RUN_SMOKE_TEST to False for a real Whisper Medium LoRA run.",
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

    import torch
    from transformers import EarlyStoppingCallback, Seq2SeqTrainer, Seq2SeqTrainingArguments, TrainerCallback

    class _TimedSaveCallback(TrainerCallback):
        def __init__(self, minutes: int) -> None:
            self.impl = TimedSaveCallback(minutes)

        def on_step_end(self, args: Any, state: Any, control: Any, **kwargs: Any) -> Any:
            return self.impl.on_step_end(args, state, control, **kwargs)

    class _EpochMetricsCallback(TrainerCallback):
        def __init__(self) -> None:
            self.latest_train_loss: Optional[float] = None

        def on_log(self, args: Any, state: Any, control: Any, logs: Optional[dict[str, Any]] = None, **kwargs: Any) -> Any:
            if logs and logs.get("loss") is not None:
                self.latest_train_loss = float(logs["loss"])
            return control

        def on_evaluate(
            self,
            args: Any,
            state: Any,
            control: Any,
            metrics: Optional[dict[str, Any]] = None,
            **kwargs: Any,
        ) -> Any:
            metrics = metrics or {}
            epoch = metrics.get("epoch", state.epoch)
            parts = [f"epoch={float(epoch):.2f}" if epoch is not None else "epoch=?"]
            if self.latest_train_loss is not None:
                parts.append(f"train_loss={self.latest_train_loss:.4f}")
            if metrics.get("eval_loss") is not None:
                parts.append(f"eval_loss={float(metrics['eval_loss']):.4f}")
            if metrics.get("eval_wer") is not None:
                parts.append(f"eval_wer={float(metrics['eval_wer']):.4f}")
            if metrics.get("eval_cer") is not None:
                parts.append(f"eval_cer={float(metrics['eval_cer']):.4f}")
            if state.best_metric is not None:
                parts.append(f"best_eval_wer={float(state.best_metric):.4f}")
            summary = "Epoch summary | " + " | ".join(parts)
            LOGGER.info(summary)
            print(summary)
            return control

    bundle = load_whisper_bundle(config)
    model = attach_lora(config, bundle.model)
    train_dataset = WhisperManifestDataset(manifest_state["train_rows"], bundle.processor, config)
    eval_dataset = WhisperManifestDataset(manifest_state["val_rows"], bundle.processor, config)
    collator = DataCollatorSpeechSeq2SeqWithPadding(bundle.processor)

    def compute_metrics(eval_pred: Any) -> dict[str, float]:
        predictions = eval_pred.predictions
        labels = eval_pred.label_ids
        if isinstance(predictions, tuple):
            predictions = predictions[0]
        pad_token_id = bundle.processor.tokenizer.pad_token_id
        labels = np.where(labels == -100, pad_token_id, labels)
        decoded_predictions = [
            normalize_arabic_text(text)
            for text in bundle.processor.batch_decode(
                predictions,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        ]
        decoded_labels = [
            normalize_arabic_text(text)
            for text in bundle.processor.batch_decode(
                labels,
                skip_special_tokens=True,
                clean_up_tokenization_spaces=True,
            )
        ]
        if not decoded_labels:
            return {"wer": 0.0, "cer": 0.0}
        wer_values = [word_error_rate(ref, pred) for ref, pred in zip(decoded_labels, decoded_predictions)]
        cer_values = [char_error_rate(ref, pred) for ref, pred in zip(decoded_labels, decoded_predictions)]
        return {
            "wer": float(statistics.mean(wer_values)) if wer_values else 0.0,
            "cer": float(statistics.mean(cer_values)) if cer_values else 0.0,
        }

    args = Seq2SeqTrainingArguments(
        output_dir=str(OUTPUT_DIR),
        per_device_train_batch_size=config.train_batch_size,
        per_device_eval_batch_size=config.eval_batch_size,
        gradient_accumulation_steps=config.gradient_accumulation_steps,
        learning_rate=config.learning_rate,
        warmup_ratio=config.warmup_ratio,
        weight_decay=config.weight_decay,
        max_grad_norm=config.max_grad_norm,
        num_train_epochs=config.num_train_epochs,
        logging_steps=config.logging_steps,
        save_strategy="epoch",
        eval_strategy="epoch",
        predict_with_generate=True,
        generation_max_length=config.generation_max_new_tokens,
        save_total_limit=config.save_total_limit,
        load_best_model_at_end=True,
        metric_for_best_model="eval_wer",
        greater_is_better=False,
        remove_unused_columns=False,
        dataloader_num_workers=config.train_dataloader_num_workers,
        fp16=bool(torch.cuda.is_available() and config.use_fp16),
        bf16=False,
        report_to=[],
        label_names=["labels"],
    )
    epoch_metrics_callback = _EpochMetricsCallback()

    trainer = Seq2SeqTrainer(
        args=args,
        model=model,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=collator,
        compute_metrics=compute_metrics,
        callbacks=[
            _TimedSaveCallback(config.checkpoint_every_minutes),
            epoch_metrics_callback,
            EarlyStoppingCallback(early_stopping_patience=config.early_stopping_patience),
        ],
        tokenizer=bundle.processor.feature_extractor,
    )
    resume_checkpoint = resolve_resume_checkpoint(config.resume_from_checkpoint)
    if resume_checkpoint:
        LOGGER.info("Resuming training from checkpoint: %s", resume_checkpoint)
        print(f"Resuming training from checkpoint: {resume_checkpoint}")
    trainer.train(resume_from_checkpoint=resume_checkpoint)
    save_lora_artifacts(model, bundle.processor, BEST_MODEL_DIR, config)
    save_lora_artifacts(model, bundle.processor, FINAL_MODEL_DIR, config)
    summary = {
        "backend": "transformers",
        "best_checkpoint": str(trainer.state.best_model_checkpoint or BEST_MODEL_DIR),
        "best_metric": trainer.state.best_metric,
        "best_metric_name": "eval_wer",
        "best_model_selected_by": "lowest eval_wer",
        "resume_from_checkpoint": resume_checkpoint,
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_rows": len(manifest_state["train_rows"]),
        "val_rows": len(manifest_state["val_rows"]),
        "epoch_metrics": epoch_metrics_from_log_history(trainer.state.log_history),
    }
    (RUN_DIR / "training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return summary


def prediction_file_health(path: Path) -> dict[str, Any]:
    records = jsonl_read(path)
    empty = [record.get("uid") for record in records if not str(record.get("prediction", "")).strip()]
    object_dump = [
        record.get("uid")
        for record in records
        if "GenerateEncoderDecoderOutput" in str(record.get("prediction", ""))
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
        "decode_guard": "Whisper predictions are decoded from generated token ids via batch_decode; object dumps are flagged in integrity checks.",
        "drop_segments_at_or_above_seconds": config.drop_audio_at_or_above_seconds,
    }
    results_path = RUN_DIR / "mini_50h_100eval_streaming_results.json"
    results_path.write_text(json.dumps(results, ensure_ascii=False, indent=2), encoding="utf-8")
    return results

