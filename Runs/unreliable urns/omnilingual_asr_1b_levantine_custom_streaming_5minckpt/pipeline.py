"""Shared helpers for the OmniLingual ASR 1B Levantine custom run notebooks."""

from __future__ import annotations

import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import nbformat
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch
from nbclient import NotebookClient

from Runs.whisper_medium_levantine_custom_streaming_5minckpt.pipeline import (
    char_error_rate,
    jsonl_read,
    load_audio_for_row,
    normalize_arabic_text,
    normalize_metric_text,
    word_error_rate,
)


REPO_ROOT = Path("/home/MohammadNabulsi/whisper")
RUN_DIR = REPO_ROOT / "Runs" / "omnilingual_asr_1b_levantine_custom_streaming_5minckpt"
SOURCE_RUN_DIR = REPO_ROOT / "Runs" / "qwen3_asr_0_6b_levantine_custom_streaming_5minckpt"
OMNI_REPO_DIR = REPO_ROOT / "third_party" / "omnilingual-asr"
OMNI_SRC_DIR = OMNI_REPO_DIR / "src"
VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"

NOTEBOOK_PATH = RUN_DIR / "omnilingual_asr_1b_levantine_custom_streaming_5minckpt_run.ipynb"
SMOKE_NOTEBOOK_PATH = RUN_DIR / "omnilingual_asr_1b_levantine_custom_streaming_5minckpt_smoketest.ipynb"
EXECUTED_NOTEBOOK_PATH = RUN_DIR / "omnilingual_asr_1b_levantine_custom_streaming_5minckpt_run_executed.ipynb"
EXECUTED_SMOKE_NOTEBOOK_PATH = RUN_DIR / "omnilingual_asr_1b_levantine_custom_streaming_5minckpt_smoketest_executed.ipynb"

LOG_DIR = RUN_DIR / "logs"
DATASET_DIR = RUN_DIR / "dataset" / "version=0"
DATASET_SUMMARY_PATH = RUN_DIR / "dataset" / "language_distribution_0.tsv"
MANIFEST_DIR = RUN_DIR / "manifests"
CONFIG_DIR = RUN_DIR / "configs"
CHECKPOINT_DIR = RUN_DIR / "checkpoints"
SMOKE_CHECKPOINT_DIR = RUN_DIR / "smoke_checkpoints"
PREDICTION_DIR = RUN_DIR / "eval_predictions"
SMOKE_PREDICTION_DIR = RUN_DIR / "eval_predictions_smoke"
SUMMARY_REPORT_PATH = RUN_DIR / "summary_report.json"
TRAINING_SUMMARY_PATH = RUN_DIR / "training_summary.json"
INTEGRITY_REPORT_PATH = RUN_DIR / "integrity_report.json"
PREP_STATE_PATH = RUN_DIR / "prepared_manifest_state.json"

SOURCE_MANIFEST_PATHS = {
    "train": SOURCE_RUN_DIR / "manifests" / "train" / "manifest_train_custom_levantine.jsonl",
    "val": SOURCE_RUN_DIR / "manifests" / "val" / "manifest_val_custom_levantine.jsonl",
    "test": SOURCE_RUN_DIR / "manifests" / "test" / "manifest_test_custom_levantine.jsonl",
}

LOGGER = logging.getLogger("omnilingual_asr_1b_run")
OMNI_INFERENCE_MAX_AUDIO_SECONDS = 40.0
OMNI_INFERENCE_CHUNK_SECONDS = 35.0


@dataclass
class RunConfig:
    run_id: str = "omnilingual_asr_1b_levantine_custom_streaming_5minckpt"
    model_card: str = "omniASR_LLM_1B_v2"
    model_family: str = "wav2vec2_llama"
    model_arch: str = "1b_v2"
    tokenizer_name: str = "omniASR_tokenizer_written_v2"
    language: str = "apc_Arab"
    sample_rate: int = 16_000
    min_audio_seconds: float = 0.3
    max_audio_seconds: float | None = None
    smoke_mode: bool = False
    eval_sample_cap: int | None = None
    run_baseline_before_train: bool = True
    run_post_train_eval: bool = True
    train_num_steps: int = 500
    validate_every_n_steps: int = 50
    checkpoint_every_n_steps: int = 50
    publish_metrics_every_n_steps: int = 10
    batch_size: int = 1
    max_num_elements: int = 480_000
    beta_corpus: float = 0.5
    beta_language: float = 0.5
    freeze_encoder_for_n_steps: int = 0
    learning_rate: float = 5e-5


def make_config(
    *,
    smoke_mode: bool = False,
    eval_sample_cap: int | None = None,
    train_num_steps: int = 500,
    run_baseline_before_train: bool = True,
    run_post_train_eval: bool = True,
) -> RunConfig:
    config = RunConfig(
        smoke_mode=smoke_mode,
        eval_sample_cap=eval_sample_cap,
        train_num_steps=train_num_steps,
        run_baseline_before_train=run_baseline_before_train,
        run_post_train_eval=run_post_train_eval,
    )
    if smoke_mode:
        config.eval_sample_cap = 1
        config.max_audio_seconds = 30.0
        config.train_num_steps = 1
        config.validate_every_n_steps = 1
        config.checkpoint_every_n_steps = 1
        config.publish_metrics_every_n_steps = 1
    return config


def ensure_run_layout() -> None:
    for path in [
        RUN_DIR,
        LOG_DIR,
        DATASET_DIR.parent,
        MANIFEST_DIR,
        CONFIG_DIR,
        CHECKPOINT_DIR,
        SMOKE_CHECKPOINT_DIR,
        PREDICTION_DIR,
        SMOKE_PREDICTION_DIR,
    ]:
        path.mkdir(parents=True, exist_ok=True)
    for split in ("train", "val", "test"):
        (MANIFEST_DIR / split).mkdir(parents=True, exist_ok=True)


def setup_logging() -> Path:
    ensure_run_layout()
    log_path = LOG_DIR / "omnilingual_asr_1b_run.log"
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


def config_snapshot(config: RunConfig) -> dict[str, Any]:
    data = asdict(config)
    data.update(
        {
            "run_dir": str(RUN_DIR),
            "source_run_dir": str(SOURCE_RUN_DIR),
            "dataset_dir": str(DATASET_DIR),
            "dataset_summary_path": str(DATASET_SUMMARY_PATH),
            "notebook_path": str(NOTEBOOK_PATH),
            "smoke_notebook_path": str(SMOKE_NOTEBOOK_PATH),
            "checkpoint_dir": str(SMOKE_CHECKPOINT_DIR if config.smoke_mode else CHECKPOINT_DIR),
            "prediction_dir": str(prediction_dir_for_config(config)),
            "omni_repo_dir": str(OMNI_REPO_DIR),
        }
    )
    return data


def prediction_dir_for_config(config: RunConfig) -> Path:
    return SMOKE_PREDICTION_DIR if config.smoke_mode else PREDICTION_DIR


def checkpoint_dir_for_config(config: RunConfig) -> Path:
    return SMOKE_CHECKPOINT_DIR if config.smoke_mode else CHECKPOINT_DIR


def dataset_asset_name(config: RunConfig) -> str:
    return f"{config.run_id}_{'smoke' if config.smoke_mode else 'full'}_dataset"


def model_asset_name(config: RunConfig) -> str:
    return f"{config.run_id}_{'smoke' if config.smoke_mode else 'full'}_checkpoint"


def dataset_card_path(config: RunConfig) -> Path:
    return OMNI_SRC_DIR / "omnilingual_asr" / "cards" / "datasets" / f"{dataset_asset_name(config)}.yaml"


def model_card_path(config: RunConfig) -> Path:
    return OMNI_SRC_DIR / "omnilingual_asr" / "cards" / "models" / f"{model_asset_name(config)}.yaml"


def _source_rows() -> dict[str, list[dict[str, Any]]]:
    return {split: jsonl_read(path) for split, path in SOURCE_MANIFEST_PATHS.items()}


def _cap_eval_rows(rows: list[dict[str, Any]], cap: int | None) -> list[dict[str, Any]]:
    if cap is None:
        return rows
    if cap <= 0:
        return []
    return rows[:cap]


def select_rows(config: RunConfig) -> dict[str, list[dict[str, Any]]]:
    rows = _source_rows()
    train_rows = rows["train"][:1] if config.smoke_mode else rows["train"]
    val_rows = rows["val"][:1] if config.smoke_mode else _cap_eval_rows(rows["val"], config.eval_sample_cap)
    test_rows = rows["test"][:1] if config.smoke_mode else _cap_eval_rows(rows["test"], config.eval_sample_cap)
    return {"train": train_rows, "val": val_rows, "test": test_rows}


def _encode_audio_bytes(row: dict[str, Any], config: RunConfig) -> tuple[bytes, int]:
    audio, sample_rate = load_audio_for_row(row, config)
    audio = np.asarray(audio, dtype=np.float32)
    buffer = io.BytesIO()
    sf.write(buffer, audio, sample_rate, format="FLAC")
    payload = buffer.getvalue()
    return payload, int(audio.shape[0])


def _dataset_records(rows: list[dict[str, Any]], split: str, config: RunConfig) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for row in rows:
        audio_bytes, audio_size = _encode_audio_bytes(row, config)
        records.append(
            {
                "uid": row["uid"],
                "audio_bytes": audio_bytes,
                "audio_size": audio_size,
                "text": normalize_arabic_text(row.get("text", "")),
                "split": split,
                "language": str(row.get("language") or config.language),
                "corpus": str(row.get("source_group") or row.get("source") or "custom_levantine"),
                "duration_seconds": float(row.get("duration") or 0.0),
            }
        )
    return records


def _write_partition(records: list[dict[str, Any]], split: str, corpus: str, language: str) -> None:
    if not records:
        return
    partition_dir = DATASET_DIR / f"corpus={corpus}" / f"split={split}" / f"language={language}"
    partition_dir.mkdir(parents=True, exist_ok=True)
    table = pa.Table.from_pydict(
        {
            "uid": [record["uid"] for record in records],
            "audio_bytes": pa.array([record["audio_bytes"] for record in records], type=pa.binary()),
            "audio_size": [record["audio_size"] for record in records],
            "text": [record["text"] for record in records],
        }
    )
    pq.write_table(table, partition_dir / "data-00000.parquet")


def _write_dataset_summary(records: list[dict[str, Any]]) -> None:
    rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], float] = {}
    for record in records:
        if record["split"] != "train":
            continue
        key = (record["corpus"], record["language"])
        grouped[key] = grouped.get(key, 0.0) + float(record["duration_seconds"]) / 3600.0
    for (corpus, language), hours in sorted(grouped.items()):
        rows.append({"corpus": corpus, "language": language, "hours": hours})
    pd.DataFrame(rows).to_csv(DATASET_SUMMARY_PATH, sep="\t", index=False)


def write_dataset_asset_card(config: RunConfig) -> Path:
    path = dataset_card_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = (
        f"name: {dataset_asset_name(config)}\n"
        "dataset_family: mixture_parquet_asr_dataset\n"
        "dataset_config:\n"
        f"  data: {DATASET_DIR}\n"
        f"tokenizer_ref: {config.tokenizer_name}\n"
    )
    path.write_text(contents, encoding="utf-8")
    return path


def write_model_asset_card(config: RunConfig, checkpoint_path: Path) -> Path:
    path = model_card_path(config)
    path.parent.mkdir(parents=True, exist_ok=True)
    contents = (
        f"name: {model_asset_name(config)}\n"
        f"model_family: {config.model_family}\n"
        f"model_arch: {config.model_arch}\n"
        f"checkpoint: {checkpoint_path}\n"
        f"tokenizer_ref: {config.tokenizer_name}\n"
    )
    path.write_text(contents, encoding="utf-8")
    return path


def write_recipe_config(config: RunConfig) -> Path:
    ensure_run_layout()
    path = CONFIG_DIR / ("llm_finetune_smoke.yaml" if config.smoke_mode else "llm_finetune.yaml")
    contents = f"""model:
  name: "{config.model_card}"

dataset:
  name: "{dataset_asset_name(config)}"
  train_split: "train"
  valid_split: "val"
  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"
  mixture_parquet_storage_config:
    dataset_summary_path: "{DATASET_SUMMARY_PATH}"
    beta_corpus: {config.beta_corpus}
    beta_language: {config.beta_language}
  asr_task_config:
    min_audio_len: {int(config.min_audio_seconds * config.sample_rate)}
    max_audio_len: {int((config.max_audio_seconds if config.max_audio_seconds is not None else 1200.0) * config.sample_rate)}
    max_num_elements: {config.max_num_elements}
    batch_size: {config.batch_size}
    num_seqs_multiple_of: 1
    batch_shuffle_window: 1
    example_shuffle_window: 1
    normalize_audio: true

tokenizer:
  name: "{config.tokenizer_name}"

optimizer:
  config:
    lr: {config.learning_rate}

trainer:
  data_parallelism: "fsdp"
  fsdp:
    granularity: "stack"
    version: "v1"
    fp32_reduce: false
  freeze_encoder_for_n_steps: {config.freeze_encoder_for_n_steps}
  mixed_precision:
    dtype: "torch.bfloat16"
  grad_accumulation:
    num_batches: 1

regime:
  num_steps: {config.train_num_steps}
  validate_after_n_steps: 0
  validate_every_n_steps: {config.validate_every_n_steps}
  checkpoint_every_n_steps: {config.checkpoint_every_n_steps}
  save_model_only: true
  publish_metrics_every_n_steps: {config.publish_metrics_every_n_steps}
"""
    path.write_text(contents, encoding="utf-8")
    return path


def prepare_dataset(config: RunConfig) -> dict[str, Any]:
    ensure_run_layout()
    config_state = asdict(config)
    if PREP_STATE_PATH.exists():
        cached_state = json.loads(PREP_STATE_PATH.read_text(encoding="utf-8"))
        if cached_state.get("config") == config_state:
            recipe_config_path = Path(cached_state["recipe_config_path"])
            selection_summary = dict(cached_state["selection_summary"])
            selection_summary["loaded_from_cache"] = True
            return {
                "train_rows": cached_state["train_rows"],
                "val_rows": cached_state["val_rows"],
                "test_rows": cached_state["test_rows"],
                "selection_summary": selection_summary,
                "recipe_config_path": recipe_config_path,
            }

    selected = select_rows(config)
    all_records: list[dict[str, Any]] = []
    manifest_rows: dict[str, list[dict[str, Any]]] = {}
    if DATASET_DIR.parent.exists():
        shutil.rmtree(DATASET_DIR.parent)
    DATASET_DIR.mkdir(parents=True, exist_ok=True)
    for split, rows in selected.items():
        manifest_path = MANIFEST_DIR / split / f"manifest_{split}.jsonl"
        manifest_path.parent.mkdir(parents=True, exist_ok=True)
        manifest_path.write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
            encoding="utf-8",
        )
        manifest_rows[split] = rows
        split_records = _dataset_records(rows, split, config)
        all_records.extend(split_records)
        grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
        for record in split_records:
            key = (record["corpus"], record["language"])
            grouped.setdefault(key, []).append(record)
        for (corpus, language), records in grouped.items():
            _write_partition(records, split, corpus, language)
    _write_dataset_summary(all_records)
    dataset_card = write_dataset_asset_card(config)
    recipe_config = write_recipe_config(config)
    selection_summary = {
        "run_id": config.run_id,
        "source_manifests": {split: str(path) for split, path in SOURCE_MANIFEST_PATHS.items()},
        "selected_counts": {split: len(rows) for split, rows in selected.items()},
        "selected_hours": {
            split: sum(float(row.get("duration") or 0.0) for row in rows) / 3600.0
            for split, rows in selected.items()
        },
        "dataset_dir": str(DATASET_DIR),
        "dataset_summary_path": str(DATASET_SUMMARY_PATH),
        "dataset_asset_card": str(dataset_card),
        "recipe_config_path": str(recipe_config),
        "loaded_from_cache": False,
    }
    (MANIFEST_DIR / "custom_selection_summary.json").write_text(
        json.dumps(selection_summary, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    prepared_state = {
        "config": config_state,
        "train_rows": manifest_rows["train"],
        "val_rows": manifest_rows["val"],
        "test_rows": manifest_rows["test"],
        "selection_summary": selection_summary,
        "recipe_config_path": str(recipe_config),
    }
    PREP_STATE_PATH.write_text(json.dumps(prepared_state, ensure_ascii=False), encoding="utf-8")
    return {
        "train_rows": manifest_rows["train"],
        "val_rows": manifest_rows["val"],
        "test_rows": manifest_rows["test"],
        "selection_summary": selection_summary,
        "recipe_config_path": recipe_config,
    }


def _omni_env() -> dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    extra = str(OMNI_REPO_DIR)
    env["PYTHONPATH"] = extra if not existing else f"{extra}:{existing}"
    env["PYTHONUNBUFFERED"] = "1"
    return env


def _normalize_inference_language(language: str, config: RunConfig) -> str:
    language = str(language or config.language).strip()
    if not language:
        return config.language

    lowered = language.lower()
    if lowered in {"arabic", "ar", "arb", "ar-sa"}:
        return config.language

    return language


def load_omnilingual_pipeline(model_card: str):
    if str(OMNI_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(OMNI_REPO_DIR))
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    return ASRInferencePipeline(model_card=model_card, dtype=dtype)


def _transcribe_with_chunking(
    pipeline: Any,
    audio: np.ndarray,
    sample_rate: int,
    *,
    language: str,
) -> str:
    duration_seconds = len(audio) / max(1, sample_rate)
    if duration_seconds <= OMNI_INFERENCE_MAX_AUDIO_SECONDS:
        prediction = pipeline.transcribe(
            [{"waveform": torch.tensor(audio, dtype=torch.float32), "sample_rate": sample_rate}],
            lang=[language],
            batch_size=1,
        )
        return str(prediction[0]).strip()

    chunk_size = int(OMNI_INFERENCE_CHUNK_SECONDS * sample_rate)
    if chunk_size <= 0:
        raise ValueError(f"Invalid chunk size computed for sample_rate={sample_rate}")

    parts: list[str] = []
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size]
        if len(chunk) == 0:
            continue
        prediction = pipeline.transcribe(
            [{"waveform": torch.tensor(chunk, dtype=torch.float32), "sample_rate": sample_rate}],
            lang=[language],
            batch_size=1,
        )
        text = str(prediction[0]).strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def _prediction_records_from_rows(
    rows: list[dict[str, Any]],
    config: RunConfig,
    *,
    model_card: str,
) -> list[dict[str, Any]]:
    pipeline = load_omnilingual_pipeline(model_card)
    records: list[dict[str, Any]] = []
    for row in rows:
        audio, sample_rate = load_audio_for_row(row, config)
        language = _normalize_inference_language(str(row.get("language") or config.language), config)
        started = time.perf_counter()
        prediction = _transcribe_with_chunking(
            pipeline,
            audio,
            sample_rate,
            language=language,
        )
        elapsed = time.perf_counter() - started
        reference = normalize_arabic_text(row.get("text", ""))
        records.append(
            {
                "uid": row["uid"],
                "source": row.get("source"),
                "source_group": row.get("source_group"),
                "split": row.get("split"),
                "duration": row.get("duration"),
                "reference": reference,
                "prediction": normalize_arabic_text(prediction),
                "normalized_reference": normalize_metric_text(reference),
                "normalized_prediction": normalize_metric_text(prediction),
                "wer": word_error_rate(reference, prediction),
                "cer": char_error_rate(reference, prediction),
                "wer_loose": word_error_rate(reference, prediction, loose=True),
                "cer_loose": char_error_rate(reference, prediction, loose=True),
                "wer_no_punct": word_error_rate(reference, prediction, punctuation_insensitive=True),
                "cer_no_punct": char_error_rate(reference, prediction, punctuation_insensitive=True),
                "inference_seconds": elapsed,
            }
        )
    return records


def _write_prediction_records(prediction_path: Path, records: list[dict[str, Any]]) -> None:
    prediction_path.write_text(
        "".join(json.dumps(record, ensure_ascii=False) + "\n" for record in records),
        encoding="utf-8",
    )


def _prediction_metrics(prediction_path: Path, records: list[dict[str, Any]]) -> dict[str, Any]:
    metrics: dict[str, Any] = {
        "num_predictions": len(records),
        "prediction_path": str(prediction_path),
        "total_hours": sum(float(record.get("duration") or 0.0) for record in records) / 3600.0,
    }
    for key in ("wer", "cer", "wer_loose", "cer_loose", "wer_no_punct", "cer_no_punct"):
        values = [float(record[key]) for record in records]
        metrics[key] = float(sum(values) / len(values)) if values else None
    return metrics


def _run_predictions_in_fresh_process(
    rows: list[dict[str, Any]],
    config: RunConfig,
    *,
    model_card: str,
    prediction_path: Path,
) -> list[dict[str, Any]]:
    input_path = prediction_path.with_name(f"{prediction_path.stem}_input.jsonl")
    config_path = prediction_path.with_name(f"{prediction_path.stem}_config.json")
    input_path.write_text(
        "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows),
        encoding="utf-8",
    )
    config_path.write_text(json.dumps(asdict(config), ensure_ascii=False), encoding="utf-8")
    script = """
import json
import sys
from pathlib import Path

REPO_ROOT = Path('/home/MohammadNabulsi/whisper')
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from Runs.omnilingual_asr_1b_levantine_custom_streaming_5minckpt.pipeline import (
    RunConfig,
    _prediction_records_from_rows,
    _write_prediction_records,
)

input_path = Path(sys.argv[1])
config_path = Path(sys.argv[2])
model_card = sys.argv[3]
prediction_path = Path(sys.argv[4])
rows = [json.loads(line) for line in input_path.read_text(encoding='utf-8').splitlines() if line.strip()]
config = RunConfig(**json.loads(config_path.read_text(encoding='utf-8')))
records = _prediction_records_from_rows(rows, config, model_card=model_card)
_write_prediction_records(prediction_path, records)
"""
    subprocess.run(
        [str(VENV_PYTHON), "-c", script, str(input_path), str(config_path), model_card, str(prediction_path)],
        cwd=REPO_ROOT,
        env=_omni_env(),
        check=True,
    )
    return [json.loads(line) for line in prediction_path.read_text(encoding="utf-8").splitlines() if line.strip()]


def run_predictions(
    rows: list[dict[str, Any]],
    config: RunConfig,
    *,
    model_card: str,
    name: str,
) -> dict[str, Any]:
    prediction_dir = prediction_dir_for_config(config)
    prediction_dir.mkdir(parents=True, exist_ok=True)
    prediction_path = prediction_dir / f"{name}.jsonl"

    if model_card == config.model_card:
        records = _prediction_records_from_rows(rows, config, model_card=model_card)
        _write_prediction_records(prediction_path, records)
    else:
        records = _run_predictions_in_fresh_process(rows, config, model_card=model_card, prediction_path=prediction_path)

    return _prediction_metrics(prediction_path, records)


def run_training(config: RunConfig, manifest_state: dict[str, Any]) -> dict[str, Any]:
    output_dir = checkpoint_dir_for_config(config)
    if config.smoke_mode and output_dir.exists():
        shutil.rmtree(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    command = [
        str(VENV_PYTHON),
        "-m",
        "workflows.recipes.wav2vec2.asr",
        str(output_dir),
        "--config-file",
        str(manifest_state["recipe_config_path"]),
    ]
    LOGGER.info("Running training command: %s", " ".join(command))
    subprocess.run(command, cwd=OMNI_REPO_DIR, env=_omni_env(), check=True)
    checkpoint_path = find_latest_checkpoint(output_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No checkpoint file found under {output_dir}")
    write_model_asset_card(config, checkpoint_path)
    summary = {
        "backend": "fairseq2",
        "best_checkpoint": str(checkpoint_path),
        "model_asset_name": model_asset_name(config),
        "output_dir": str(output_dir),
        "completed_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "train_rows": len(manifest_state["train_rows"]),
        "val_rows": len(manifest_state["val_rows"]),
    }
    TRAINING_SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def find_latest_checkpoint(root: Path) -> Path | None:
    checkpoint_dirs = sorted(
        [path for path in root.glob("**/checkpoints") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for checkpoints_dir in checkpoint_dirs:
        step_models: list[tuple[int, Path]] = []
        for step_dir in checkpoints_dir.glob("step_*"):
            if not step_dir.is_dir():
                continue
            try:
                step_nr = int(step_dir.name.split("_", maxsplit=1)[1])
            except (IndexError, ValueError):
                continue
            model_dir = step_dir / "model"
            if model_dir.is_dir():
                step_models.append((step_nr, model_dir))
        if step_models:
            return max(step_models, key=lambda item: item[0])[1]

    candidates: list[Path] = []
    for pattern in ("**/model*.pt", "**/*.pt"):
        for path in root.glob(pattern):
            if not path.is_file():
                continue
            lower = str(path).lower()
            if any(token in lower for token in ("optimizer", "trainer", "rng", "metric", "data_reader")):
                continue
            candidates.append(path)
        if candidates:
            break
    if not candidates:
        return None
    return max(candidates, key=lambda path: path.stat().st_mtime)


def write_summary_report(
    config: RunConfig,
    selection_summary: dict[str, Any],
    baseline_metrics: dict[str, Any] | None,
    val_metrics: dict[str, Any] | None,
    test_metrics: dict[str, Any] | None,
    training_summary: dict[str, Any],
) -> dict[str, Any]:
    summary = {
        "run_id": config.run_id,
        "model_card": config.model_card,
        "selection_summary": selection_summary,
        "baseline_test_metrics": baseline_metrics,
        "val_prediction_metrics": val_metrics,
        "test_prediction_metrics": test_metrics,
        "training_summary": training_summary,
        "dataset_dir": str(DATASET_DIR),
        "checkpoint_dir": str(checkpoint_dir_for_config(config)),
    }
    SUMMARY_REPORT_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def write_integrity_report(
    config: RunConfig,
    training_summary: dict[str, Any],
    summary_report: dict[str, Any],
) -> dict[str, Any]:
    report = {
        "run_dir": str(RUN_DIR),
        "omni_repo_dir": str(OMNI_REPO_DIR),
        "model_card": config.model_card,
        "checkpoint_dir": str(checkpoint_dir_for_config(config)),
        "prediction_dir": str(prediction_dir_for_config(config)),
        "latest_checkpoint": training_summary.get("best_checkpoint"),
        "summary_report_written": bool(summary_report),
    }
    INTEGRITY_REPORT_PATH.write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return report


def _notebook_cells(smoke_mode: bool) -> list[Any]:
    return [
        nbformat.v4.new_markdown_cell(
            "# OmniLingual ASR 1B Levantine Custom Run\n"
            "This notebook mirrors the Levantine Whisper run flow using the official OmniLingual 1B LLM-ASR recipe, "
            "while reusing the exact same source train/val/test split manifests."
        ),
        nbformat.v4.new_code_cell(
            "from pathlib import Path\n"
            "import json\n"
            "import sys\n\n"
            "REPO_ROOT = Path('/home/MohammadNabulsi/whisper')\n"
            "if str(REPO_ROOT) not in sys.path:\n"
            "    sys.path.insert(0, str(REPO_ROOT))\n\n"
            "from Runs.omnilingual_asr_1b_levantine_custom_streaming_5minckpt.pipeline import (\n"
            "    NOTEBOOK_PATH,\n"
            "    SMOKE_NOTEBOOK_PATH,\n"
            "    config_snapshot,\n"
            "    ensure_run_layout,\n"
            "    make_config,\n"
            "    setup_logging,\n"
            ")\n\n"
            f"RUN_SMOKE_TEST = {str(smoke_mode)}\n"
            f"EVAL_SAMPLE_CAP = {1 if smoke_mode else 'None'}\n"
            f"TRAIN_NUM_STEPS = {1 if smoke_mode else 500}\n"
            "RUN_BASELINE_BEFORE_TRAIN = True\n"
            "RUN_POST_TRAIN_EVAL = True\n\n"
            "config = make_config(\n"
            "    smoke_mode=RUN_SMOKE_TEST,\n"
            "    eval_sample_cap=EVAL_SAMPLE_CAP,\n"
            "    train_num_steps=TRAIN_NUM_STEPS,\n"
            "    run_baseline_before_train=RUN_BASELINE_BEFORE_TRAIN,\n"
            "    run_post_train_eval=RUN_POST_TRAIN_EVAL,\n"
            ")\n\n"
            "ensure_run_layout()\n"
            "log_path = setup_logging()\n"
            "print(json.dumps(config_snapshot(config), ensure_ascii=False, indent=2))\n"
            "print('Notebook path:', SMOKE_NOTEBOOK_PATH if RUN_SMOKE_TEST else NOTEBOOK_PATH)\n"
            "print('Log path:', log_path)\n"
        ),
        nbformat.v4.new_code_cell(
            "from Runs.omnilingual_asr_1b_levantine_custom_streaming_5minckpt.pipeline import PREP_STATE_PATH, prepare_dataset\n\n"
            "manifest_state = prepare_dataset(config)\n"
            "selection_summary = manifest_state['selection_summary']\n"
            "print(json.dumps(selection_summary, ensure_ascii=False, indent=2))\n"
            "print('Prep cache:', PREP_STATE_PATH)\n"
            "print('Loaded from cache:', selection_summary.get('loaded_from_cache', False))\n"
        ),
        nbformat.v4.new_code_cell(
            "from Runs.omnilingual_asr_1b_levantine_custom_streaming_5minckpt.pipeline import run_predictions\n\n"
            "baseline_test_metrics = None\n"
            "if config.run_baseline_before_train:\n"
            "    baseline_test_metrics = run_predictions(\n"
            "        manifest_state['test_rows'],\n"
            "        config,\n"
            "        model_card=config.model_card,\n"
            "        name='baseline_test_predictions',\n"
            "    )\n"
            "print(json.dumps(baseline_test_metrics, ensure_ascii=False, indent=2))\n"
        ),
        nbformat.v4.new_code_cell(
            "from Runs.omnilingual_asr_1b_levantine_custom_streaming_5minckpt.pipeline import run_training\n\n"
            "training_summary = run_training(config, manifest_state)\n"
            "print(json.dumps(training_summary, ensure_ascii=False, indent=2))\n"
        ),
        nbformat.v4.new_code_cell(
            "from Runs.omnilingual_asr_1b_levantine_custom_streaming_5minckpt.pipeline import model_asset_name, run_predictions\n\n"
            "local_model_card = model_asset_name(config)\n"
            "val_prediction_metrics = run_predictions(\n"
            "    manifest_state['val_rows'],\n"
            "    config,\n"
            "    model_card=local_model_card,\n"
            "    name='tuned_val_predictions',\n"
            ") if config.run_post_train_eval else None\n\n"
            "test_prediction_metrics = run_predictions(\n"
            "    manifest_state['test_rows'],\n"
            "    config,\n"
            "    model_card=local_model_card,\n"
            "    name='tuned_test_predictions',\n"
            ") if config.run_post_train_eval else None\n\n"
            "print('Validation metrics:')\n"
            "print(json.dumps(val_prediction_metrics, ensure_ascii=False, indent=2))\n"
            "print('Test metrics:')\n"
            "print(json.dumps(test_prediction_metrics, ensure_ascii=False, indent=2))\n"
        ),
        nbformat.v4.new_code_cell(
            "from Runs.omnilingual_asr_1b_levantine_custom_streaming_5minckpt.pipeline import write_integrity_report, write_summary_report\n\n"
            "summary_report = write_summary_report(\n"
            "    config,\n"
            "    selection_summary,\n"
            "    baseline_test_metrics,\n"
            "    val_prediction_metrics,\n"
            "    test_prediction_metrics,\n"
            "    training_summary,\n"
            ")\n"
            "integrity_report = write_integrity_report(config, training_summary, summary_report)\n"
            "print(json.dumps(summary_report, ensure_ascii=False, indent=2))\n"
            "print(json.dumps(integrity_report, ensure_ascii=False, indent=2))\n"
        ),
    ]


def write_notebooks() -> dict[str, str]:
    ensure_run_layout()
    smoke_nb = nbformat.v4.new_notebook(cells=_notebook_cells(smoke_mode=True))
    full_nb = nbformat.v4.new_notebook(cells=_notebook_cells(smoke_mode=False))
    nbformat.write(smoke_nb, SMOKE_NOTEBOOK_PATH)
    nbformat.write(full_nb, NOTEBOOK_PATH)
    return {"smoke_notebook": str(SMOKE_NOTEBOOK_PATH), "run_notebook": str(NOTEBOOK_PATH)}


def execute_notebook(input_path: Path, output_path: Path, *, timeout: int = 0) -> Path:
    notebook = nbformat.read(input_path, as_version=4)
    client = NotebookClient(notebook, timeout=timeout or None, kernel_name="python3")
    client.execute()
    nbformat.write(notebook, output_path)
    return output_path
