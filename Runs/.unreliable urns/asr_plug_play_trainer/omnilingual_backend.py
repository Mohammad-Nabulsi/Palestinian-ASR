from __future__ import annotations

import csv
import io
import json
import logging
import os
import shutil
import subprocess
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple



LOGGER = logging.getLogger("asr_universal")
REPO_ROOT = Path(__file__).resolve().parents[2]
OMNI_REPO_DIR = REPO_ROOT / "third_party" / "omnilingual-asr"
OMNI_SRC_DIR = OMNI_REPO_DIR / "src"
DEFAULT_VENV_PYTHON = REPO_ROOT / ".venv" / "bin" / "python"
OMNI_INFERENCE_MAX_AUDIO_SECONDS = 40.0
OMNI_INFERENCE_CHUNK_SECONDS = 35.0


@dataclass
class OmnilingualModelInfo:
    model_card: str
    model_family: str
    model_arch: str
    tokenizer_name: str


def resolve_omnilingual_model(model_id: str) -> OmnilingualModelInfo:
    model_key = str(model_id or "").strip().lower().replace("-", "_").replace("/", "_")
    tokenizer_name = "omniASR_tokenizer_written_v2"
    if "ctc" in model_key:
        if "300m" in model_key:
            return OmnilingualModelInfo("omniASR_CTC_300M_v2", "wav2vec2_ctc", "300m_v2", tokenizer_name)
        if "1b" in model_key:
            return OmnilingualModelInfo("omniASR_CTC_1B_v2", "wav2vec2_ctc", "1b_v2", tokenizer_name)
        raise ValueError(f"Unsupported Omnilingual CTC model_id={model_id}")
    if "300m" in model_key:
        return OmnilingualModelInfo("omniASR_LLM_300M_v2", "wav2vec2_llama", "300m_v2", tokenizer_name)
    if "1b" in model_key:
        return OmnilingualModelInfo("omniASR_LLM_1B_v2", "wav2vec2_llama", "1b_v2", tokenizer_name)
    if "3b" in model_key:
        return OmnilingualModelInfo("omniASR_LLM_3B_v2", "wav2vec2_llama", "3b_v2", tokenizer_name)
    if "7b" in model_key:
        return OmnilingualModelInfo("omniASR_LLM_7B_v2", "wav2vec2_llama", "7b_v2", tokenizer_name)
    raise ValueError(
        f"Could not map Omnilingual model_id={model_id} to an official local recipe card. "
        "Use a name containing 300M/1B/3B/7B and optionally CTC."
    )


def _omni_env(cfg: Dict[str, Any]) -> Dict[str, str]:
    env = os.environ.copy()
    existing = env.get("PYTHONPATH", "")
    env["PYTHONPATH"] = str(OMNI_REPO_DIR) if not existing else f"{OMNI_REPO_DIR}:{existing}"
    env["PYTHONUNBUFFERED"] = "1"
    cache_dir = cfg.get("model", {}).get("cache_dir")
    if cache_dir:
        env["HF_HOME"] = str(cache_dir)
    return env


def _omni_python(cfg: Dict[str, Any]) -> str:
    explicit = cfg.get("model", {}).get("omnilingual_python")
    if explicit:
        return str(explicit)
    if DEFAULT_VENV_PYTHON.exists():
        return str(DEFAULT_VENV_PYTHON)
    return sys.executable


def _prediction_dir(work_dir: Path) -> Path:
    path = work_dir / "predictions"
    path.mkdir(parents=True, exist_ok=True)
    return path


def _read_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _write_jsonl(path: Path, rows: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(json.dumps(row, ensure_ascii=False) + "\n" for row in rows), encoding="utf-8")


def _normalize_arabic_text(text: Any) -> str:
    return " ".join(str(text or "").strip().split())


def _normalize_metric_text(text: Any) -> str:
    text = _normalize_arabic_text(text)
    return (
        text.replace("إ", "ا")
        .replace("أ", "ا")
        .replace("آ", "ا")
        .replace("ى", "ي")
        .replace("ؤ", "و")
        .replace("ئ", "ي")
    )


def _edit_distance(a: List[str], b: List[str]) -> int:
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, start=1):
        curr = [i]
        for j, cb in enumerate(b, start=1):
            curr.append(min(curr[j - 1] + 1, prev[j] + 1, prev[j - 1] + (ca != cb)))
        prev = curr
    return prev[-1]


def _compute_wer_cer(preds: List[str], refs: List[str]) -> Dict[str, float]:
    total_word_edits = 0
    total_words = 0
    total_char_edits = 0
    total_chars = 0
    for pred, ref in zip(preds, refs):
        pred_norm = _normalize_metric_text(pred)
        ref_norm = _normalize_metric_text(ref)
        pred_words = pred_norm.split()
        ref_words = ref_norm.split()
        total_word_edits += _edit_distance(pred_words, ref_words)
        total_words += max(1, len(ref_words))
        total_char_edits += _edit_distance(list(pred_norm), list(ref_norm))
        total_chars += max(1, len(ref_norm))
    return {
        "wer": total_word_edits / total_words if total_words else 0.0,
        "cer": total_char_edits / total_chars if total_chars else 0.0,
    }


def _normalize_inference_language(language: str, cfg: Dict[str, Any]) -> str:
    language = str(language or '').strip()
    if not language:
        return str(cfg.get('model', {}).get('omnilingual_language') or cfg.get('data', {}).get('language') or 'apc_Arab')
    lowered = language.lower()
    if lowered in {'ar', 'arabic', 'arb', 'ar-sa'}:
        return str(cfg.get('model', {}).get('omnilingual_language') or 'apc_Arab')
    return language


def _audio_libs():
    try:
        import numpy as np  # type: ignore
        import soundfile as sf  # type: ignore
    except Exception as exc:
        raise RuntimeError("Omnilingual backend requires numpy and soundfile in the active Python environment.") from exc
    return np, sf


def _load_audio(path: str, sample_rate: int):
    np, sf = _audio_libs()
    audio, sr = sf.read(path, dtype="float32", always_2d=False)
    audio = np.asarray(audio, dtype=np.float32)
    if audio.ndim == 2:
        audio = audio.mean(axis=1 if audio.shape[0] > audio.shape[1] else 0).astype(np.float32)
    if int(sr) != int(sample_rate):
        try:
            import librosa  # type: ignore
        except Exception as exc:  # pragma: no cover
            raise RuntimeError("librosa is required when Omnilingual input audio is not already at the target sample rate.") from exc
        audio = librosa.resample(audio, orig_sr=int(sr), target_sr=int(sample_rate)).astype(np.float32)
        sr = sample_rate
    return audio, int(sr)


def _prepared_rows(work_dir: Path) -> Dict[str, List[Dict[str, Any]]]:
    prepared = work_dir / "prepared"
    return {
        "train": _read_jsonl(prepared / "train.jsonl"),
        "validation": _read_jsonl(prepared / "validation.jsonl"),
        "test": _read_jsonl(prepared / "test.jsonl"),
    }


def _omni_paths(work_dir: Path) -> Dict[str, Path]:
    root = work_dir / "omnilingual_recipe"
    return {
        "root": root,
        "dataset_dir": root / "dataset" / "version=0",
        "summary_tsv": root / "dataset" / "language_distribution_0.tsv",
        "cards_dataset_dir": OMNI_SRC_DIR / "omnilingual_asr" / "cards" / "datasets",
        "cards_model_dir": OMNI_SRC_DIR / "omnilingual_asr" / "cards" / "models",
        "config_dir": root / "configs",
        "checkpoints_dir": root / "checkpoints",
        "state_path": root / "prepared_state.json",
    }


def _run_asset_name(work_dir: Path) -> str:
    return f"{work_dir.name}_omnilingual_dataset"


def _tuned_asset_name(work_dir: Path) -> str:
    return f"{work_dir.name}_omnilingual_tuned"


def _write_partition(records: List[Dict[str, Any]], split: str, corpus: str, language: str, dataset_dir: Path) -> None:
    import pyarrow as pa  # type: ignore
    import pyarrow.parquet as pq  # type: ignore

    if not records:
        return
    partition_dir = dataset_dir / f"corpus={corpus}" / f"split={split}" / f"language={language}"
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


def prepare_omnilingual_recipe_dataset(cfg: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    work_dir = Path(cfg.get("output", {}).get("work_dir", "asr_run"))
    sample_rate = int(cfg.get("model", {}).get("sample_rate", 16000))
    info = resolve_omnilingual_model(model_id)
    paths = _omni_paths(work_dir)
    rows_by_split = _prepared_rows(work_dir)

    if paths["root"].exists():
        shutil.rmtree(paths["root"])
    paths["dataset_dir"].mkdir(parents=True, exist_ok=True)
    paths["config_dir"].mkdir(parents=True, exist_ok=True)
    paths["cards_dataset_dir"].mkdir(parents=True, exist_ok=True)
    paths["cards_model_dir"].mkdir(parents=True, exist_ok=True)

    all_train_stats: Dict[Tuple[str, str], float] = {}
    for split_name, source_name in [("train", "train"), ("val", "validation"), ("test", "test")]:
        grouped: Dict[Tuple[str, str], List[Dict[str, Any]]] = {}
        for row in rows_by_split[source_name]:
            _, sf = _audio_libs()
            audio, sr = _load_audio(row["audio_path"], sample_rate)
            payload = io.BytesIO()
            sf.write(payload, audio, sr, format="FLAC")
            corpus = str(row.get("source_group") or row.get("source") or "custom_levantine")
            language = _normalize_inference_language(str(row.get("language") or cfg.get("data", {}).get("language") or "apc_Arab"), cfg)
            record = {
                "uid": row["uid"],
                "audio_bytes": payload.getvalue(),
                "audio_size": int(audio.shape[0]),
                "text": _normalize_arabic_text(row.get("text", "")),
                "duration_seconds": float(row.get("duration") or 0.0),
            }
            grouped.setdefault((corpus, language), []).append(record)
            if split_name == "train":
                key = (corpus, language)
                all_train_stats[key] = all_train_stats.get(key, 0.0) + record["duration_seconds"] / 3600.0
        for (corpus, language), records in grouped.items():
            _write_partition(records, split_name, corpus, language, paths["dataset_dir"])

    with paths["summary_tsv"].open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["corpus", "language", "hours"], delimiter="\t")
        writer.writeheader()
        for (corpus, language), hours in sorted(all_train_stats.items()):
            writer.writerow({"corpus": corpus, "language": language, "hours": hours})

    dataset_asset_name = _run_asset_name(work_dir)
    dataset_card = paths["cards_dataset_dir"] / f"{dataset_asset_name}.yaml"
    dataset_card.write_text(
        "\n".join(
            [
                f"name: {dataset_asset_name}",
                "dataset_family: mixture_parquet_asr_dataset",
                "dataset_config:",
                f"  data: {paths['dataset_dir']}",
                f"tokenizer_ref: {info.tokenizer_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )

    training = cfg.get("training", {})
    max_audio_len_seconds = cfg.get("model", {}).get("max_audio_seconds")
    if max_audio_len_seconds is None:
        max_audio_len_seconds = 1200.0
    recipe_config = paths["config_dir"] / "finetune.yaml"
    recipe_config.write_text(
        f"""model:
  name: "{info.model_card}"

dataset:
  name: "{dataset_asset_name}"
  train_split: "train"
  valid_split: "val"
  storage_mode: "MIXTURE_PARQUET"
  task_mode: "ASR"
  mixture_parquet_storage_config:
    dataset_summary_path: "{paths['summary_tsv']}"
    beta_corpus: {float(training.get('beta_corpus', 0.5))}
    beta_language: {float(training.get('beta_language', 0.5))}
  asr_task_config:
    min_audio_len: {int(float(cfg.get('data', {}).get('min_seconds', 0.05)) * sample_rate)}
    max_audio_len: {int(float(max_audio_len_seconds) * sample_rate)}
    max_num_elements: {int(training.get('max_num_elements', 480000))}
    batch_size: {int(training.get('per_device_train_batch_size', 1))}
    num_seqs_multiple_of: 1
    batch_shuffle_window: {int(training.get('batch_shuffle_window', 1))}
    example_shuffle_window: {int(training.get('example_shuffle_window', 1))}
    normalize_audio: true

tokenizer:
  name: "{info.tokenizer_name}"

optimizer:
  config:
    lr: {float(training.get('learning_rate', 5e-5))}

trainer:
  data_parallelism: "{training.get('omnilingual_data_parallelism', 'fsdp')}"
  fsdp:
    granularity: "{training.get('omnilingual_fsdp_granularity', 'stack')}"
    version: "{training.get('omnilingual_fsdp_version', 'v1')}"
    fp32_reduce: false
  freeze_encoder_for_n_steps: {int(training.get('freeze_encoder_for_n_steps', 0))}
  mixed_precision:
    dtype: "{training.get('omnilingual_dtype', 'torch.bfloat16')}"
  grad_accumulation:
    num_batches: {int(training.get('gradient_accumulation_steps', 1))}

regime:
  num_steps: {int(training.get('train_num_steps', 500))}
  validate_after_n_steps: 0
  validate_every_n_steps: {int(training.get('validate_every_n_steps', 50))}
  checkpoint_every_n_steps: {int(training.get('checkpoint_every_n_steps', 50))}
  save_model_only: true
  publish_metrics_every_n_steps: {int(training.get('logging_steps', 10))}
""",
        encoding="utf-8",
    )

    state = {
        "model_info": info.__dict__,
        "dataset_asset_name": dataset_asset_name,
        "recipe_config_path": str(recipe_config),
        "dataset_dir": str(paths["dataset_dir"]),
        "dataset_summary_path": str(paths["summary_tsv"]),
    }
    paths["state_path"].write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    return state


def find_latest_checkpoint(root: Path) -> Optional[Path]:
    checkpoint_dirs = sorted(
        [path for path in root.glob("**/checkpoints") if path.is_dir()],
        key=lambda path: path.stat().st_mtime,
        reverse=True,
    )
    for checkpoints_dir in checkpoint_dirs:
        step_models: List[Tuple[int, Path]] = []
        for step_dir in checkpoints_dir.glob("step_*"):
            if not step_dir.is_dir():
                continue
            try:
                step_nr = int(step_dir.name.split("_", maxsplit=1)[1])
            except Exception:
                continue
            model_dir = step_dir / "model"
            if model_dir.is_dir():
                step_models.append((step_nr, model_dir))
        if step_models:
            return max(step_models, key=lambda item: item[0])[1]
    return None


def train_omnilingual(cfg: Dict[str, Any], model_id: str) -> Dict[str, Any]:
    work_dir = Path(cfg.get("output", {}).get("work_dir", "asr_run"))
    paths = _omni_paths(work_dir)
    state_path = paths["state_path"]
    if not state_path.exists():
        state = prepare_omnilingual_recipe_dataset(cfg, model_id)
    else:
        state = json.loads(state_path.read_text(encoding="utf-8"))
    output_dir = paths["checkpoints_dir"]
    output_dir.mkdir(parents=True, exist_ok=True)

    command = [
        _omni_python(cfg),
        "-m",
        "workflows.recipes.wav2vec2.asr",
        str(output_dir),
        "--config-file",
        str(state["recipe_config_path"]),
    ]
    LOGGER.info("Running Omnilingual training command: %s", " ".join(command))
    subprocess.run(command, cwd=OMNI_REPO_DIR, env=_omni_env(cfg), check=True)

    checkpoint_path = find_latest_checkpoint(output_dir)
    if checkpoint_path is None:
        raise FileNotFoundError(f"No Omnilingual checkpoint found under {output_dir}")

    info = OmnilingualModelInfo(**state["model_info"])
    asset_name = _tuned_asset_name(work_dir)
    model_card_path = paths["cards_model_dir"] / f"{asset_name}.yaml"
    model_card_path.write_text(
        "\n".join(
            [
                f"name: {asset_name}",
                f"model_family: {info.model_family}",
                f"model_arch: {info.model_arch}",
                f"checkpoint: {checkpoint_path}",
                f"tokenizer_ref: {info.tokenizer_name}",
                "",
            ]
        ),
        encoding="utf-8",
    )
    summary = {
        "backend": "omnilingual_recipe",
        "base_model_card": info.model_card,
        "tuned_model_card": asset_name,
        "best_checkpoint": str(checkpoint_path),
        "recipe_config_path": state["recipe_config_path"],
        "dataset_dir": state["dataset_dir"],
    }
    (work_dir / "omnilingual_training_summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    return summary


def _load_inference_pipeline():
    if str(OMNI_REPO_DIR) not in sys.path:
        sys.path.insert(0, str(OMNI_REPO_DIR))
    from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline  # type: ignore

    import torch

    dtype = torch.bfloat16 if torch.cuda.is_available() and torch.cuda.is_bf16_supported() else torch.float32
    return ASRInferencePipeline, torch, dtype


def _transcribe_with_chunking(pipeline: Any, torch_mod: Any, audio: Any, sample_rate: int, language: str) -> str:
    duration_seconds = len(audio) / max(1, sample_rate)
    if duration_seconds <= OMNI_INFERENCE_MAX_AUDIO_SECONDS:
        prediction = pipeline.transcribe([{"waveform": torch_mod.tensor(audio, dtype=torch_mod.float32), "sample_rate": sample_rate}], lang=[language], batch_size=1)
        return str(prediction[0]).strip()
    chunk_size = int(OMNI_INFERENCE_CHUNK_SECONDS * sample_rate)
    parts: List[str] = []
    for start in range(0, len(audio), chunk_size):
        chunk = audio[start : start + chunk_size]
        if len(chunk) == 0:
            continue
        prediction = pipeline.transcribe([{"waveform": torch_mod.tensor(chunk, dtype=torch_mod.float32), "sample_rate": sample_rate}], lang=[language], batch_size=1)
        text = str(prediction[0]).strip()
        if text:
            parts.append(text)
    return " ".join(parts).strip()


def eval_omnilingual(cfg: Dict[str, Any], model_id: str, split: str = "test") -> Dict[str, Any]:
    work_dir = Path(cfg.get("output", {}).get("work_dir", "asr_run"))
    rows = _prepared_rows(work_dir)[split]
    state = json.loads(_omni_paths(work_dir)["state_path"].read_text(encoding="utf-8")) if _omni_paths(work_dir)["state_path"].exists() else None
    if (OMNI_SRC_DIR / "omnilingual_asr" / "cards" / "models" / f"{_tuned_asset_name(work_dir)}.yaml").exists():
        model_card = _tuned_asset_name(work_dir)
    elif state:
        model_card = state["model_info"]["model_card"]
    else:
        model_card = resolve_omnilingual_model(model_id).model_card

    ASRInferencePipeline, torch_mod, dtype = _load_inference_pipeline()
    pipeline = ASRInferencePipeline(model_card=model_card, dtype=dtype)

    preds: List[str] = []
    refs: List[str] = []
    prediction_rows: List[Dict[str, Any]] = []
    for index, row in enumerate(rows, start=1):
        LOGGER.info("Omnilingual eval %d/%d: %s", index, len(rows), row["uid"])
        audio, sample_rate = _load_audio(row["audio_path"], int(cfg.get("model", {}).get("sample_rate", 16000)))
        language = _normalize_inference_language(str(row.get("language") or cfg.get("data", {}).get("language") or "apc_Arab"), cfg)
        started = time.perf_counter()
        pred = _transcribe_with_chunking(pipeline, torch_mod, audio, sample_rate, language)
        elapsed = time.perf_counter() - started
        preds.append(pred)
        refs.append(row["text"])
        prediction_rows.append(
            {
                "uid": row["uid"],
                "audio_path": row["audio_path"],
                "reference": row["text"],
                "prediction": pred,
                "duration": row.get("duration"),
                "language": language,
                "inference_seconds": elapsed,
            }
        )

    metrics = _compute_wer_cer(preds, refs)
    prediction_path = _prediction_dir(work_dir) / f"omnilingual_{split}_predictions.jsonl"
    _write_jsonl(prediction_path, prediction_rows)
    result = {"split": split, "model_card": model_card, "prediction_path": str(prediction_path), "n": len(rows), **metrics}
    (work_dir / f"omnilingual_{split}_metrics.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    return result
