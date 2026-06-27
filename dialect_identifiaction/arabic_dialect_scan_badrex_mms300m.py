#!/usr/bin/env python3
"""
Arabic dialect distribution scanner using:
  badrex/mms-300m-arabic-dialect-identifier

What it does:
- Scans Parquet and Arrow shards.
- Extracts audio from common schemas: audio dict, bytes, audio_bytes, path.
- Runs the Hugging Face audio-classification model locally.
- Writes:
    1) predictions.jsonl          -> one record per sample
    2) .logs/dialect_progress.log -> human-readable live percentages
    3) .logs/progress_latest.json -> machine-readable latest state
    4) summary.json               -> final/latest summary

Important:
- The model predicts broad ADI-5 groups only:
  MSA, Egyptian, Gulf, Levantine, Maghrebi.
- "unseen" here is NOT a trained class. It is a heuristic bucket:
  top model confidence < UNKNOWN_THRESHOLD.
"""

from __future__ import annotations

import argparse
import io
import json
import math
import os
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Tuple

import numpy as np
import torch
import torchaudio
import pyarrow.parquet as pq
from datasets import Dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


# -----------------------------
# Configuration defaults
# -----------------------------

DEFAULT_MODEL_ID = "badrex/mms-300m-arabic-dialect-identifier"
DEFAULT_TARGET_SR = 16_000
DEFAULT_UNKNOWN_THRESHOLD = 0.55
DEFAULT_MIN_SECONDS = 2.0


# -----------------------------
# File discovery
# -----------------------------

def discover_shards(data_roots: List[Path]) -> List[Path]:
    """Find Parquet and Arrow files under one or more paths."""
    files: List[Path] = []
    for root in data_roots:
        root = Path(root).expanduser()
        if root.is_file() and root.suffix.lower() in {".parquet", ".arrow"}:
            files.append(root)
        elif root.is_dir():
            files.extend(sorted(root.rglob("*.parquet")))
            files.extend(sorted(root.rglob("*.arrow")))
        else:
            print(f"[WARN] Path not found or unsupported: {root}")
    # stable order, dedupe
    return sorted(set(files), key=lambda p: str(p))


def iter_rows_from_file(path: Path, parquet_batch_size: int = 64) -> Iterator[Tuple[int, Dict[str, Any]]]:
    """Yield (row_idx, row_dict) from .parquet or .arrow."""
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        pf = pq.ParquetFile(path)
        row_offset = 0
        for batch in pf.iter_batches(batch_size=parquet_batch_size):
            data = batch.to_pydict()
            n = batch.num_rows
            keys = list(data.keys())
            for i in range(n):
                yield row_offset + i, {k: data[k][i] for k in keys}
            row_offset += n
        return

    if suffix == ".arrow":
        # Works for Hugging Face Arrow shards written by datasets.
        try:
            ds = Dataset.from_file(str(path))
            for i, row in enumerate(ds):
                yield i, dict(row)
            return
        except Exception as e:
            raise RuntimeError(
                f"Could not read Arrow file with datasets.Dataset.from_file: {path}. "
                f"If your .arrow is not a Hugging Face Arrow file, convert it first or add a custom reader. "
                f"Original error: {repr(e)}"
            )

    raise ValueError(f"Unsupported file suffix: {path}")


# -----------------------------
# Audio loading / extraction
# -----------------------------

def _to_float32_mono(wave: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave)

    if wave.ndim == 2:
        # Accept both [channels, time] and [time, channels]
        if wave.shape[0] <= 8 and wave.shape[0] < wave.shape[1]:
            wave = wave.mean(axis=0)
        else:
            wave = wave.mean(axis=1)

    if wave.dtype.kind in {"i", "u"}:
        max_val = np.iinfo(wave.dtype).max
        wave = wave.astype(np.float32) / max_val
    else:
        wave = wave.astype(np.float32)

    # Remove NaN/Inf without changing valid values.
    wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
    return wave


def _decode_audio_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    """
    Decode bytes using torchaudio first.
    This handles WAV and often MP3/FLAC depending on installed backends.
    """
    bio = io.BytesIO(audio_bytes)
    waveform, sr = torchaudio.load(bio)
    wave = waveform.mean(dim=0).cpu().numpy()
    return _to_float32_mono(wave), int(sr)


def _load_audio_path(path: str | Path, base_dir: Optional[Path] = None) -> Tuple[np.ndarray, int]:
    p = Path(path)
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    waveform, sr = torchaudio.load(str(p))
    wave = waveform.mean(dim=0).cpu().numpy()
    return _to_float32_mono(wave), int(sr)


def extract_audio(row: Dict[str, Any], shard_path: Path) -> Tuple[np.ndarray, int, str]:
    """
    Supports common ASR dataset schemas:
    - row["audio"] = {"array": ..., "sampling_rate": ...}
    - row["audio"] = {"bytes": ..., "path": ...}
    - row["bytes"] = b"..."
    - row["audio_bytes"] = b"..."
    - row["path"] / row["audio_path"] = filepath
    """
    base_dir = shard_path.parent

    # 1) Hugging Face-style audio object
    audio = row.get("audio")
    if isinstance(audio, dict):
        if audio.get("array") is not None:
            sr = int(audio.get("sampling_rate") or DEFAULT_TARGET_SR)
            return _to_float32_mono(np.asarray(audio["array"])), sr, "audio.array"

        if audio.get("bytes") is not None:
            return (*_decode_audio_bytes(audio["bytes"]), "audio.bytes")

        if audio.get("path"):
            return (*_load_audio_path(audio["path"], base_dir=base_dir), "audio.path")

    # 2) Direct bytes columns
    for key in ["bytes", "audio_bytes", "wav_bytes"]:
        val = row.get(key)
        if isinstance(val, (bytes, bytearray, memoryview)):
            return (*_decode_audio_bytes(bytes(val)), key)

    # 3) Direct path columns
    for key in ["path", "audio_path", "file", "filepath", "wav", "wav_path"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return (*_load_audio_path(val, base_dir=base_dir), key)

    raise ValueError(f"No usable audio column found. Available columns: {list(row.keys())}")


def resample_if_needed(wave: np.ndarray, sr: int, target_sr: int) -> Tuple[np.ndarray, int]:
    wave = _to_float32_mono(wave)
    if sr == target_sr:
        return wave, sr

    wav_t = torch.from_numpy(wave)
    resampled = torchaudio.functional.resample(wav_t, orig_freq=sr, new_freq=target_sr)
    return resampled.cpu().numpy().astype(np.float32), target_sr


# -----------------------------
# Prediction
# -----------------------------

def load_model(model_id: str, device: str):
    processor = AutoFeatureExtractor.from_pretrained(model_id)
    model = AutoModelForAudioClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()

    id2label = {}
    for k, v in model.config.id2label.items():
        id2label[int(k)] = v

    labels = [id2label[i] for i in sorted(id2label)]
    return processor, model, id2label, labels


@torch.no_grad()
def predict_one(
    wave: np.ndarray,
    sr: int,
    processor,
    model,
    id2label: Dict[int, str],
    device: str,
) -> List[Dict[str, float | str]]:
    inputs = processor(
        wave,
        sampling_rate=sr,
        return_tensors="pt",
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()

    preds = [
        {"label": id2label[i], "score": float(probs[i])}
        for i in range(len(probs))
    ]
    preds.sort(key=lambda x: x["score"], reverse=True)
    return preds


# -----------------------------
# Logging / resume
# -----------------------------

def make_uid(shard_path: Path, row_idx: int, row: Dict[str, Any]) -> str:
    for key in ["uid", "id", "seg_id", "segment_id", "recording_id"]:
        if row.get(key) is not None:
            return f"{shard_path.name}:{key}={row[key]}"
    return f"{shard_path.name}:row={row_idx}"


def pick_text(row: Dict[str, Any]) -> Optional[str]:
    for key in ["text", "transcription", "transcript", "raw_text", "normalized", "sentence"]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def load_existing_state(predictions_path: Path) -> Tuple[set, Counter, int, int, int]:
    done_uids = set()
    counts = Counter()
    processed_ok = 0
    skipped = 0
    errors = 0

    if not predictions_path.exists():
        return done_uids, counts, processed_ok, skipped, errors

    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            uid = rec.get("uid")
            if uid:
                done_uids.add(uid)
            status = rec.get("status")
            if status == "ok":
                counts[rec.get("final_label", "unseen")] += 1
                processed_ok += 1
            elif status and status.startswith("skipped"):
                skipped += 1
            elif status == "error":
                errors += 1

    return done_uids, counts, processed_ok, skipped, errors


def pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else round(100.0 * n / d, 4)


def write_progress(
    *,
    log_path: Path,
    latest_json_path: Path,
    summary_json_path: Path,
    counts: Counter,
    labels: List[str],
    processed_ok: int,
    skipped: int,
    errors: int,
    attempted: int,
    current_file: str,
    current_uid: str,
    start_time: float,
) -> None:
    elapsed = time.time() - start_time
    label_order = list(labels) + ["unseen"]

    percentages = {
        label: pct(counts.get(label, 0), processed_ok)
        for label in label_order
    }

    state = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "attempted_in_this_run": attempted,
        "processed_ok_total": processed_ok,
        "skipped_total": skipped,
        "errors_total": errors,
        "current_file": current_file,
        "current_uid": current_uid,
        "elapsed_seconds": round(elapsed, 2),
        "dialect_counts": {label: int(counts.get(label, 0)) for label in label_order},
        "dialect_percentages": percentages,
        "unseen_count": int(counts.get("unseen", 0)),
        "unseen_percentage": percentages.get("unseen", 0.0),
    }

    latest_json_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    summary_json_path.write_text(
        json.dumps(state, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    pct_bits = " ".join(f"{label}={percentages[label]:.2f}%" for label in label_order)
    line = (
        f"[{state['time']}] "
        f"ok={processed_ok} skipped={skipped} errors={errors} "
        f"unseen={percentages.get('unseen', 0.0):.2f}% "
        f"{pct_bits} "
        f"file={current_file} uid={current_uid}"
    )

    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


# -----------------------------
# Main run
# -----------------------------

def run(args: argparse.Namespace) -> None:
    data_roots = [Path(p) for p in args.data_root]
    output_dir = Path(args.output_dir)
    log_dir = output_dir / ".logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.jsonl"
    progress_log_path = log_dir / "dialect_progress.log"
    latest_json_path = log_dir / "progress_latest.json"
    summary_json_path = output_dir / "summary.json"

    shards = discover_shards(data_roots)
    if not shards:
        raise FileNotFoundError(f"No .parquet or .arrow files found under: {data_roots}")

    print(f"Found {len(shards)} shards.")
    for p in shards[:10]:
        print(" -", p)
    if len(shards) > 10:
        print(f" ... plus {len(shards) - 10} more")

    device = args.device
    if device == "auto":
        device = "cuda" if torch.cuda.is_available() else "cpu"

    print(f"Loading model: {args.model_id}")
    print(f"Device: {device}")
    processor, model, id2label, labels = load_model(args.model_id, device)
    print(f"Model labels: {labels}")

    if args.resume:
        done_uids, counts, processed_ok, skipped, errors = load_existing_state(predictions_path)
        print(f"Resume enabled. Loaded {len(done_uids)} completed/skipped/error UIDs.")
    else:
        done_uids, counts, processed_ok, skipped, errors = set(), Counter(), 0, 0, 0
        if predictions_path.exists():
            backup = predictions_path.with_suffix(".jsonl.bak")
            predictions_path.rename(backup)
            print(f"Existing predictions moved to: {backup}")

    start_time = time.time()
    attempted = 0

    with predictions_path.open("a", encoding="utf-8") as pred_out:
        stop = False

        for shard_path in shards:
            if stop:
                break

            print(f"\nScanning shard: {shard_path}")

            try:
                row_iter = iter_rows_from_file(shard_path, parquet_batch_size=args.parquet_batch_size)
                for row_idx, row in row_iter:
                    uid = make_uid(shard_path, row_idx, row)

                    if uid in done_uids:
                        continue

                    if args.max_samples is not None and processed_ok >= args.max_samples:
                        stop = True
                        break

                    attempted += 1
                    record: Dict[str, Any] = {
                        "uid": uid,
                        "source_file": str(shard_path),
                        "row_idx": row_idx,
                        "text": pick_text(row),
                    }

                    try:
                        wave, sr, audio_source = extract_audio(row, shard_path)
                        duration_sec_original = len(wave) / float(sr) if sr else 0.0

                        if duration_sec_original < args.min_seconds:
                            skipped += 1
                            record.update({
                                "status": "skipped_short",
                                "audio_source": audio_source,
                                "duration_sec": round(duration_sec_original, 4),
                                "reason": f"duration < {args.min_seconds}s",
                            })
                        else:
                            wave, sr = resample_if_needed(wave, sr, args.target_sr)
                            duration_sec = len(wave) / float(sr)

                            preds = predict_one(
                                wave=wave,
                                sr=sr,
                                processor=processor,
                                model=model,
                                id2label=id2label,
                                device=device,
                            )

                            top = preds[0]
                            top_label = str(top["label"])
                            top_score = float(top["score"])
                            final_label = top_label if top_score >= args.unknown_threshold else "unseen"

                            counts[final_label] += 1
                            processed_ok += 1

                            record.update({
                                "status": "ok",
                                "audio_source": audio_source,
                                "duration_sec": round(duration_sec, 4),
                                "top_label": top_label,
                                "top_score": top_score,
                                "final_label": final_label,
                                "unknown_threshold": args.unknown_threshold,
                                "predictions": preds,
                            })

                    except Exception as e:
                        errors += 1
                        record.update({
                            "status": "error",
                            "error": repr(e),
                            "traceback": traceback.format_exc(limit=3),
                        })

                    pred_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    pred_out.flush()

                    done_uids.add(uid)

                    if args.log_every == 1 or attempted % args.log_every == 0:
                        write_progress(
                            log_path=progress_log_path,
                            latest_json_path=latest_json_path,
                            summary_json_path=summary_json_path,
                            counts=counts,
                            labels=labels,
                            processed_ok=processed_ok,
                            skipped=skipped,
                            errors=errors,
                            attempted=attempted,
                            current_file=str(shard_path),
                            current_uid=uid,
                            start_time=start_time,
                        )

            except Exception as e:
                errors += 1
                err_rec = {
                    "uid": f"{shard_path.name}:SHARD_ERROR",
                    "source_file": str(shard_path),
                    "status": "error",
                    "error": repr(e),
                    "traceback": traceback.format_exc(limit=5),
                }
                pred_out.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
                pred_out.flush()
                print(f"[ERROR] Could not scan shard {shard_path}: {repr(e)}")

    write_progress(
        log_path=progress_log_path,
        latest_json_path=latest_json_path,
        summary_json_path=summary_json_path,
        counts=counts,
        labels=labels,
        processed_ok=processed_ok,
        skipped=skipped,
        errors=errors,
        attempted=attempted,
        current_file="DONE",
        current_uid="DONE",
        start_time=start_time,
    )

    print("\nDone.")
    print(f"Predictions: {predictions_path}")
    print(f"Progress log: {progress_log_path}")
    print(f"Latest JSON: {latest_json_path}")
    print(f"Summary JSON: {summary_json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data-root",
        action="append",
        required=True,
        help="Directory or shard file to scan. Repeat for multiple roots.",
    )
    parser.add_argument(
        "--output-dir",
        default="/home/MohammadNabulsi/whisper/Runs/dialect_scan_badrex_mms300m",
        help="Where outputs/logs are written.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--target-sr", type=int, default=DEFAULT_TARGET_SR)
    parser.add_argument("--unknown-threshold", type=float, default=DEFAULT_UNKNOWN_THRESHOLD)
    parser.add_argument("--min-seconds", type=float, default=DEFAULT_MIN_SECONDS)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--parquet-batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
