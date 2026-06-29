#!/usr/bin/env python3
"""
Arabic dialect distribution scanner using:
  badrex/mms-300m-arabic-dialect-identifier

What it does:
- Scans Parquet and Arrow shards.
- Filters to rows whose saved text-model LEV score meets a threshold.
- Extracts audio from common schemas: audio dict, audio bytes, audio_bytes, path.
- Runs the Hugging Face audio-classification model locally.
- Writes:
    1) predictions.jsonl          -> one record per processed sample
    2) row_probabilities.jsonl    -> per-row text+audio probabilities
    3) dialect_progress.log       -> human-readable live percentages
    4) progress_latest.json       -> machine-readable latest state
    5) .logs/dialect_progress.log -> legacy hidden copy for compatibility
    6) .logs/progress_latest.json -> legacy hidden copy for compatibility
    7) summary.json               -> final/latest summary

Important:
- The model predicts broad ADI-5 groups only.
- "unseen" here is NOT a trained class. It is a heuristic bucket:
  top model confidence < UNKNOWN_THRESHOLD.
- This run only processes rows that already met the text-side candidate filter.
"""

from __future__ import annotations

import argparse
import io
import json
import time
import traceback
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import numpy as np
import pyarrow.parquet as pq
import torch
import torchaudio
from datasets import Dataset
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


DEFAULT_MODEL_ID = "badrex/mms-300m-arabic-dialect-identifier"
DEFAULT_TARGET_SR = 16_000
DEFAULT_UNKNOWN_THRESHOLD = 0.55
DEFAULT_MIN_SECONDS = 2.0
DEFAULT_TEXT_TARGET_LABEL = "LEV"
DEFAULT_TEXT_TARGET_THRESHOLD = 0.80
DEFAULT_TEXT_PROBABILITIES_PATH = (
    "/home/MohammadNabulsi/whisper/"
    "Runs/text_dialect_scan_marbertv2_written_clean_masc_c_qasr/row_probabilities.jsonl"
)


def discover_shards(data_roots: List[Path]) -> List[Path]:
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
    return sorted(set(files), key=lambda p: str(p))


def iter_rows_from_file(path: Path, parquet_batch_size: int = 64) -> Iterator[Tuple[int, Dict[str, Any]]]:
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


def make_source_row_key(shard_name: str, row_idx: int) -> str:
    return f"{shard_name}:row={row_idx}"


def load_text_candidates(
    probabilities_path: Path,
    target_label: str,
    threshold: float,
) -> Tuple[Dict[str, Dict[str, Any]], Dict[str, set], Dict[str, Any]]:
    candidates_by_row_key: Dict[str, Dict[str, Any]] = {}
    row_indices_by_shard: Dict[str, set] = defaultdict(set)
    source_counts: Counter = Counter()
    rows_scored = 0

    with probabilities_path.open("r", encoding="utf-8") as f:
        for line in f:
            rec = json.loads(line)
            label_scores = rec.get("label_scores") or {}
            target_score = label_scores.get(target_label)
            if target_score is None:
                continue

            rows_scored += 1
            if float(target_score) < threshold:
                continue

            uid = rec.get("uid")
            source_file = rec.get("source_file")
            row_idx = rec.get("row_idx")
            if source_file is None or row_idx is None:
                continue

            shard_name = Path(str(source_file)).name
            row_idx = int(row_idx)
            source_row_key = make_source_row_key(shard_name, row_idx)
            source = rec.get("source")

            candidates_by_row_key[source_row_key] = {
                "uid": uid,
                "source_row_key": source_row_key,
                "source_file": str(source_file),
                "row_idx": row_idx,
                "source": source,
                "text": rec.get("text"),
                "text_top_label": rec.get("top_label"),
                "text_top_score": rec.get("top_score"),
                "text_final_label": rec.get("final_label"),
                "text_unknown_threshold": rec.get("unknown_threshold"),
                "text_label_scores": label_scores,
                "text_target_label": rec.get("target_label", target_label),
                "text_target_label_score": float(target_score),
                "text_target_threshold": rec.get("target_threshold", threshold),
                "text_selection_threshold": float(threshold),
                "text_meets_target_threshold": rec.get(
                    "meets_target_threshold",
                    float(target_score) >= threshold,
                ),
            }
            row_indices_by_shard[shard_name].add(row_idx)
            source_counts[str(source or "unknown")] += 1

    stats = {
        "rows_scored": rows_scored,
        "candidate_rows": len(candidates_by_row_key),
        "candidate_shards": len(row_indices_by_shard),
        "source_counts": dict(source_counts),
    }
    return candidates_by_row_key, row_indices_by_shard, stats


def _to_float32_mono(wave: np.ndarray) -> np.ndarray:
    wave = np.asarray(wave)

    if wave.ndim == 2:
        if wave.shape[0] <= 8 and wave.shape[0] < wave.shape[1]:
            wave = wave.mean(axis=0)
        else:
            wave = wave.mean(axis=1)

    if wave.dtype.kind in {"i", "u"}:
        max_val = np.iinfo(wave.dtype).max
        wave = wave.astype(np.float32) / max_val
    else:
        wave = wave.astype(np.float32)

    wave = np.nan_to_num(wave, nan=0.0, posinf=0.0, neginf=0.0)
    return wave


def _looks_like_encoded_audio(audio_bytes: bytes) -> bool:
    prefixes = (
        b"RIFF",
        b"fLaC",
        b"OggS",
        b"ID3",
    )
    return any(audio_bytes.startswith(prefix) for prefix in prefixes)


def _decode_pcm16le_bytes(audio_bytes: bytes, sampling_rate: int) -> Tuple[np.ndarray, int]:
    wave = np.frombuffer(audio_bytes, dtype="<i2")
    if wave.size == 0:
        return np.zeros(0, dtype=np.float32), int(sampling_rate)
    wave = wave.astype(np.float32) / 32768.0
    return _to_float32_mono(wave), int(sampling_rate)


def _decode_audio_bytes(audio_bytes: bytes) -> Tuple[np.ndarray, int]:
    bio = io.BytesIO(audio_bytes)
    waveform, sr = torchaudio.load(bio)
    wave = waveform.mean(dim=0).cpu().numpy()
    return _to_float32_mono(wave), int(sr)


def _decode_audio_bytes_with_fallback(
    audio_bytes: bytes,
    row: Dict[str, Any],
    shard_path: Path,
) -> Tuple[np.ndarray, int, str]:
    sampling_rate = row.get("sampling_rate")
    if sampling_rate is not None and not _looks_like_encoded_audio(audio_bytes):
        return (*_decode_pcm16le_bytes(audio_bytes, int(sampling_rate)), "pcm16le.bytes")

    try:
        return (*_decode_audio_bytes(audio_bytes), "encoded.bytes")
    except Exception:
        if sampling_rate is not None:
            return (*_decode_pcm16le_bytes(audio_bytes, int(sampling_rate)), "pcm16le.bytes_fallback")
        raise


def _load_audio_path(path: str | Path, base_dir: Optional[Path] = None) -> Tuple[np.ndarray, int]:
    p = Path(path)
    if not p.is_absolute() and base_dir is not None:
        p = base_dir / p
    waveform, sr = torchaudio.load(str(p))
    wave = waveform.mean(dim=0).cpu().numpy()
    return _to_float32_mono(wave), int(sr)


def extract_audio(row: Dict[str, Any], shard_path: Path) -> Tuple[np.ndarray, int, str]:
    base_dir = shard_path.parent

    audio = row.get("audio")
    if isinstance(audio, dict):
        if audio.get("array") is not None:
            sr = int(audio.get("sampling_rate") or row.get("sampling_rate") or DEFAULT_TARGET_SR)
            return _to_float32_mono(np.asarray(audio["array"])), sr, "audio.array"

        if audio.get("bytes") is not None:
            wave, sr, source_kind = _decode_audio_bytes_with_fallback(bytes(audio["bytes"]), row, shard_path)
            return wave, sr, f"audio.bytes:{source_kind}"

        if audio.get("path"):
            return (*_load_audio_path(audio["path"], base_dir=base_dir), "audio.path")

    if isinstance(audio, (bytes, bytearray, memoryview)):
        wave, sr, source_kind = _decode_audio_bytes_with_fallback(bytes(audio), row, shard_path)
        return wave, sr, f"audio:{source_kind}"

    for key in ["bytes", "audio_bytes", "wav_bytes"]:
        val = row.get(key)
        if isinstance(val, (bytes, bytearray, memoryview)):
            wave, sr, source_kind = _decode_audio_bytes_with_fallback(bytes(val), row, shard_path)
            return wave, sr, f"{key}:{source_kind}"

    for key in ["path", "audio_path", "file", "filepath", "wav", "wav_path", "original_audio_path"]:
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

    preds = [{"label": id2label[i], "score": float(probs[i])} for i in range(len(probs))]
    preds.sort(key=lambda x: x["score"], reverse=True)
    return preds


def make_uid(shard_path: Path, row_idx: int, row: Dict[str, Any]) -> str:
    for key in ["uid", "id", "seg_id", "segment_id", "recording_id", "video_id"]:
        if row.get(key) is not None:
            return f"{shard_path.name}:{key}={row[key]}"
    return make_source_row_key(shard_path.name, row_idx)


def pick_text(row: Dict[str, Any]) -> Optional[str]:
    for key in [
        "manual_normalized_transcript",
        "normalized_transcript",
        "transcript",
        "text",
        "raw_text",
        "normalized",
        "sentence",
    ]:
        val = row.get(key)
        if isinstance(val, str) and val.strip():
            return val
    return None


def load_existing_state(predictions_path: Path) -> Tuple[set, Counter, int, int, int]:
    done_keys = set()
    counts = Counter()
    processed_ok = 0
    skipped = 0
    errors = 0

    if not predictions_path.exists():
        return done_keys, counts, processed_ok, skipped, errors

    with predictions_path.open("r", encoding="utf-8") as f:
        for line in f:
            try:
                rec = json.loads(line)
            except Exception:
                continue
            done_key = rec.get("source_row_key") or rec.get("uid")
            if done_key:
                done_keys.add(done_key)
            status = rec.get("status")
            if status == "ok":
                counts[rec.get("final_label", "unseen")] += 1
                processed_ok += 1
            elif status and status.startswith("skipped"):
                skipped += 1
            elif status == "error":
                errors += 1

    return done_keys, counts, processed_ok, skipped, errors


def pct(n: int, d: int) -> float:
    return 0.0 if d == 0 else round(100.0 * n / d, 4)


def _write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")


def write_progress(
    *,
    log_paths: List[Path],
    latest_json_paths: List[Path],
    summary_json_path: Path,
    counts: Counter,
    labels: List[str],
    processed_ok: int,
    skipped: int,
    errors: int,
    attempted: int,
    total_candidates: int,
    text_probabilities_path: str,
    text_target_label: str,
    text_target_threshold: float,
    current_file: str,
    current_uid: str,
    start_time: float,
) -> None:
    elapsed = time.time() - start_time
    label_order = list(labels) + ["unseen"]
    completed_total = processed_ok + skipped + errors

    percentages = {label: pct(counts.get(label, 0), processed_ok) for label in label_order}

    state = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "attempted_in_this_run": attempted,
        "completed_total": completed_total,
        "processed_ok_total": processed_ok,
        "skipped_total": skipped,
        "errors_total": errors,
        "total_candidates": total_candidates,
        "overall_progress_percentage": pct(completed_total, total_candidates),
        "text_probabilities_path": text_probabilities_path,
        "text_target_label": text_target_label,
        "text_target_threshold": text_target_threshold,
        "current_file": current_file,
        "current_uid": current_uid,
        "elapsed_seconds": round(elapsed, 2),
        "dialect_counts": {label: int(counts.get(label, 0)) for label in label_order},
        "dialect_percentages": percentages,
        "unseen_count": int(counts.get("unseen", 0)),
        "unseen_percentage": percentages.get("unseen", 0.0),
    }

    for latest_json_path in latest_json_paths:
        _write_json(latest_json_path, state)
    _write_json(summary_json_path, state)

    pct_bits = " ".join(f"{label}={percentages[label]:.2f}%" for label in label_order)
    line = (
        f"[{state['time']}] "
        f"ok={processed_ok} skipped={skipped} errors={errors} "
        f"progress={state['overall_progress_percentage']:.2f}% "
        f"unseen={percentages.get('unseen', 0.0):.2f}% "
        f"{pct_bits} "
        f"file={current_file} uid={current_uid}"
    )

    for log_path in log_paths:
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def build_probability_record(record: Dict[str, Any]) -> Dict[str, Any]:
    keys = [
        "uid",
        "source_row_key",
        "source_file",
        "row_idx",
        "source",
        "text",
        "status",
        "audio_source",
        "duration_sec",
        "top_label",
        "top_score",
        "final_label",
        "unknown_threshold",
        "predictions",
        "label_scores",
        "text_top_label",
        "text_top_score",
        "text_final_label",
        "text_unknown_threshold",
        "text_label_scores",
        "text_target_label",
        "text_target_label_score",
        "text_target_threshold",
        "text_selection_threshold",
        "text_meets_target_threshold",
        "reason",
        "error",
    ]
    return {key: record[key] for key in keys if key in record}


def run(args: argparse.Namespace) -> None:
    data_roots = [Path(p) for p in args.data_root]
    text_probabilities_path = Path(args.text_probabilities_path)
    output_dir = Path(args.output_dir)
    log_dir = output_dir / ".logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.jsonl"
    row_probabilities_path = output_dir / "row_probabilities.jsonl"
    progress_log_path = output_dir / "dialect_progress.log"
    progress_log_legacy_path = log_dir / "dialect_progress.log"
    latest_json_path = output_dir / "progress_latest.json"
    latest_json_legacy_path = log_dir / "progress_latest.json"
    summary_json_path = output_dir / "summary.json"

    if not text_probabilities_path.exists():
        raise FileNotFoundError(f"Text probabilities file not found: {text_probabilities_path}")

    candidates_by_row_key, row_indices_by_shard, candidate_stats = load_text_candidates(
        probabilities_path=text_probabilities_path,
        target_label=args.text_target_label,
        threshold=args.text_target_threshold,
    )
    if not candidates_by_row_key:
        raise RuntimeError(
            "No text candidates matched the requested threshold. "
            f"target_label={args.text_target_label} threshold={args.text_target_threshold}"
        )

    shards = discover_shards(data_roots)
    candidate_shard_names = set(row_indices_by_shard)
    shards = [path for path in shards if path.name in candidate_shard_names]
    if not shards:
        raise FileNotFoundError(
            "No candidate shards found under the requested data roots for the filtered text candidates: "
            f"{data_roots}"
        )

    print("Loaded text candidates:")
    print(json.dumps(candidate_stats, ensure_ascii=False, indent=2))
    print(f"Found {len(shards)} candidate shards under the requested data roots.")
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
        done_keys, counts, processed_ok, skipped, errors = load_existing_state(predictions_path)
        print(f"Resume enabled. Loaded {len(done_keys)} completed/skipped/error row keys.")
    else:
        done_keys, counts, processed_ok, skipped, errors = set(), Counter(), 0, 0, 0
        if predictions_path.exists():
            backup = predictions_path.with_suffix(".jsonl.bak")
            predictions_path.rename(backup)
            print(f"Existing predictions moved to: {backup}")
        if row_probabilities_path.exists():
            backup = row_probabilities_path.with_suffix(".jsonl.bak")
            row_probabilities_path.rename(backup)
            print(f"Existing row probabilities moved to: {backup}")

    start_time = time.time()
    attempted = 0
    total_candidates = len(candidates_by_row_key)

    with predictions_path.open("a", encoding="utf-8") as pred_out, row_probabilities_path.open("a", encoding="utf-8") as prob_out:
        stop = False

        for shard_path in shards:
            if stop:
                break

            print(f"\nScanning shard: {shard_path}")

            try:
                row_iter = iter_rows_from_file(shard_path, parquet_batch_size=args.parquet_batch_size)
                candidate_row_indices = row_indices_by_shard.get(shard_path.name, set())
                for row_idx, row in row_iter:
                    if row_idx not in candidate_row_indices:
                        continue

                    source_row_key = make_source_row_key(shard_path.name, row_idx)
                    if source_row_key in done_keys:
                        continue

                    text_candidate = candidates_by_row_key.get(source_row_key)
                    if text_candidate is None:
                        continue

                    if args.max_samples is not None and processed_ok >= args.max_samples:
                        stop = True
                        break

                    attempted += 1
                    record: Dict[str, Any] = {
                        "uid": text_candidate.get("uid") or make_uid(shard_path, row_idx, row),
                        "source_row_key": source_row_key,
                        "source_file": str(shard_path),
                        "row_idx": row_idx,
                        "source": text_candidate.get("source"),
                        "text": text_candidate.get("text") or pick_text(row),
                        "text_top_label": text_candidate.get("text_top_label"),
                        "text_top_score": text_candidate.get("text_top_score"),
                        "text_final_label": text_candidate.get("text_final_label"),
                        "text_unknown_threshold": text_candidate.get("text_unknown_threshold"),
                        "text_label_scores": text_candidate.get("text_label_scores"),
                        "text_target_label": text_candidate.get("text_target_label"),
                        "text_target_label_score": text_candidate.get("text_target_label_score"),
                        "text_target_threshold": text_candidate.get("text_target_threshold"),
                        "text_selection_threshold": text_candidate.get("text_selection_threshold"),
                        "text_meets_target_threshold": text_candidate.get("text_meets_target_threshold"),
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
                            label_scores = {str(pred["label"]): float(pred["score"]) for pred in preds}

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
                                "label_scores": label_scores,
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
                    prob_out.write(json.dumps(build_probability_record(record), ensure_ascii=False) + "\n")
                    prob_out.flush()
                    done_keys.add(source_row_key)

                    if args.log_every == 1 or attempted % args.log_every == 0:
                        write_progress(
                            log_paths=[progress_log_path, progress_log_legacy_path],
                            latest_json_paths=[latest_json_path, latest_json_legacy_path],
                            summary_json_path=summary_json_path,
                            counts=counts,
                            labels=labels,
                            processed_ok=processed_ok,
                            skipped=skipped,
                            errors=errors,
                            attempted=attempted,
                            total_candidates=total_candidates,
                            text_probabilities_path=str(text_probabilities_path),
                            text_target_label=args.text_target_label,
                            text_target_threshold=args.text_target_threshold,
                            current_file=str(shard_path),
                            current_uid=source_row_key,
                            start_time=start_time,
                        )

            except Exception as e:
                errors += 1
                err_rec = {
                    "uid": f"{shard_path.name}:SHARD_ERROR",
                    "source_row_key": f"{shard_path.name}:SHARD_ERROR",
                    "source_file": str(shard_path),
                    "status": "error",
                    "error": repr(e),
                    "traceback": traceback.format_exc(limit=5),
                }
                pred_out.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
                pred_out.flush()
                prob_out.write(json.dumps(build_probability_record(err_rec), ensure_ascii=False) + "\n")
                prob_out.flush()
                print(f"[ERROR] Could not scan shard {shard_path}: {repr(e)}")

    write_progress(
        log_paths=[progress_log_path, progress_log_legacy_path],
        latest_json_paths=[latest_json_path, latest_json_legacy_path],
        summary_json_path=summary_json_path,
        counts=counts,
        labels=labels,
        processed_ok=processed_ok,
        skipped=skipped,
        errors=errors,
        attempted=attempted,
        total_candidates=total_candidates,
        text_probabilities_path=str(text_probabilities_path),
        text_target_label=args.text_target_label,
        text_target_threshold=args.text_target_threshold,
        current_file="DONE",
        current_uid="DONE",
        start_time=start_time,
    )

    print("\nDone.")
    print(f"Predictions: {predictions_path}")
    print(f"Row probabilities: {row_probabilities_path}")
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
        default=(
            "/home/MohammadNabulsi/whisper/"
            "Runs/dialect_scan_badrex_mms300m_lev08_text_candidates_masc_c_qasr"
        ),
        help="Where outputs/logs are written.",
    )
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--target-sr", type=int, default=DEFAULT_TARGET_SR)
    parser.add_argument("--unknown-threshold", type=float, default=DEFAULT_UNKNOWN_THRESHOLD)
    parser.add_argument("--min-seconds", type=float, default=DEFAULT_MIN_SECONDS)
    parser.add_argument("--text-probabilities-path", default=DEFAULT_TEXT_PROBABILITIES_PATH)
    parser.add_argument("--text-target-label", default=DEFAULT_TEXT_TARGET_LABEL)
    parser.add_argument("--text-target-threshold", type=float, default=DEFAULT_TEXT_TARGET_THRESHOLD)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--parquet-batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")

    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
