#!/usr/bin/env python3
"""
Arabic written-dialect distribution scanner using:
  IbrahimAmin/marbertv2-arabic-written-dialect-classifier

What it does:
- Scans Parquet and Arrow shards.
- Extracts text from common schemas.
- Runs the Hugging Face text-classification model locally.
- Writes:
    1) predictions.jsonl            -> one record per sample
    2) row_probabilities.jsonl      -> per-row label probabilities for threshold experiments
    3) dialect_progress.log         -> human-readable live progress in the run folder
    4) progress_latest.json         -> machine-readable latest state in the run folder
    5) .logs/dialect_progress.log   -> legacy hidden copy for compatibility
    6) .logs/progress_latest.json   -> legacy hidden copy for compatibility
    7) summary.json                 -> final/latest summary

Important:
- The model predicts five broad written-dialect groups:
  MAGHREB, LEV, MSA, GLF, EGY.
- "unseen" here is NOT a trained class. It is a heuristic bucket:
  top model confidence < UNKNOWN_THRESHOLD.
- The dedicated row_probabilities.jsonl file is intended for later
  threshold sweeps, including LEV candidate mining, without rerunning inference.
"""

from __future__ import annotations

import argparse
import json
import shutil
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, TextIO, Tuple

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_ID = "IbrahimAmin/marbertv2-arabic-written-dialect-classifier"
DEFAULT_UNKNOWN_THRESHOLD = 0.55
DEFAULT_TARGET_LABEL = "LEV"
DEFAULT_TARGET_THRESHOLD = 0.4
DEFAULT_BATCH_SIZE_CUDA = 256
DEFAULT_BATCH_SIZE_CPU = 32
TEXT_FIELD_CANDIDATES = [
    "manual_normalized_transcript",
    "normalized_transcript",
    "transcript",
    "text",
    "raw_text",
    "normalized",
    "sentence",
    "transcription",
    "prompt",
]
SOURCE_FIELD_CANDIDATES = [
    "source",
    "source_name",
    "dataset_source",
    "dataset",
]
SOURCE_ALIAS_MAP = {
    "masc_c": {
        "masc_c",
        "masc-c",
        "masc_c_only",
        "masc",
    },
    "qasr": {
        "qasr",
        "processed_qasr_segments",
    },
}


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


def normalize_source_name(value: Optional[str]) -> Optional[str]:
    if value is None:
        return None
    normalized = str(value).strip().lower().replace("-", "_").replace(" ", "_")
    for canonical, aliases in SOURCE_ALIAS_MAP.items():
        if normalized == canonical or normalized in aliases:
            return canonical
    return normalized or None


def infer_source_from_row(row: Dict[str, Any]) -> Optional[str]:
    for key in SOURCE_FIELD_CANDIDATES:
        value = row.get(key)
        normalized = normalize_source_name(value if isinstance(value, str) else None)
        if normalized:
            return normalized

    source_file = row.get("source_file")
    if isinstance(source_file, str):
        lowered = source_file.lower()
        if "masc_c" in lowered:
            return "masc_c"
        if "qasr" in lowered or "processed_qasr_segments" in lowered:
            return "qasr"
    return None


def infer_source_from_path(path: Path) -> Optional[str]:
    lowered = path.name.lower()
    if lowered.startswith("masc_c_only__") or "masc_c" in lowered:
        return "masc_c"
    if lowered.startswith("processed_qasr_segments__") or "qasr" in lowered:
        return "qasr"
    return None


def _iter_arrow_ipc_rows(path: Path) -> Iterator[Tuple[int, Dict[str, Any]]]:
    with pa.memory_map(str(path), "r") as source:
        try:
            reader = ipc.open_file(source)
            table = reader.read_all()
        except Exception:
            source.seek(0)
            reader = ipc.open_stream(source)
            table = reader.read_all()

    for i, row in enumerate(table.to_pylist()):
        yield i, row


def iter_rows_from_file(path: Path, parquet_batch_size: int = 256) -> Iterator[Tuple[int, Dict[str, Any]]]:
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
        except Exception:
            yield from _iter_arrow_ipc_rows(path)
            return

    raise ValueError(f"Unsupported file suffix: {path}")


def count_rows_in_file(path: Path) -> int:
    suffix = path.suffix.lower()

    if suffix == ".parquet":
        return int(pq.ParquetFile(path).metadata.num_rows)

    if suffix == ".arrow":
        try:
            ds = Dataset.from_file(str(path))
            return int(ds.num_rows)
        except Exception:
            with pa.memory_map(str(path), "r") as source:
                try:
                    reader = ipc.open_file(source)
                    return int(reader.read_all().num_rows)
                except Exception:
                    source.seek(0)
                    reader = ipc.open_stream(source)
                    return int(reader.read_all().num_rows)

    raise ValueError(f"Unsupported file suffix: {path}")


def make_uid(shard_path: Path, row_idx: int, row: Dict[str, Any]) -> str:
    for key in ["uid", "id", "seg_id", "segment_id", "recording_id", "video_id"]:
        if row.get(key) is not None:
            return f"{shard_path.name}:{key}={row[key]}"
    return f"{shard_path.name}:row={row_idx}"


def pick_text(row: Dict[str, Any]) -> Tuple[Optional[str], Optional[str]]:
    for key in TEXT_FIELD_CANDIDATES:
        val = row.get(key)
        if isinstance(val, str):
            cleaned = " ".join(val.strip().split())
            if cleaned:
                return cleaned, key
    return None, None


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


def format_duration(seconds: Optional[float]) -> Optional[str]:
    if seconds is None:
        return None
    seconds = max(0, int(seconds))
    hours, rem = divmod(seconds, 3600)
    minutes, secs = divmod(rem, 60)
    if hours:
        return f"{hours:d}h{minutes:02d}m{secs:02d}s"
    if minutes:
        return f"{minutes:d}m{secs:02d}s"
    return f"{secs:d}s"


def sync_visible_progress_artifacts(
    *,
    legacy_progress_log_path: Path,
    visible_progress_log_path: Path,
    legacy_latest_json_path: Path,
    visible_latest_json_path: Path,
) -> None:
    if legacy_progress_log_path.exists() and not visible_progress_log_path.exists():
        shutil.copy2(legacy_progress_log_path, visible_progress_log_path)
    if legacy_latest_json_path.exists() and not visible_latest_json_path.exists():
        shutil.copy2(legacy_latest_json_path, visible_latest_json_path)


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
    total_rows: int,
    completed_before_run: int,
    current_file: str,
    current_uid: str,
    start_time: float,
) -> None:
    elapsed = time.time() - start_time
    label_order = list(labels) + ["unseen"]
    completed_total = processed_ok + skipped + errors
    newly_completed = max(completed_total - completed_before_run, 0)
    remaining_rows = max(total_rows - completed_total, 0)
    rows_per_second = round(newly_completed / elapsed, 2) if elapsed > 0 else 0.0
    eta_seconds = (remaining_rows / rows_per_second) if rows_per_second > 0 else None

    percentages = {
        label: pct(counts.get(label, 0), processed_ok)
        for label in label_order
    }

    state = {
        "time": time.strftime("%Y-%m-%d %H:%M:%S"),
        "attempted_in_this_run": attempted,
        "completed_total": completed_total,
        "completed_before_run": completed_before_run,
        "total_rows": total_rows,
        "overall_progress_percentage": pct(completed_total, total_rows),
        "processed_ok_total": processed_ok,
        "skipped_total": skipped,
        "errors_total": errors,
        "current_file": current_file,
        "current_uid": current_uid,
        "elapsed_seconds": round(elapsed, 2),
        "rows_per_second": rows_per_second,
        "estimated_remaining_seconds": round(eta_seconds, 2) if eta_seconds is not None else None,
        "estimated_remaining_human": format_duration(eta_seconds),
        "dialect_counts": {label: int(counts.get(label, 0)) for label in label_order},
        "dialect_percentages": percentages,
        "unseen_count": int(counts.get("unseen", 0)),
        "unseen_percentage": percentages.get("unseen", 0.0),
    }

    payload = json.dumps(state, ensure_ascii=False, indent=2)
    for latest_json_path in list(dict.fromkeys(latest_json_paths)):
        latest_json_path.write_text(payload, encoding="utf-8")
    summary_json_path.write_text(payload, encoding="utf-8")

    pct_bits = " ".join(f"{label}={percentages[label]:.2f}%" for label in label_order)
    line = (
        f"[{state['time']}] "
        f"progress={state['overall_progress_percentage']:.2f}% "
        f"completed={completed_total}/{total_rows} "
        f"ok={processed_ok} skipped={skipped} errors={errors} "
        f"rate={rows_per_second:.2f} rows/s "
        f"eta={state['estimated_remaining_human'] or 'n/a'} "
        f"unseen={percentages.get('unseen', 0.0):.2f}% "
        f"{pct_bits} "
        f"file={current_file} uid={current_uid}"
    )

    for log_path in list(dict.fromkeys(log_paths)):
        with log_path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def classify_texts(
    texts: List[str],
    *,
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    id2label: Dict[int, str],
    device: str,
    max_length: int,
) -> List[List[Dict[str, float | str]]]:
    inputs = tokenizer(
        texts,
        return_tensors="pt",
        truncation=True,
        max_length=max_length,
        padding=True,
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    use_autocast = device == "cuda"
    autocast_dtype = torch.bfloat16 if use_autocast and torch.cuda.is_bf16_supported() else torch.float16

    with torch.inference_mode():
        if use_autocast:
            with torch.autocast(device_type="cuda", dtype=autocast_dtype):
                logits = model(**inputs).logits
        else:
            logits = model(**inputs).logits
        prob_rows = torch.softmax(logits.float(), dim=-1).cpu().tolist()

    batched_preds: List[List[Dict[str, float | str]]] = []
    for probs in prob_rows:
        preds = [
            {"label": str(id2label[i]), "score": float(probs[i])}
            for i in range(len(probs))
        ]
        preds.sort(key=lambda x: x["score"], reverse=True)
        batched_preds.append(preds)
    return batched_preds


def is_cuda_oom(exc: Exception) -> bool:
    msg = str(exc).lower()
    return "out of memory" in msg or "cuda error: out of memory" in msg


def build_probability_record(
    *,
    base_record: Dict[str, Any],
    text: str,
    text_field: str,
    preds: List[Dict[str, float | str]],
    top_label: str,
    top_score: float,
    final_label: str,
    unknown_threshold: float,
    target_label: Optional[str],
    target_threshold: Optional[float],
) -> Dict[str, Any]:
    label_scores = {
        str(pred["label"]): float(pred["score"])
        for pred in preds
    }
    target_label_score = label_scores.get(target_label) if target_label else None
    meets_target_threshold = (
        target_label_score is not None and target_threshold is not None and target_label_score >= target_threshold
    )

    return {
        **base_record,
        "text_field": text_field,
        "text": text,
        "top_label": top_label,
        "top_score": top_score,
        "final_label": final_label,
        "unknown_threshold": unknown_threshold,
        "predictions": preds,
        "label_scores": label_scores,
        "target_label": target_label,
        "target_label_score": target_label_score,
        "target_threshold": target_threshold,
        "meets_target_threshold": meets_target_threshold,
    }


def flush_pending_items(
    *,
    pending_items: List[Dict[str, Any]],
    tokenizer: AutoTokenizer,
    model: AutoModelForSequenceClassification,
    id2label: Dict[int, str],
    device: str,
    max_length: int,
    unknown_threshold: float,
    target_label: Optional[str],
    target_threshold: Optional[float],
    pred_out: TextIO,
    prob_out: TextIO,
    done_uids: set,
    counts: Counter,
    processed_ok: int,
    errors: int,
    batch_size: int,
) -> Tuple[int, int, int, Optional[str]]:
    if not pending_items:
        return processed_ok, errors, batch_size, None

    effective_batch_size = max(1, batch_size)
    idx = 0
    last_uid: Optional[str] = None

    while idx < len(pending_items):
        chunk = pending_items[idx:idx + effective_batch_size]
        try:
            batched_preds = classify_texts(
                [str(item["text"]) for item in chunk],
                tokenizer=tokenizer,
                model=model,
                id2label=id2label,
                device=device,
                max_length=max_length,
            )
        except RuntimeError as e:
            if device == "cuda" and is_cuda_oom(e) and effective_batch_size > 1:
                new_batch_size = max(1, effective_batch_size // 2)
                print(f"[WARN] CUDA OOM at batch_size={effective_batch_size}; retrying with batch_size={new_batch_size}")
                torch.cuda.empty_cache()
                effective_batch_size = new_batch_size
                continue

            if len(chunk) > 1:
                print(f"[WARN] Batch inference failed for {len(chunk)} rows; retrying individually. Error: {repr(e)}")
                effective_batch_size = 1
                continue

            errors += 1
            item = chunk[0]
            record = {
                **item["base_record"],
                "status": "error",
                "error": repr(e),
                "traceback": traceback.format_exc(limit=3),
            }
            pred_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            pred_out.flush()
            done_uids.add(item["uid"])
            last_uid = str(item["uid"])
            idx += 1
            if device == "cuda":
                torch.cuda.empty_cache()
            continue
        except Exception as e:
            if len(chunk) > 1:
                print(f"[WARN] Batch inference failed for {len(chunk)} rows; retrying individually. Error: {repr(e)}")
                effective_batch_size = 1
                continue

            errors += 1
            item = chunk[0]
            record = {
                **item["base_record"],
                "status": "error",
                "error": repr(e),
                "traceback": traceback.format_exc(limit=3),
            }
            pred_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            pred_out.flush()
            done_uids.add(item["uid"])
            last_uid = str(item["uid"])
            idx += 1
            continue

        for item, preds in zip(chunk, batched_preds):
            top = preds[0]
            top_label = str(top["label"])
            top_score = float(top["score"])
            final_label = top_label if top_score >= unknown_threshold else "unseen"

            counts[final_label] += 1
            processed_ok += 1

            probability_record = build_probability_record(
                base_record=item["base_record"],
                text=str(item["text"]),
                text_field=str(item["text_field"]),
                preds=preds,
                top_label=top_label,
                top_score=top_score,
                final_label=final_label,
                unknown_threshold=unknown_threshold,
                target_label=target_label,
                target_threshold=target_threshold,
            )
            record = {
                **item["base_record"],
                "status": "ok",
                **{k: v for k, v in probability_record.items() if k not in item["base_record"]},
            }

            prob_out.write(json.dumps(probability_record, ensure_ascii=False) + "\n")
            pred_out.write(json.dumps(record, ensure_ascii=False) + "\n")
            done_uids.add(item["uid"])
            last_uid = str(item["uid"])

        prob_out.flush()
        pred_out.flush()
        idx += len(chunk)

    return processed_ok, errors, effective_batch_size, last_uid


def infer_device(requested_device: str) -> str:
    if requested_device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return requested_device


def resolve_batch_size(requested_batch_size: Optional[int], device: str) -> int:
    if requested_batch_size and requested_batch_size > 0:
        return requested_batch_size
    return DEFAULT_BATCH_SIZE_CUDA if device == "cuda" else DEFAULT_BATCH_SIZE_CPU


def run(args: argparse.Namespace) -> None:
    data_roots = [Path(p) for p in args.data_root]
    allowed_sources = {
        normalized
        for normalized in (normalize_source_name(value) for value in args.allowed_source)
        if normalized
    }
    output_dir = Path(args.output_dir)
    log_dir = output_dir / ".logs"
    output_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)

    predictions_path = output_dir / "predictions.jsonl"
    probabilities_path = output_dir / "row_probabilities.jsonl"
    progress_log_path = log_dir / "dialect_progress.log"
    visible_progress_log_path = output_dir / "dialect_progress.log"
    latest_json_path = log_dir / "progress_latest.json"
    visible_latest_json_path = output_dir / "progress_latest.json"
    summary_json_path = output_dir / "summary.json"

    shards = discover_shards(data_roots)
    if allowed_sources:
        shards = [path for path in shards if infer_source_from_path(path) in allowed_sources]
        print(f"Applied source filter: {sorted(allowed_sources)}")
    print(f"Found {len(shards)} shards.")
    for p in shards[:10]:
        print(" -", p)
    if len(shards) > 10:
        print(f" ... plus {len(shards) - 10} more")

    if not shards:
        raise SystemExit("No parquet/arrow shards found.")

    sync_visible_progress_artifacts(
        legacy_progress_log_path=progress_log_path,
        visible_progress_log_path=visible_progress_log_path,
        legacy_latest_json_path=latest_json_path,
        visible_latest_json_path=visible_latest_json_path,
    )

    print("Counting total rows across shards for overall progress tracking...")
    total_rows = sum(count_rows_in_file(path) for path in shards)
    print(f"Total rows discovered: {total_rows}")
    print(f"Visible progress log: {visible_progress_log_path}")
    print(f"Visible latest status: {visible_latest_json_path}")

    print(f"Loading model: {args.model_id}")
    tokenizer = AutoTokenizer.from_pretrained(args.model_id)
    model = AutoModelForSequenceClassification.from_pretrained(args.model_id)

    device = infer_device(args.device)
    print("Device:", device)
    if device == "cuda":
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    model = model.to(device)
    model.eval()
    batch_size = resolve_batch_size(args.batch_size, device)
    print(f"Inference batch size: {batch_size}")

    raw_id2label = model.config.id2label
    if isinstance(next(iter(raw_id2label.keys())), str):
        id2label = {int(k): v for k, v in raw_id2label.items()}
    else:
        id2label = raw_id2label
    labels = [str(id2label[i]) for i in sorted(id2label.keys())]
    print("Model labels:", labels)
    print("Target label for later thresholding:", args.target_label)

    if args.resume:
        done_uids, counts, processed_ok, skipped, errors = load_existing_state(predictions_path)
        print(f"Resume enabled. Loaded {len(done_uids)} completed/skipped/error UIDs.")
    else:
        done_uids, counts, processed_ok, skipped, errors = set(), Counter(), 0, 0, 0

    attempted = 0
    start_time = time.time()
    completed_before_run = processed_ok + skipped + errors
    next_log_at = completed_before_run + max(1, args.log_every)

    def maybe_write_progress(current_file: str, current_uid: str, force: bool = False) -> None:
        nonlocal next_log_at
        completed_total = processed_ok + skipped + errors
        if not force and completed_total < next_log_at:
            return
        write_progress(
            log_paths=[progress_log_path, visible_progress_log_path],
            latest_json_paths=[latest_json_path, visible_latest_json_path],
            summary_json_path=summary_json_path,
            counts=counts,
            labels=labels,
            processed_ok=processed_ok,
            skipped=skipped,
            errors=errors,
            attempted=attempted,
            total_rows=total_rows,
            completed_before_run=completed_before_run,
            current_file=current_file,
            current_uid=current_uid,
            start_time=start_time,
        )
        while next_log_at <= completed_total:
            next_log_at += max(1, args.log_every)

    with predictions_path.open("a", encoding="utf-8") as pred_out, probabilities_path.open("a", encoding="utf-8") as prob_out:
        for shard_path in shards:
            print(f"\nScanning shard: {shard_path}")
            pending_items: List[Dict[str, Any]] = []
            try:
                for row_idx, row in iter_rows_from_file(shard_path):
                    if args.max_samples is not None and attempted >= args.max_samples:
                        if pending_items:
                            processed_ok, errors, batch_size, last_uid = flush_pending_items(
                                pending_items=pending_items,
                                tokenizer=tokenizer,
                                model=model,
                                id2label=id2label,
                                device=device,
                                max_length=args.max_length,
                                unknown_threshold=args.unknown_threshold,
                                target_label=args.target_label,
                                target_threshold=args.target_threshold,
                                pred_out=pred_out,
                                prob_out=prob_out,
                                done_uids=done_uids,
                                counts=counts,
                                processed_ok=processed_ok,
                                errors=errors,
                                batch_size=batch_size,
                            )
                            pending_items = []
                            maybe_write_progress(str(shard_path), last_uid or f"{shard_path.name}:batch_flush")
                        print("Reached max_samples; stopping.")
                        maybe_write_progress(str(shard_path), f"{shard_path.name}:row={row_idx}", force=True)
                        return

                    uid = make_uid(shard_path, row_idx, row)
                    if uid in done_uids:
                        continue

                    attempted += 1
                    row_source = infer_source_from_row(row) or infer_source_from_path(shard_path)
                    base_record: Dict[str, Any] = {
                        "uid": uid,
                        "source_file": str(shard_path),
                        "row_idx": row_idx,
                        "source": row_source,
                    }

                    try:
                        if allowed_sources and row_source not in allowed_sources:
                            skipped += 1
                            record = {
                                **base_record,
                                "status": "skipped_filtered_source",
                                "allowed_sources": sorted(allowed_sources),
                            }
                            pred_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                            pred_out.flush()
                            done_uids.add(uid)
                            maybe_write_progress(str(shard_path), uid)
                            continue
                        text, text_field = pick_text(row)
                        if text is None:
                            skipped += 1
                            record = {
                                **base_record,
                                "status": "skipped_no_text",
                                "available_keys": sorted(list(row.keys()))[:100],
                            }
                            pred_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                            pred_out.flush()
                            done_uids.add(uid)
                            maybe_write_progress(str(shard_path), uid)
                        else:
                            pending_items.append({
                                "uid": uid,
                                "base_record": base_record,
                                "text": text,
                                "text_field": str(text_field),
                            })
                            if len(pending_items) >= batch_size:
                                processed_ok, errors, batch_size, last_uid = flush_pending_items(
                                    pending_items=pending_items,
                                    tokenizer=tokenizer,
                                    model=model,
                                    id2label=id2label,
                                    device=device,
                                    max_length=args.max_length,
                                    unknown_threshold=args.unknown_threshold,
                                    target_label=args.target_label,
                                    target_threshold=args.target_threshold,
                                    pred_out=pred_out,
                                    prob_out=prob_out,
                                    done_uids=done_uids,
                                    counts=counts,
                                    processed_ok=processed_ok,
                                    errors=errors,
                                    batch_size=batch_size,
                                )
                                pending_items = []
                                maybe_write_progress(str(shard_path), last_uid or uid)

                    except Exception as e:
                        errors += 1
                        record = {
                            **base_record,
                            "status": "error",
                            "error": repr(e),
                            "traceback": traceback.format_exc(limit=3),
                        }
                        pred_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                        pred_out.flush()
                        done_uids.add(uid)
                        maybe_write_progress(str(shard_path), uid)

                if pending_items:
                    processed_ok, errors, batch_size, last_uid = flush_pending_items(
                        pending_items=pending_items,
                        tokenizer=tokenizer,
                        model=model,
                        id2label=id2label,
                        device=device,
                        max_length=args.max_length,
                        unknown_threshold=args.unknown_threshold,
                        target_label=args.target_label,
                        target_threshold=args.target_threshold,
                        pred_out=pred_out,
                        prob_out=prob_out,
                        done_uids=done_uids,
                        counts=counts,
                        processed_ok=processed_ok,
                        errors=errors,
                        batch_size=batch_size,
                    )
                    pending_items = []
                    maybe_write_progress(str(shard_path), last_uid or f"{shard_path.name}:batch_flush")

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
                maybe_write_progress(str(shard_path), err_rec["uid"], force=True)

    maybe_write_progress("DONE", "DONE", force=True)
    print("\nCompleted.")
    print(summary_json_path.read_text(encoding="utf-8"))


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description="Arabic written-dialect scanner")
    ap.add_argument(
        "--data-root",
        action="append",
        required=True,
        help="Directory or specific .parquet/.arrow file. Can be passed multiple times.",
    )
    ap.add_argument("--output-dir", required=True)
    ap.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    ap.add_argument("--unknown-threshold", type=float, default=DEFAULT_UNKNOWN_THRESHOLD)
    ap.add_argument("--target-label", default=DEFAULT_TARGET_LABEL)
    ap.add_argument("--target-threshold", type=float, default=DEFAULT_TARGET_THRESHOLD)
    ap.add_argument("--max-length", type=int, default=512)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--log-every", type=int, default=250, help="Write progress after this many completed rows.")
    ap.add_argument("--batch-size", type=int, default=None, help="Inference batch size. Defaults to 256 on CUDA, 32 on CPU.")
    ap.add_argument("--allowed-source", action="append", default=[], help="Limit processing to rows/files from these sources, e.g. masc_c or qasr.")
    ap.add_argument("--max-samples", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    return ap


if __name__ == "__main__":
    args = build_parser().parse_args()
    run(args)
