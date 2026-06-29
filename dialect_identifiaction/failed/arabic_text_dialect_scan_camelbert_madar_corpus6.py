#!/usr/bin/env python3
"""
Arabic text dialect distribution scanner using:
  CAMeL-Lab/bert-base-arabic-camelbert-mix-did-madar-corpus6

What it does:
- Scans Parquet and Arrow shards.
- Extracts transcription text from common schemas.
- Runs the Hugging Face text-classification model locally.
- Writes:
    1) predictions.jsonl           -> one record per sample
    2) .logs/dialect_progress.log  -> human-readable live percentages
    3) .logs/progress_latest.json  -> machine-readable latest state
    4) summary.json                -> final/latest summary

Important:
- The model predicts MADAR Corpus 6 city labels.
- "unseen" here is NOT a trained class. It is a heuristic bucket:
  top model confidence < UNKNOWN_THRESHOLD.
"""

from __future__ import annotations

import argparse
import json
import time
import traceback
from collections import Counter
from pathlib import Path
from typing import Any, Dict, Iterator, List, Optional, Tuple

import pyarrow as pa
import pyarrow.ipc as ipc
import pyarrow.parquet as pq
import torch
from datasets import Dataset
from transformers import AutoModelForSequenceClassification, AutoTokenizer


DEFAULT_MODEL_ID = "CAMeL-Lab/bert-base-arabic-camelbert-mix-did-madar-corpus6"
DEFAULT_UNKNOWN_THRESHOLD = 0.55
TEXT_FIELD_CANDIDATES = [
    "normalized_transcript",
    "transcript",
    "text",
    "raw_text",
    "normalized",
    "sentence",
    "transcription",
    "prompt",
]


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
        except Exception:
            yield from _iter_arrow_ipc_rows(path)
            return

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


def write_progress(*, log_path: Path, latest_json_path: Path, summary_json_path: Path, counts: Counter, labels: List[str], processed_ok: int, skipped: int, errors: int, attempted: int, current_file: str, current_uid: str, start_time: float) -> None:
    elapsed = time.time() - start_time
    label_order = list(labels) + ["unseen"]
    percentages = {label: pct(counts.get(label, 0), processed_ok) for label in label_order}
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
    latest_json_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    summary_json_path.write_text(json.dumps(state, ensure_ascii=False, indent=2), encoding="utf-8")
    pct_bits = " ".join(f"{label}={percentages[label]:.2f}%" for label in label_order)
    line = (
        f"[{state['time']}] ok={processed_ok} skipped={skipped} errors={errors} "
        f"unseen={percentages.get('unseen', 0.0):.2f}% {pct_bits} "
        f"file={current_file} uid={current_uid}"
    )
    with log_path.open("a", encoding="utf-8") as f:
        f.write(line + "\n")


def load_model(model_id: str, device: str):
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    model = AutoModelForSequenceClassification.from_pretrained(model_id)
    model.to(device)
    model.eval()
    id2label = {int(k): v for k, v in model.config.id2label.items()}
    labels = [id2label[i] for i in sorted(id2label)]
    return tokenizer, model, id2label, labels


@torch.no_grad()
def predict_one(text: str, tokenizer, model, id2label: Dict[int, str], device: str, max_length: int) -> List[Dict[str, float | str]]:
    inputs = tokenizer(text, return_tensors="pt", truncation=True, max_length=max_length, padding=False)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    logits = model(**inputs).logits[0]
    probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
    preds = [{"label": id2label[i], "score": float(probs[i])} for i in range(len(probs))]
    preds.sort(key=lambda x: x["score"], reverse=True)
    return preds


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
    tokenizer, model, id2label, labels = load_model(args.model_id, device)
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
                    text, text_field = pick_text(row)
                    record = {"uid": uid, "source_file": str(shard_path), "row_idx": row_idx, "text": text, "text_field": text_field}
                    try:
                        if not text:
                            skipped += 1
                            record.update({"status": "skipped_no_text", "reason": f"No non-empty text found in: {TEXT_FIELD_CANDIDATES}"})
                        else:
                            preds = predict_one(text=text, tokenizer=tokenizer, model=model, id2label=id2label, device=device, max_length=args.max_length)
                            top = preds[0]
                            top_label = str(top["label"])
                            top_score = float(top["score"])
                            final_label = top_label if top_score >= args.unknown_threshold else "unseen"
                            counts[final_label] += 1
                            processed_ok += 1
                            record.update({"status": "ok", "text_length_chars": len(text), "top_label": top_label, "top_score": top_score, "final_label": final_label, "unknown_threshold": args.unknown_threshold, "predictions": preds})
                    except Exception as e:
                        errors += 1
                        record.update({"status": "error", "error": repr(e), "traceback": traceback.format_exc(limit=3)})
                    pred_out.write(json.dumps(record, ensure_ascii=False) + "\n")
                    pred_out.flush()
                    done_uids.add(uid)
                    if args.log_every == 1 or attempted % args.log_every == 0:
                        write_progress(log_path=progress_log_path, latest_json_path=latest_json_path, summary_json_path=summary_json_path, counts=counts, labels=labels, processed_ok=processed_ok, skipped=skipped, errors=errors, attempted=attempted, current_file=str(shard_path), current_uid=uid, start_time=start_time)
            except Exception as e:
                errors += 1
                err_rec = {"uid": f"{shard_path.name}:SHARD_ERROR", "source_file": str(shard_path), "status": "error", "error": repr(e), "traceback": traceback.format_exc(limit=5)}
                pred_out.write(json.dumps(err_rec, ensure_ascii=False) + "\n")
                pred_out.flush()
                print(f"[ERROR] Could not scan shard {shard_path}: {repr(e)}")
    write_progress(log_path=progress_log_path, latest_json_path=latest_json_path, summary_json_path=summary_json_path, counts=counts, labels=labels, processed_ok=processed_ok, skipped=skipped, errors=errors, attempted=attempted, current_file="DONE", current_uid="DONE", start_time=start_time)
    print("\nDone.")
    print(f"Predictions: {predictions_path}")
    print(f"Progress log: {progress_log_path}")
    print(f"Latest JSON: {latest_json_path}")
    print(f"Summary JSON: {summary_json_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-root", action="append", required=True, help="Directory or shard file to scan. Repeat for multiple roots.")
    parser.add_argument("--output-dir", default="/home/MohammadNabulsi/whisper/Runs/text_dialect_scan_camelbert_madar_corpus6", help="Where outputs/logs are written.")
    parser.add_argument("--model-id", default=DEFAULT_MODEL_ID)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--unknown-threshold", type=float, default=DEFAULT_UNKNOWN_THRESHOLD)
    parser.add_argument("--max-length", type=int, default=512)
    parser.add_argument("--max-samples", type=int, default=None)
    parser.add_argument("--log-every", type=int, default=1)
    parser.add_argument("--parquet-batch-size", type=int, default=64)
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    run(parse_args())
