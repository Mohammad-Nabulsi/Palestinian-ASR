"""Resumable runner: zero-shot Whisper-large-v3 predictions + WER/CER for one
or all datasets discovered by datasets.discover_eval_targets().

Resume model: predictions are appended to outputs/<run>/<dataset>/predictions.jsonl
as soon as each batch finishes (flush + fsync). On restart, uids already present
in that file are skipped, so a crash/kill loses at most one in-flight batch.

Usage:
    python run_eval.py --all
    python run_eval.py --dataset masc_c_only --batch-size 8
    python run_eval.py --dataset omnilingual_apc_full --limit 20   # smoke test
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from audio_io import decode_audio_cell  # noqa: E402
from datasets import discover_eval_targets  # noqa: E402
from evaluate import evaluate  # noqa: E402
from whisper_predict import WhisperPredictConfig, WhisperPredictor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "whisper_large_v3_zero_shot"
PROGRESS_LOG_INTERVAL_S = 120  # log progress every 2 minutes, per instructions
REF_COLUMN = "manual_normalized_transcript"


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"zero_shot_eval.{log_path.stem}")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()
    fmt = logging.Formatter("%(asctime)s | %(levelname)s | %(message)s")
    fh = logging.FileHandler(log_path, mode="a")
    fh.setFormatter(fmt)
    logger.addHandler(fh)
    sh = logging.StreamHandler(sys.stdout)
    sh.setFormatter(fmt)
    logger.addHandler(sh)
    return logger


def load_done_uids(predictions_path: Path) -> set[str]:
    if not predictions_path.exists():
        return set()
    done = set()
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                done.add(json.loads(line)["uid"])
            except (json.JSONDecodeError, KeyError):
                continue  # tolerate a truncated last line from a hard kill
    return done


def iter_rows(shard_paths: list[Path]):
    for shard in shard_paths:
        df = pd.read_parquet(shard)
        for row_idx, row in df.iterrows():
            uid = f"{shard.name}::{row_idx}"
            yield uid, shard.name, row
        del df
        gc.collect()


def run_dataset(
    dataset_key: str,
    shard_paths: list[Path],
    predictor: WhisperPredictor,
    out_dir: Path,
    batch_size: int,
    limit: int | None,
    logger: logging.Logger,
) -> Path:
    dataset_dir = out_dir / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = dataset_dir / "predictions.jsonl"
    metrics_path = dataset_dir / "metrics.json"

    done_uids = load_done_uids(predictions_path)
    logger.info("[%s] resume: %d rows already done", dataset_key, len(done_uids))

    total_seen = 0
    total_new = 0
    t_start = time.time()
    t_last_log = t_start

    with open(predictions_path, "a", encoding="utf-8") as out_f:
        batch_uid, batch_audio, batch_ref, batch_meta = [], [], [], []

        def flush_batch():
            nonlocal total_new
            if not batch_uid:
                return
            t0 = time.time()
            hyps = predictor.predict(batch_audio)
            infer_s = time.time() - t0
            for uid, ref, meta, hyp in zip(batch_uid, batch_ref, batch_meta, hyps):
                rec = {
                    "uid": uid,
                    "shard": meta["shard"],
                    "row_idx": meta["row_idx"],
                    "reference": ref,
                    "hypothesis": hyp,
                    "duration_s": meta["duration_s"],
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            total_new += len(batch_uid)
            logger.debug("[%s] batch of %d transcribed in %.1fs", dataset_key, len(batch_uid), infer_s)
            batch_uid.clear()
            batch_audio.clear()
            batch_ref.clear()
            batch_meta.clear()

        for uid, shard_name, row in iter_rows(shard_paths):
            if limit is not None and total_seen >= limit:
                break
            total_seen += 1
            if uid in done_uids:
                continue
            try:
                audio = decode_audio_cell(row["audio"])
            except Exception:
                logger.exception("[%s] failed to decode audio for %s; skipping", dataset_key, uid)
                continue

            ref = row.get(REF_COLUMN, "") if hasattr(row, "get") else row[REF_COLUMN]
            batch_uid.append(uid)
            batch_audio.append(audio)
            batch_ref.append("" if ref is None else str(ref))
            batch_meta.append({
                "shard": shard_name,
                "row_idx": int(uid.rsplit("::", 1)[1]),
                "duration_s": float(row.get("duration", 0.0)) if hasattr(row, "get") else None,
            })

            if len(batch_uid) >= batch_size:
                flush_batch()

            now = time.time()
            if now - t_last_log >= PROGRESS_LOG_INTERVAL_S:
                elapsed = now - t_start
                rate = total_new / elapsed if elapsed > 0 else 0.0
                mem = (
                    f"{torch.cuda.memory_allocated() / 1e9:.2f}GB alloc / "
                    f"{torch.cuda.memory_reserved() / 1e9:.2f}GB reserved"
                    if torch.cuda.is_available() else "cpu"
                )
                logger.info(
                    "[%s] progress: %d new / %d seen (+%d resumed) | %.2f rows/s | elapsed=%.0fs | gpu=%s",
                    dataset_key, total_new, total_seen, len(done_uids), rate, elapsed, mem,
                )
                t_last_log = now

        flush_batch()

    logger.info(
        "[%s] prediction pass done: %d new rows written (%d total incl. resumed) in %.0fs",
        dataset_key, total_new, total_new + len(done_uids), time.time() - t_start,
    )

    # ---- evaluate ----
    refs, hyps = [], []
    with open(predictions_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            rec = json.loads(line)
            refs.append(rec["reference"])
            hyps.append(rec["hypothesis"])

    metrics = evaluate(refs, hyps)
    summary = {k: v for k, v in metrics.items() if k != "per_utterance"}
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info("[%s] metrics: %s", dataset_key, summary)
    return metrics_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="dataset key from datasets.discover_eval_targets(); omit with --all")
    ap.add_argument("--all", action="store_true", help="run every discovered dataset")
    ap.add_argument("--model-id", default="openai/whisper-large-v3")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--limit", type=int, default=None, help="cap rows per dataset (smoke testing)")
    ap.add_argument("--out-dir", default=str(DEFAULT_OUT_DIR))
    args = ap.parse_args()

    if not args.dataset and not args.all:
        ap.error("pass --dataset <key> or --all")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "run.log")

    targets = discover_eval_targets()
    if args.all:
        keys = list(targets.keys())
    else:
        if args.dataset not in targets:
            ap.error(f"unknown dataset {args.dataset!r}; choices: {sorted(targets)}")
        keys = [args.dataset]

    config = WhisperPredictConfig(model_id=args.model_id, batch_size=args.batch_size)
    predictor = WhisperPredictor(config)

    all_metrics = {}
    for key in keys:
        metrics_path = run_dataset(key, targets[key], predictor, out_dir, args.batch_size, args.limit, logger)
        all_metrics[key] = json.loads(metrics_path.read_text(encoding="utf-8"))

    summary_path = out_dir / "summary.json"
    summary = {
        k: {kk: vv for kk, vv in v.items() if kk != "per_utterance"} for k, v in all_metrics.items()
    }
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(summary, f, ensure_ascii=False, indent=2)
    logger.info("Summary written to %s: %s", summary_path, summary)


if __name__ == "__main__":
    main()
