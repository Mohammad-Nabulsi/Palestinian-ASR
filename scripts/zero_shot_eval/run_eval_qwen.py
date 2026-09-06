"""Resumable runner: zero-shot Qwen/Qwen3-ASR-0.6B-hf predictions + WER/CER for
one or all datasets discovered by datasets.discover_eval_targets() -- the Qwen3
sibling of run_eval.py (Whisper) / run_eval_cohere.py (Cohere), reusing the same
datasets/audio_io/evaluate APIs so results are directly comparable.

Batching: HF-style length-grouped batching, same idea as transformers'
LengthGroupedSampler and identical in shape to run_eval_cohere.py -- rows are
sorted by the `duration` column (no audio decode needed for that) so each batch
holds similarly-sized clips, then greedily packed under both a row-count cap and
a summed raw-duration budget. Qwen3-ASR's processor already pads each batch
dynamically to that batch's own longest clip (verified empirically, see
qwen_predict.py), so length-grouping is what keeps that dynamic padding cheap --
without it, one long clip in an otherwise-short batch would drag every row up to
its length.

The duration budget here (DEFAULT_MAX_BATCH_AUDIO_S) is sized off direct
benchmarking on this GPU (RTX PRO 4000 Blackwell, 24GB): Qwen3-ASR-0.6B is small
enough that even a summed 1600s/batch (e.g. 24 rows x ~68s) peaked at ~13GB, well
under budget, so the default here is deliberately more generous than Cohere's
(which is a much bigger model). QwenPredictor's own OOM backoff is the final
safety net regardless.

Resume model: identical to run_eval.py / run_eval_cohere.py -- predictions are
appended to outputs/<run>/<dataset>/predictions.jsonl as soon as each batch
finishes (flush + fsync); uids already present are skipped on restart.

Usage:
    python run_eval_qwen.py --all
    python run_eval_qwen.py --dataset masc_c_only --max-rows 64
    python run_eval_qwen.py --dataset omnilingual_apc_full --limit 20   # smoke test
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
from qwen_predict import QwenPredictConfig, QwenPredictor  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUT_DIR = REPO_ROOT / "outputs" / "qwen3_asr_0_6b_zero_shot"
PROGRESS_LOG_INTERVAL_S = 120  # log progress every 2 minutes, per instructions
REF_COLUMN_CANDIDATES = ("manual_normalized_transcript", "transcription", "text", "raw_text")

# Length-grouped batching knobs -- see module docstring for how these were sized.
DEFAULT_MAX_ROWS = 64
DEFAULT_MAX_BATCH_AUDIO_S = 900.0


def resolve_ref_column(columns) -> str:
    for c in REF_COLUMN_CANDIDATES:
        if c in columns:
            return c
    raise ValueError(f"no ground-truth column found; tried {REF_COLUMN_CANDIDATES}, have {list(columns)}")


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger(f"zero_shot_eval_qwen.{log_path.stem}")
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


def load_dataset_rows(shard_paths: list[Path]) -> list[dict]:
    """Load every shard fully into memory (small: <2GB per dataset here) and
    return per-row metadata + a handle back to the row for lazy audio decode.
    """
    rows = []
    for shard in shard_paths:
        df = pd.read_parquet(shard)
        ref_column = resolve_ref_column(df.columns)
        for row_idx, row in df.iterrows():
            duration = float(row["duration"]) if "duration" in df.columns and row["duration"] is not None else 0.0
            ref = row.get(ref_column, "")
            rows.append({
                "uid": f"{shard.name}::{row_idx}",
                "shard": shard.name,
                "row_idx": int(row_idx),
                "duration": duration,
                "ref": "" if ref is None else str(ref),
                "audio_cell": row["audio"],
            })
        del df
        gc.collect()
    return rows


def build_length_grouped_batches(
    rows: list[dict], max_rows: int, max_batch_audio_s: float
) -> list[list[dict]]:
    """Sort by duration (length-grouping, HF LengthGroupedSampler style) then
    greedily pack into batches capped by both row count and raw-duration
    budget (bounds the batch's longest clip, which the processor pads every
    row in the batch up to)."""
    ordered = sorted(rows, key=lambda r: r["duration"])
    batches: list[list[dict]] = []
    cur: list[dict] = []
    cur_dur = 0.0
    for r in ordered:
        would_dur = cur_dur + r["duration"]
        if cur and (len(cur) >= max_rows or would_dur > max_batch_audio_s):
            batches.append(cur)
            cur, cur_dur = [], 0.0
        cur.append(r)
        cur_dur += r["duration"]
    if cur:
        batches.append(cur)
    return batches


def run_dataset(
    dataset_key: str,
    shard_paths: list[Path],
    predictor: QwenPredictor,
    out_dir: Path,
    max_rows: int,
    max_batch_audio_s: float,
    limit: int | None,
    logger: logging.Logger,
) -> Path:
    dataset_dir = out_dir / dataset_key
    dataset_dir.mkdir(parents=True, exist_ok=True)
    predictions_path = dataset_dir / "predictions.jsonl"
    metrics_path = dataset_dir / "metrics.json"

    done_uids = load_done_uids(predictions_path)
    logger.info("[%s] resume: %d rows already done", dataset_key, len(done_uids))

    logger.info("[%s] loading %d shard(s) into memory", dataset_key, len(shard_paths))
    all_rows = load_dataset_rows(shard_paths)
    total_seen = len(all_rows)
    if limit is not None:
        all_rows = all_rows[:limit]
    pending = [r for r in all_rows if r["uid"] not in done_uids]
    logger.info(
        "[%s] %d total rows, %d already done, %d pending (limit=%s)",
        dataset_key, total_seen, len(done_uids), len(pending), limit,
    )

    batches = build_length_grouped_batches(pending, max_rows, max_batch_audio_s)
    logger.info(
        "[%s] length-grouped into %d batches (max_rows=%d, max_batch_audio_s=%.0f)",
        dataset_key, len(batches), max_rows, max_batch_audio_s,
    )

    total_new = 0
    t_start = time.time()
    t_last_log = t_start

    with open(predictions_path, "a", encoding="utf-8") as out_f:
        for bi, batch in enumerate(batches):
            audios = []
            for r in batch:
                try:
                    audios.append(decode_audio_cell(r["audio_cell"]))
                except Exception:
                    logger.exception("[%s] failed to decode audio for %s; using silence", dataset_key, r["uid"])
                    audios.append(None)

            keep_idx = [i for i, a in enumerate(audios) if a is not None]
            if not keep_idx:
                continue
            hyps = predictor.predict([audios[i] for i in keep_idx])

            for pos, i in enumerate(keep_idx):
                r = batch[i]
                rec = {
                    "uid": r["uid"],
                    "shard": r["shard"],
                    "row_idx": r["row_idx"],
                    "reference": r["ref"],
                    "hypothesis": hyps[pos],
                    "duration_s": r["duration"],
                }
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
            out_f.flush()
            os.fsync(out_f.fileno())
            total_new += len(keep_idx)

            now = time.time()
            if now - t_last_log >= PROGRESS_LOG_INTERVAL_S or bi == len(batches) - 1:
                elapsed = now - t_start
                rate = total_new / elapsed if elapsed > 0 else 0.0
                mem = (
                    f"{torch.cuda.memory_allocated() / 1e9:.2f}GB alloc / "
                    f"{torch.cuda.memory_reserved() / 1e9:.2f}GB reserved"
                    if torch.cuda.is_available() else "cpu"
                )
                logger.info(
                    "[%s] progress: batch %d/%d | %d new / %d pending (+%d resumed) | "
                    "%.2f rows/s | elapsed=%.0fs | gpu=%s",
                    dataset_key, bi + 1, len(batches), total_new, len(pending), len(done_uids),
                    rate, elapsed, mem,
                )
                t_last_log = now

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
    if limit is not None:
        metrics["note"] = f"PARTIAL: run capped at limit={limit} rows for smoke testing."
    with open(metrics_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    logger.info("[%s] metrics: %s", dataset_key, summary)
    return metrics_path


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", help="dataset key from datasets.discover_eval_targets(); omit with --all")
    ap.add_argument("--all", action="store_true", help="run every discovered dataset")
    ap.add_argument("--model-id", default="Qwen/Qwen3-ASR-0.6B-hf")
    ap.add_argument("--max-rows", type=int, default=DEFAULT_MAX_ROWS, help="max rows per batch")
    ap.add_argument("--max-batch-audio-s", type=float, default=DEFAULT_MAX_BATCH_AUDIO_S,
                     help="max summed raw audio seconds per batch")
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

    config = QwenPredictConfig(model_id=args.model_id)
    predictor = QwenPredictor(config)

    all_metrics = {}
    for key in keys:
        metrics_path = run_dataset(
            key, targets[key], predictor, out_dir,
            args.max_rows, args.max_batch_audio_s, args.limit, logger,
        )
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
