#!/usr/bin/env python3
"""LoRA fine-tune openai/whisper-medium over an ARBITRARY ORDERED SEQUENCE of the
four confidence-ranked 50h chunks -- the generalization of
train_whisper_medium_lora_progressive.py, which hardcodes the forward order
(1,2,3,4) repeated per epoch.

Chunk numbering here is the USER-FACING one: chunk 1 is the most-confident 50h,
chunk 4 the least (index 0..3 internally). `--sequence 4,3,4,3` therefore means
"train on the 4th chunk, then the 3rd, then both again for a second epoch" -- four
stages, one adapter + val/test eval per stage.

Everything else matches the progressive run exactly so the resulting adapters are
comparable to it: same base model, same LoRA config (r=32, alpha=32, dropout=0.05,
same target_modules), same batch size, same lr/warmup, same chunk membership
(read from the same train_full_rank.json), same val/test sets and decode settings.
The one thing that necessarily differs is OneCycleLR's `total_steps`, which is the
length of THIS sequence rather than the forward run's 8 stages -- the schedule is a
property of a run's length, not a hyperparameter, so a shorter run rides a
correspondingly compressed cycle.

`--init-from DIR` starts from another run's stage checkpoint (adapter + optimizer
state) instead of the base model, and `--init-stage-offset N` declares that DIR
already contains the first N stages of this sequence, so those stages are skipped
and the LR schedule is fast-forwarded past their steps. That is how the reverse-all
run (4,3,2,1) reuses the reverse-2 run's first epoch (4,3) rather than retraining
it: the shared prefix is trained once.

Resumable on the same terms as the progressive script: (stage_pos, step_in_stage)
plus optimizer/scheduler/RNG state and the LoRA weights are checkpointed together
every --save-every steps, so a kill/OOM/restart repeats no completed stage and
skips no row of the in-progress one.
"""
from __future__ import annotations

import argparse
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.train_whisper_medium_lora_progressive import (  # noqa: E402
    CHUNK_HOURS,
    N_CHUNKS,
    SEED,
    ParquetAudioTextDataset,
    build_chunk_indices,
    build_model,
    run_eval,
    train_one_chunk,
)
from scripts.eval_val_loss import compute_loss  # noqa: E402


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_whisper_lora_sequence")
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


def parse_sequence(spec: str) -> list[int]:
    """"4,3,2,1" -> [3, 2, 1, 0]: user-facing chunk numbers to 0-based indices."""
    out = []
    for part in spec.split(","):
        n = int(part.strip())
        if not 1 <= n <= N_CHUNKS:
            raise ValueError(f"chunk number {n} out of range 1..{N_CHUNKS}")
        out.append(n - 1)
    if not out:
        raise ValueError("empty sequence")
    return out


def stage_tag(stage_pos: int, chunk_idx: int) -> str:
    """s{position in the sequence}_c{user-facing chunk number}, e.g. s3_c2."""
    return f"s{stage_pos + 1}_c{chunk_idx + 1}"


def save_state(model, optimizer, scheduler, stage_pos: int, step_in_stage: int,
               stage_done: bool, state_dir: Path) -> None:
    """Model weights and counters in one call, so a resume can never restore an
    optimizer position ahead of the weights (the desync bug the progressive
    script hit)."""
    state_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(state_dir / "adapter")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "stage_pos": stage_pos,
            "step_in_stage": step_in_stage,
            "stage_done": stage_done,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "np_rng": np.random.get_state(),
            "py_rng": random.getstate(),
        },
        state_dir / "train_state.pt",
    )


def load_resume_state(model, optimizer, scheduler, state_dir: Path,
                      logger: logging.Logger) -> tuple[int, int]:
    path = state_dir / "train_state.pt"
    if not path.exists():
        return 0, 0
    state = torch.load(path, map_location="cpu", weights_only=False)
    model.load_adapter(str(state_dir / "adapter"), adapter_name="default", is_trainable=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["np_rng"])
    random.setstate(state["py_rng"])

    # Two writers land in this directory with different key names: save_state()
    # below (stage_pos/step_in_stage/stage_done) at stage boundaries, and the
    # progressive script's save_train_state() -- reused verbatim inside
    # train_one_chunk -- every --save-every steps mid-stage, which spells the
    # same three things epoch/step_in_chunk/chunk_done and passes stage_pos+1 as
    # its "epoch". Read whichever schema is on disk rather than forking the
    # shared training loop just to rename its fields.
    if "stage_pos" in state:
        stage_pos, step_in_stage, done = state["stage_pos"], state["step_in_stage"], state["stage_done"]
    else:
        stage_pos, step_in_stage, done = state["epoch"] - 1, state["step_in_chunk"], state["chunk_done"]
    if done:
        stage_pos += 1
        step_in_stage = 0
    logger.info("resumed at stage_pos=%d step_in_stage=%d", stage_pos, step_in_stage)
    return stage_pos, step_in_stage


def load_init_from(model, optimizer, init_dir: Path, logger: logging.Logger) -> None:
    """Adopt another run's LoRA weights (and optimizer moments, when present) as
    this run's starting point. The scheduler is NOT taken from there -- this run
    builds its own cycle over its own length and fast-forwards it by
    --init-stage-offset stages' worth of steps."""
    model.load_adapter(str(init_dir / "adapter"), adapter_name="default", is_trainable=True)
    logger.info("initialized LoRA weights from %s", init_dir / "adapter")
    st = init_dir / "train_state.pt"
    if st.exists():
        state = torch.load(st, map_location="cpu", weights_only=False)
        try:
            optimizer.load_state_dict(state["optimizer"])
            logger.info("adopted optimizer moments from %s", st)
        except ValueError as exc:  # different param grouping -- fall back to fresh moments
            logger.warning("could not adopt optimizer state (%s); starting with fresh moments", exc)
    else:
        logger.warning("%s has no train_state.pt; starting with fresh optimizer moments", st)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--sequence", required=True,
                    help="chunk numbers in training order, 1=most confident 50h, e.g. '4,3,4,3'")
    ap.add_argument("--run-name", required=True, help="label used in logs and the results file")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--init-from", default=None,
                    help="stage checkpoint dir (holding adapter/ and train_state.pt) to start from")
    ap.add_argument("--init-stage-offset", type=int, default=0,
                    help="how many leading stages of --sequence --init-from already covers")
    ap.add_argument("--train-parquet", default="/workspace/asr/Palestinian-ASR/eval_sets/custom_train_val/train_full_200h.parquet")
    ap.add_argument("--rank-meta", default="/workspace/asr/Palestinian-ASR/eval_sets/custom_train_val/train_full_rank.json")
    ap.add_argument("--val-parquet", default="/workspace/asr/Palestinian-ASR/eval_sets/custom_train_val/val.parquet")
    ap.add_argument("--test-parquet", default="/workspace/asr/Palestinian-ASR/eval_sets/custom_test/masc_qasr_test.parquet")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    sequence = parse_sequence(args.sequence)
    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "train.log")
    logger.info("run=%s sequence=%s (user-facing chunk numbers: %s) init_from=%s offset=%d",
                args.run_name, sequence, args.sequence, args.init_from, args.init_stage_offset)

    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    torch.cuda.manual_seed_all(SEED)

    logger.info("loading datasets")
    train_ds = ParquetAudioTextDataset(Path(args.train_parquet))
    val_ds = ParquetAudioTextDataset(Path(args.val_parquet))
    test_ds = ParquetAudioTextDataset(Path(args.test_parquet))
    logger.info("train=%d val=%d test=%d rows", len(train_ds), len(val_ds), len(test_ds))

    chunk_indices = build_chunk_indices(train_ds, Path(args.rank_meta), logger)

    model, processor = build_model(logger)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    def steps_for(chunk_idx: int) -> int:
        return (len(chunk_indices[chunk_idx]) + args.batch_size - 1) // args.batch_size

    total_steps_all = sum(steps_for(c) for c in sequence)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps_all, pct_start=args.warmup_ratio
    )
    logger.info("sequence spans %d optimizer steps across %d stages", total_steps_all, len(sequence))

    state_dir = out_dir / "resume_state"
    start_stage, resume_step = load_resume_state(model, optimizer, scheduler, state_dir, logger)

    if start_stage == 0 and resume_step == 0 and args.init_from:
        # Fresh start of a run that inherits a prefix: adopt the weights, skip the
        # covered stages, and advance the LR schedule past their steps so stage
        # N+1 begins at the LR this run's own cycle prescribes for that point.
        load_init_from(model, optimizer, Path(args.init_from), logger)
        start_stage = args.init_stage_offset
        skip_steps = sum(steps_for(c) for c in sequence[:start_stage])
        for _ in range(skip_steps):
            scheduler.step()
        logger.info("skipped %d inherited stages, fast-forwarded scheduler %d steps to lr=%.2e",
                    start_stage, skip_steps, scheduler.get_last_lr()[0])

    results_path = out_dir / "all_results.json"
    all_results = json.loads(results_path.read_text()) if results_path.exists() else {}

    run_t0 = time.time()
    for stage_pos in range(start_stage, len(sequence)):
        chunk_idx = sequence[stage_pos]
        tag = stage_tag(stage_pos, chunk_idx)
        ckpt_dir = out_dir / "checkpoints" / tag

        if (ckpt_dir / "summary.json").exists():
            logger.info("%s already complete, skipping", tag)
            resume_step = 0
            continue

        this_resume_step = resume_step if stage_pos == start_stage else 0
        logger.info("=== %s: stage %d/%d, chunk %d ===", tag, stage_pos + 1, len(sequence), chunk_idx + 1)
        # train_one_chunk's (epoch, chunk_idx) pair only feeds its shuffle seed and
        # log labels; stage_pos stands in for epoch so each stage of this sequence
        # gets its own row order even when it revisits a chunk.
        loss_history = train_one_chunk(
            model, processor, optimizer, scheduler, train_ds, chunk_indices[chunk_idx],
            stage_pos + 1, chunk_idx, this_resume_step, device, args.batch_size,
            args.save_every, args.log_every, out_dir, logger)
        resume_step = 0

        ckpt_dir.mkdir(parents=True, exist_ok=True)
        # A stage checkpoint carries its own optimizer state too, so another run
        # can use it as an --init-from point (that is how 4,3,2,1 picks up 4,3).
        save_state(model, optimizer, scheduler, stage_pos, 0, stage_done=True, state_dir=ckpt_dir)
        save_state(model, optimizer, scheduler, stage_pos, 0, stage_done=True, state_dir=state_dir)
        logger.info("%s adapter + state saved", tag)

        val_metrics = run_eval(model, processor, val_ds, args.eval_batch_size, logger, f"{tag}/val")
        test_metrics = run_eval(model, processor, test_ds, args.eval_batch_size, logger, f"{tag}/test")
        # Held-out likelihood alongside WER/CER: a teacher-forced pass is ~2 min
        # against the ~11 min autoregressive decode, and it is the only signal that
        # distinguishes a WER plateau from actual overfitting. The forward run has
        # no such number natively -- scripts/eval_val_loss.py backfills it there.
        val_loss = compute_loss(model, processor, val_ds, args.eval_batch_size, logger, f"{tag}/val")
        test_loss = compute_loss(model, processor, test_ds, args.eval_batch_size, logger, f"{tag}/test")
        logger.info("[loss:%s] val=%.4f (ppl %.2f) test=%.4f (ppl %.2f)", tag,
                    val_loss["loss"], val_loss["perplexity"],
                    test_loss["loss"], test_loss["perplexity"])

        summary = {
            "run": args.run_name, "tag": tag, "stage": stage_pos + 1,
            "chunk": chunk_idx + 1, "sequence": args.sequence,
            "cumulative_hours_trained": CHUNK_HOURS * (stage_pos + 1),
            "elapsed_total_s": time.time() - run_t0,
            "val": {k: v for k, v in val_metrics.items() if k != "per_utterance"},
            "test": {k: v for k, v in test_metrics.items() if k != "per_utterance"},
            "val_loss": val_loss,
            "test_loss": test_loss,
            "train_loss": {
                "curve": loss_history,
                "mean": (sum(v for _, v in loss_history) / len(loss_history)) if loss_history else None,
                "first": loss_history[0][1] if loss_history else None,
                "last": loss_history[-1][1] if loss_history else None,
            },
        }
        (ckpt_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        all_results[tag] = summary
        results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
        logger.info("SUMMARY %s: %s", tag, json.dumps(summary, ensure_ascii=False))

    logger.info("ALL DONE in %.0fs", time.time() - run_t0)


if __name__ == "__main__":
    main()
