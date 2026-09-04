#!/usr/bin/env python3
"""LoRA fine-tune openai/whisper-medium progressively over the 200h speaker-disjoint
train block, in four confidence-ordered 50h chunks, for 3 total epochs -- 12
checkpoints, each evaluated on the speaker-disjoint val and test sets.

Chunking: the train block's 3,978 speakers are ranked by score_product_align (the
same ranking that built val/test -- see SPEAKER_DISJOINT_SELECTION.md) and sliced
into four consecutive ~50h chunks in that order, so "the first 50h" always means the
same, most-confident subset of speakers. One epoch = walking chunk 0, 1, 2, 3 in that
order once; three epochs means each chunk is walked three times, and a chunk's own
shuffle is reseeded per (epoch, chunk) so the row order differs each pass without
losing reproducibility. The model, optimizer, and LR schedule are never reset between
chunks or epochs -- this is one continuous run of 12 sub-epochs, not 12 independent
training jobs, matching the existing FINAL_200h adapters' own epoch_N/ checkpoint
convention.

Checkpoint naming: ep{E}_h{H} for E in 1..3, H in (50,100,150,200) cumulative hours
within that epoch. Each one gets its own adapter save + val/test WER/CER.

LoRA config matches the existing FINAL_200h/run2_custom_mix200h_100h_2ep adapter
(r=32, alpha=32, dropout=0.05, target_modules=[q_proj,fc1,v_proj,k_proj,fc2,out_proj]).

Resumable: (epoch, chunk_idx, step_within_chunk) plus optimizer/scheduler/RNG state
are checkpointed every `--save-every` steps and at every chunk boundary. Re-running
the same command picks up exactly where it left off -- a kill/OOM/restart repeats no
completed chunk and skips no row within the in-progress one.
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
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset, Subset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.zero_shot_eval.audio_io import decode_audio_cell  # noqa: E402
from scripts.zero_shot_eval.evaluate import evaluate  # noqa: E402

SEED = 42
BASE_MODEL = "openai/whisper-medium"
LANGUAGE = "arabic"
TASK = "transcribe"
CHUNK_HOURS = 50.0
N_CHUNKS = 4
N_EPOCHS = 2  # was 3; cut short per instruction -- finish epoch 2, skip epoch 3


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_whisper_lora_progressive")
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


class ParquetAudioTextDataset(Dataset):
    """Keeps the parquet as a pyarrow Table (compact columnar storage) and
    materializes only one row's worth of Python objects per __getitem__ call.

    `table.to_pydict()` on a table this size (129k rows with embedded audio
    bytes, ~17-20GB in Arrow's own in-memory format) duplicates the whole
    thing into Python-level bytes/list/dict objects -- both copies alive at
    once during the conversion -- which is exactly what pushed this process
    past this container's ~57GB cgroup memory limit and got it SIGKILLed
    (exit 137) before a single training step ran. Per-row `.as_py()` calls
    materialize only what's needed for that sample, so only one compact Arrow
    copy of the dataset ever sits in memory.

    `speaker_key`/`source` (needed for chunk assignment, not per-sample
    training) ARE pulled once via to_pydict(), but only those two short-string
    columns -- negligible next to the audio column.
    """

    def __init__(self, path: Path, ref_column_candidates=("text", "transcription", "manual_normalized_transcript")):
        self.table = pq.read_table(path)
        cols = self.table.column_names
        self.ref_col = next((c for c in ref_column_candidates if c in cols), None)
        if self.ref_col is None:
            raise ValueError(f"{path}: no ref column found, have {cols}")
        self.has_uid = "uid" in cols
        meta_cols = [c for c in ("speaker_key", "source") if c in cols]
        meta = self.table.select(meta_cols).to_pydict() if meta_cols else {}
        self.speaker_key = meta.get("speaker_key")
        self.source = meta.get("source")

    def __len__(self):
        return self.table.num_rows

    def __getitem__(self, idx):
        audio_cell = self.table.column("audio")[idx].as_py()
        text = self.table.column(self.ref_col)[idx].as_py()
        uid = self.table.column("uid")[idx].as_py() if self.has_uid else str(idx)
        return {
            "audio": decode_audio_cell(audio_cell),
            "text": text or "",
            "uid": uid,
        }


def make_collate_fn(processor):
    def collate(batch):
        audios = [b["audio"] for b in batch]
        texts = [b["text"] for b in batch]
        feats = processor.feature_extractor(audios, sampling_rate=16000, return_tensors="pt")
        labels = processor.tokenizer(
            texts, return_tensors="pt", padding=True, truncation=True, max_length=256
        )
        label_ids = labels["input_ids"].masked_fill(labels["attention_mask"] == 0, -100)
        return {
            "input_features": feats["input_features"],
            "labels": label_ids,
            "texts": texts,
            "uids": [b["uid"] for b in batch],
        }

    return collate


def build_chunk_indices(train_ds: ParquetAudioTextDataset, rank_meta_path: Path, logger: logging.Logger) -> list[list[int]]:
    """Row indices for each of the 4 confidence-ordered ~50h chunks.

    rank_meta (written by the extraction script) gives every train speaker's
    cumulative hours in score order; a row belongs to chunk k if its speaker's
    cumulative-hours mark falls in ((k)*CHUNK_HOURS, (k+1)*CHUNK_HOURS].
    """
    rank_meta = json.loads(rank_meta_path.read_text())
    boundary_of_speaker: dict[tuple[str, str], int] = {}
    for entry in rank_meta:
        chunk = min(N_CHUNKS - 1, int(entry["cum_hours"] // CHUNK_HOURS))
        boundary_of_speaker[(entry["source"], entry["speaker_key"])] = chunk

    chunks: list[list[int]] = [[] for _ in range(N_CHUNKS)]
    unresolved = 0
    for i in range(len(train_ds)):
        key = (train_ds.source[i], train_ds.speaker_key[i])
        chunk = boundary_of_speaker.get(key)
        if chunk is None:
            unresolved += 1
            continue
        chunks[chunk].append(i)

    for k, idxs in enumerate(chunks):
        logger.info("chunk %d (target <=%.0fh): %d rows", k, (k + 1) * CHUNK_HOURS, len(idxs))
    if unresolved:
        logger.warning("%d rows had no speaker-rank match and were excluded", unresolved)
    return chunks


def build_model(logger: logging.Logger):
    from peft import LoraConfig, get_peft_model
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=dtype)
    base.generation_config.language = LANGUAGE
    base.generation_config.task = TASK
    base.generation_config.forced_decoder_ids = None

    lora_config = LoraConfig(
        r=32, lora_alpha=32, lora_dropout=0.05, bias="none",
        target_modules=["q_proj", "fc1", "v_proj", "k_proj", "fc2", "out_proj"],
    )
    model = get_peft_model(base, lora_config)
    # LoRA freezes the backbone, so without this the checkpointed segments see an
    # input that doesn't require grad and produce no gradient through the frozen
    # path -- gradient checkpointing needs this to work at all.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    logger.info("trainable params: %s", model.get_nb_trainable_parameters())
    return model, processor


def save_train_state(model, optimizer, scheduler, epoch: int, chunk_idx: int, step_in_chunk: int,
                      chunk_done: bool, out_dir: Path):
    """Saves the model weights ALONGSIDE the optimizer/scheduler/step counters,
    in the same call. Originally these were saved separately -- counters every
    --save-every steps, model weights only at chunk boundaries -- so a crash
    between boundaries and resume would restore an optimizer/scheduler position
    ahead of where the actual LoRA weights were, silently reverting up to
    --save-every steps of real training on every resume. Saving both together
    keeps them atomically consistent.
    """
    state_dir = out_dir / "resume_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(state_dir / "adapter")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "epoch": epoch, "chunk_idx": chunk_idx, "step_in_chunk": step_in_chunk,
            "chunk_done": chunk_done,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "np_rng": np.random.get_state(),
            "py_rng": random.getstate(),
        },
        state_dir / "train_state.pt",
    )


def load_train_state_if_present(model, optimizer, scheduler, out_dir: Path, logger: logging.Logger):
    state_path = out_dir / "resume_state" / "train_state.pt"
    if not state_path.exists():
        return 0, 0, 0
    state = torch.load(state_path, map_location="cpu", weights_only=False)
    epoch, chunk_idx, step_in_chunk = state["epoch"], state["chunk_idx"], state["step_in_chunk"]

    # resume_state/adapter is saved in the same call as the counters above, so
    # it is always exactly consistent with step_in_chunk -- load it unconditionally
    # rather than searching for the last chunk-boundary checkpoint, which could
    # lag behind by up to --save-every steps.
    model.load_adapter(str(out_dir / "resume_state" / "adapter"), adapter_name="default", is_trainable=True)
    logger.info("loaded adapter weights from resume_state (in sync with step_in_chunk=%d)", step_in_chunk)

    if state["chunk_done"]:
        chunk_idx += 1
        step_in_chunk = 0
        if chunk_idx >= N_CHUNKS:
            chunk_idx = 0
            epoch += 1
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["np_rng"])
    random.setstate(state["py_rng"])
    logger.info("resumed at epoch=%d chunk_idx=%d step_in_chunk=%d", epoch, chunk_idx, step_in_chunk)
    return epoch, chunk_idx, step_in_chunk


def latest_checkpoint_dir(out_dir: Path, epoch: int, chunk_idx: int) -> Path | None:
    """The most recently completed chunk's adapter, walking backward from
    (epoch, chunk_idx) -- the starting point for resuming mid-chunk."""
    e, c = epoch, chunk_idx - 1
    while e >= 1:
        while c >= 0:
            tag = f"ep{e}_h{int((c + 1) * CHUNK_HOURS)}"
            d = out_dir / "checkpoints" / tag
            if (d / "adapter").exists():
                return d
            c -= 1
        e -= 1
        c = N_CHUNKS - 1
    return None


@torch.no_grad()
def run_eval(model, processor, dataset: ParquetAudioTextDataset, batch_size: int, logger: logging.Logger, tag: str) -> dict:
    model.eval()
    device = next(model.parameters()).device
    refs, hyps = [], []
    collate = make_collate_fn(processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate,
                        num_workers=0)  # see train_one_chunk's num_workers=0 comment
    t0 = time.time()
    for bi, batch in enumerate(loader):
        feats = batch["input_features"].to(device=device, dtype=model.dtype)
        gen_ids = model.generate(
            input_features=feats, max_new_tokens=256,
            no_repeat_ngram_size=3, repetition_penalty=1.2,
            language=LANGUAGE, task=TASK,
        )
        texts = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        refs.extend(batch["texts"])
        hyps.extend([t.strip() for t in texts])
        if bi % 20 == 0:
            logger.info("[eval:%s] batch %d/%d (%.0fs elapsed)", tag, bi + 1, len(loader), time.time() - t0)
    model.train()
    metrics = evaluate(refs, hyps)
    logger.info("[eval:%s] n=%d WER=%.4f CER=%.4f (%.0fs)", tag, metrics["n_scored"],
                metrics["wer"] or -1, metrics["cer"] or -1, time.time() - t0)
    return metrics


def train_one_chunk(model, processor, optimizer, scheduler, train_ds, chunk_indices,
                     epoch: int, chunk_idx: int, resume_step: int,
                     device, batch_size: int, save_every: int, log_every: int,
                     out_dir: Path, logger: logging.Logger) -> list[tuple[int, float]]:
    """Trains one chunk and returns its [(step, mean loss over the last log_every
    steps)] history, so a caller can record the training-loss curve in a summary
    instead of leaving it only in the log text. The progressive run's own main()
    ignores the return value."""
    subset = Subset(train_ds, chunk_indices)
    g = torch.Generator().manual_seed(SEED + epoch * 1000 + chunk_idx)
    collate = make_collate_fn(processor)
    # num_workers=0 deliberately: this container has a ~57GB cgroup memory
    # ceiling and the in-memory train table alone is ~41GB (see
    # ParquetAudioTextDataset's docstring). Forked worker processes gradually
    # duplicate pages of that table via CPython refcounting even on read-only
    # access, which is what risks tipping over the ceiling a second time.
    # Per-sample WAV decode is cheap (no resample needed, already 16kHz), so
    # GPU compute stays the bottleneck even single-process.
    loader = DataLoader(subset, batch_size=batch_size, shuffle=True, generator=g,
                        collate_fn=collate, num_workers=0, drop_last=False)
    total_steps = len(loader)
    hours_tag = int((chunk_idx + 1) * CHUNK_HOURS)
    logger.info("=== epoch %d chunk %d (<=%dh): %d rows, %d steps, resume_step=%d ===",
                epoch, chunk_idx, hours_tag, len(chunk_indices), total_steps, resume_step)

    model.train()
    t0 = time.time()
    running_loss = 0.0
    loss_history: list[tuple[int, float]] = []
    trainable = [p for p in model.parameters() if p.requires_grad]
    for step, batch in enumerate(loader):
        if step < resume_step:
            continue
        feats = batch["input_features"].to(device=device, dtype=model.dtype)
        labels = batch["labels"].to(device)
        out = model(input_features=feats, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(trainable, 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        if (step + 1) % log_every == 0:
            avg = running_loss / log_every
            loss_history.append((step + 1, avg))
            elapsed = time.time() - t0
            rate = (step + 1 - resume_step) / elapsed if elapsed > 0 else 0
            eta_s = (total_steps - step - 1) / rate if rate > 0 else float("inf")
            logger.info(
                "ep%d/h%d step %d/%d loss=%.4f lr=%.2e %.2f steps/s eta=%.0fs",
                epoch, hours_tag, step + 1, total_steps, avg, scheduler.get_last_lr()[0], rate, eta_s,
            )
            running_loss = 0.0

        if (step + 1) % save_every == 0:
            save_train_state(model, optimizer, scheduler, epoch, chunk_idx, step + 1, chunk_done=False, out_dir=out_dir)
            logger.info("resume-state saved at ep%d/h%d step %d", epoch, hours_tag, step + 1)

    logger.info("chunk done in %.0fs", time.time() - t0)
    return loss_history


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", default="/workspace/asr/Palestinian-ASR/eval_sets/custom_train_val/train_full_200h.parquet")
    ap.add_argument("--rank-meta", default="/workspace/asr/Palestinian-ASR/eval_sets/custom_train_val/train_full_rank.json")
    ap.add_argument("--val-parquet", default="/workspace/asr/Palestinian-ASR/eval_sets/custom_train_val/val.parquet")
    ap.add_argument("--test-parquet", default="/workspace/asr/Palestinian-ASR/eval_sets/custom_test/masc_qasr_test.parquet")
    ap.add_argument("--out-dir", default="/root/Palestinian-ASR/outputs/whisper_medium_lora_progressive")
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--eval-batch-size", type=int, default=32)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    (out_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "train.log")

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

    # One continuous LR schedule across all N_EPOCHS * N_CHUNKS sub-epochs.
    total_steps_all = sum(
        (len(chunk_indices[c]) + args.batch_size - 1) // args.batch_size
        for _ in range(N_EPOCHS) for c in range(N_CHUNKS)
    )
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps_all, pct_start=args.warmup_ratio
    )

    start_epoch, start_chunk, resume_step = load_train_state_if_present(model, optimizer, scheduler, out_dir, logger)

    all_results = {}
    results_path = out_dir / "all_results.json"
    if results_path.exists():
        all_results = json.loads(results_path.read_text())

    run_t0 = time.time()
    for epoch in range(start_epoch or 1, N_EPOCHS + 1):
        chunk_range = range(start_chunk if epoch == start_epoch else 0, N_CHUNKS)
        for chunk_idx in chunk_range:
            hours_tag = int((chunk_idx + 1) * CHUNK_HOURS)
            tag = f"ep{epoch}_h{hours_tag}"
            ckpt_dir = out_dir / "checkpoints" / tag

            already_done = (ckpt_dir / "summary.json").exists()
            if already_done:
                logger.info("%s already complete, skipping", tag)
                resume_step = 0
                continue

            this_resume_step = resume_step if (epoch == start_epoch and chunk_idx == start_chunk) else 0
            train_one_chunk(model, processor, optimizer, scheduler, train_ds, chunk_indices[chunk_idx],
                            epoch, chunk_idx, this_resume_step, device, args.batch_size,
                            args.save_every, args.log_every, out_dir, logger)
            resume_step = 0

            # Adapter must land on disk BEFORE chunk_done=True is recorded: a crash
            # between the two, in the other order, would mark the chunk done while
            # latest_checkpoint_dir() finds no adapter for it and silently resumes
            # from a stale, older checkpoint instead.
            ckpt_dir.mkdir(parents=True, exist_ok=True)
            model.save_pretrained(ckpt_dir / "adapter")
            save_train_state(model, optimizer, scheduler, epoch, chunk_idx, 0, chunk_done=True, out_dir=out_dir)
            logger.info("%s adapter saved", tag)

            val_metrics = run_eval(model, processor, val_ds, args.eval_batch_size, logger, f"{tag}/val")
            test_metrics = run_eval(model, processor, test_ds, args.eval_batch_size, logger, f"{tag}/test")

            summary = {
                "tag": tag, "epoch": epoch, "cumulative_hours": hours_tag,
                "elapsed_total_s": time.time() - run_t0,
                "val": {k: v for k, v in val_metrics.items() if k != "per_utterance"},
                "test": {k: v for k, v in test_metrics.items() if k != "per_utterance"},
            }
            (ckpt_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
            all_results[tag] = summary
            results_path.write_text(json.dumps(all_results, indent=2, ensure_ascii=False))
            logger.info("SUMMARY %s: %s", tag, json.dumps(summary, ensure_ascii=False))

    logger.info("ALL DONE in %.0fs", time.time() - run_t0)


if __name__ == "__main__":
    main()
