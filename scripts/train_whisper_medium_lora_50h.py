#!/usr/bin/env python3
"""LoRA fine-tune openai/whisper-medium for 1 epoch on the 50h speaker-disjoint
train subset, then evaluate on the speaker-disjoint val and test sets.

Data provenance (see SPEAKER_DISJOINT_SELECTION.md): speakers are ranked by
score_product_align and assigned to train/val/test as consecutive slices of that
ranking, so no speaker appears in more than one split. This run trains on the
top 644 train-block speakers (50.19h) rather than the full ~200h train block.

LoRA config matches the existing FINAL_200h/run2_custom_mix200h_100h_2ep adapter
(r=32, alpha=32, dropout=0.05, target_modules=[q_proj,fc1,v_proj,k_proj,fc2,out_proj])
so this run's numbers are directly comparable to that baseline.

Resumable: every `--save-every` steps, the adapter weights, optimizer/scheduler
state, RNG state, and step counter are checkpointed. Re-running the same command
loads the checkpoint and continues from that step over the same seeded shuffle
order, so a kill/OOM/restart repeats no work and skips no rows.
"""
from __future__ import annotations

import argparse
import io
import json
import logging
import random
import sys
import time
from pathlib import Path

import numpy as np
import pyarrow.parquet as pq
import torch
from torch.utils.data import DataLoader, Dataset

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.zero_shot_eval.audio_io import decode_audio_cell  # noqa: E402
from scripts.zero_shot_eval.evaluate import evaluate  # noqa: E402

SEED = 42
BASE_MODEL = "openai/whisper-medium"
LANGUAGE = "arabic"
TASK = "transcribe"


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("train_whisper_lora_50h")
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
    """Loads one parquet fully into memory (audio bytes + text), decodes lazily."""

    def __init__(self, path: Path, ref_column_candidates=("text", "transcription", "manual_normalized_transcript")):
        table = pq.read_table(path)
        cols = table.column_names
        ref_col = next((c for c in ref_column_candidates if c in cols), None)
        if ref_col is None:
            raise ValueError(f"{path}: no ref column found, have {cols}")
        d = table.to_pydict()
        self.audio = d["audio"]
        self.text = d[ref_col]
        self.uid = d["uid"] if "uid" in cols else [str(i) for i in range(len(self.text))]
        self.duration = d.get("duration")

    def __len__(self):
        return len(self.text)

    def __getitem__(self, idx):
        return {
            "audio": decode_audio_cell(self.audio[idx]),
            "text": self.text[idx] or "",
            "uid": self.uid[idx],
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
    # input that doesn't require grad and silently produce a None/zero gradient
    # through the frozen path -- gradient checkpointing needs this to work at all.
    model.enable_input_require_grads()
    model.gradient_checkpointing_enable()
    logger.info("trainable params: %s", model.get_nb_trainable_parameters())
    return model, processor


def save_checkpoint(model, optimizer, scheduler, step: int, epoch_done: bool, out_dir: Path):
    ckpt_dir = out_dir / "checkpoint"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(ckpt_dir / "adapter")
    torch.save(
        {
            "optimizer": optimizer.state_dict(),
            "scheduler": scheduler.state_dict(),
            "step": step,
            "epoch_done": epoch_done,
            "torch_rng": torch.get_rng_state(),
            "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
            "np_rng": np.random.get_state(),
            "py_rng": random.getstate(),
        },
        ckpt_dir / "train_state.pt",
    )
    (ckpt_dir / "state.json").write_text(json.dumps({"step": step, "epoch_done": epoch_done}, indent=2))


def load_checkpoint_if_present(model, optimizer, scheduler, out_dir: Path, logger: logging.Logger) -> int:
    ckpt_dir = out_dir / "checkpoint"
    state_path = ckpt_dir / "train_state.pt"
    if not state_path.exists():
        return 0
    from peft import PeftModel  # noqa: F401

    state = torch.load(state_path, map_location="cpu", weights_only=False)
    adapter_path = ckpt_dir / "adapter"
    model.load_adapter(str(adapter_path), adapter_name="default", is_trainable=True)
    optimizer.load_state_dict(state["optimizer"])
    scheduler.load_state_dict(state["scheduler"])
    torch.set_rng_state(state["torch_rng"])
    if state["cuda_rng"] is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda_rng"])
    np.random.set_state(state["np_rng"])
    random.setstate(state["py_rng"])
    step = state["step"]
    logger.info("resumed from checkpoint at step %d (epoch_done=%s)", step, state["epoch_done"])
    return 0 if state["epoch_done"] else step


@torch.no_grad()
def run_eval(model, processor, dataset: ParquetAudioTextDataset, batch_size: int, logger: logging.Logger, tag: str) -> dict:
    model.eval()
    device = next(model.parameters()).device
    refs, hyps = [], []
    collate = make_collate_fn(processor)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=collate)
    t0 = time.time()
    for bi, batch in enumerate(loader):
        feats = batch["input_features"].to(device=device, dtype=model.dtype)
        with torch.autocast(device_type="cuda" if torch.cuda.is_available() else "cpu", dtype=model.dtype):
            gen_ids = model.generate(
                input_features=feats, max_new_tokens=256,
                no_repeat_ngram_size=3, repetition_penalty=1.2,
                language=LANGUAGE, task=TASK,
            )
        texts = processor.tokenizer.batch_decode(gen_ids, skip_special_tokens=True)
        refs.extend(batch["texts"])
        hyps.extend([t.strip() for t in texts])
        if bi % 10 == 0:
            logger.info("[eval:%s] batch %d/%d (%.0fs elapsed)", tag, bi + 1, len(loader), time.time() - t0)
    model.train()
    metrics = evaluate(refs, hyps)
    logger.info("[eval:%s] n=%d WER=%.4f CER=%.4f", tag, metrics["n_scored"], metrics["wer"] or -1, metrics["cer"] or -1)
    return metrics


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train-parquet", default="/workspace/asr/Palestinian-ASR/eval_sets/custom_train_val/train_50h.parquet")
    ap.add_argument("--val-parquet", default="/workspace/asr/Palestinian-ASR/eval_sets/custom_train_val/val.parquet")
    ap.add_argument("--test-parquet", default="/workspace/asr/Palestinian-ASR/eval_sets/custom_test/masc_qasr_test.parquet")
    ap.add_argument("--out-dir", default="/root/Palestinian-ASR/outputs/whisper_medium_lora_50h_1ep")
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--log-every", type=int, default=20)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
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

    model, processor = build_model(logger)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model.to(device)

    collate = make_collate_fn(processor)
    # A fixed generator seeded from SEED gives the same shuffle order across a
    # resumed run as the original one, so `skip_steps` below lands on the same rows.
    g = torch.Generator().manual_seed(SEED)
    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, generator=g, collate_fn=collate, drop_last=False)

    total_steps = len(train_loader)
    optimizer = torch.optim.AdamW([p for p in model.parameters() if p.requires_grad], lr=args.lr)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(
        optimizer, max_lr=args.lr, total_steps=total_steps, pct_start=args.warmup_ratio
    )

    skip_steps = load_checkpoint_if_present(model, optimizer, scheduler, out_dir, logger)
    logger.info("total_steps=%d skip_steps=%d", total_steps, skip_steps)

    model.train()
    t0 = time.time()
    running_loss = 0.0
    for step, batch in enumerate(train_loader):
        if step < skip_steps:
            continue
        feats = batch["input_features"].to(device=device, dtype=model.dtype)
        labels = batch["labels"].to(device)
        out = model(input_features=feats, labels=labels)
        loss = out.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_([p for p in model.parameters() if p.requires_grad], 1.0)
        optimizer.step()
        scheduler.step()
        optimizer.zero_grad()

        running_loss += loss.item()
        if (step + 1) % args.log_every == 0:
            avg = running_loss / args.log_every
            elapsed = time.time() - t0
            rate = (step + 1 - skip_steps) / elapsed if elapsed > 0 else 0
            eta_s = (total_steps - step - 1) / rate if rate > 0 else float("inf")
            logger.info(
                "step %d/%d loss=%.4f lr=%.2e %.2f steps/s eta=%.0fs",
                step + 1, total_steps, avg, scheduler.get_last_lr()[0], rate, eta_s,
            )
            running_loss = 0.0

        if (step + 1) % args.save_every == 0:
            save_checkpoint(model, optimizer, scheduler, step + 1, epoch_done=False, out_dir=out_dir)
            logger.info("checkpoint saved at step %d", step + 1)

    save_checkpoint(model, optimizer, scheduler, total_steps, epoch_done=True, out_dir=out_dir)
    logger.info("training done in %.0fs, final checkpoint saved", time.time() - t0)

    model.save_pretrained(out_dir / "final_adapter")
    logger.info("final adapter saved to %s", out_dir / "final_adapter")

    val_metrics = run_eval(model, processor, val_ds, args.eval_batch_size, logger, "val")
    test_metrics = run_eval(model, processor, test_ds, args.eval_batch_size, logger, "test")

    summary = {
        "base_model": BASE_MODEL,
        "train_rows": len(train_ds),
        "train_hours_target": 50.0,
        "epochs": 1,
        "lora": {"r": 32, "alpha": 32, "dropout": 0.05,
                 "target_modules": ["q_proj", "fc1", "v_proj", "k_proj", "fc2", "out_proj"]},
        "val": {k: v for k, v in val_metrics.items() if k != "per_utterance"},
        "test": {k: v for k, v in test_metrics.items() if k != "per_utterance"},
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))
    (out_dir / "val_predictions.json").write_text(json.dumps(val_metrics["per_utterance"], ensure_ascii=False))
    (out_dir / "test_predictions.json").write_text(json.dumps(test_metrics["per_utterance"], ensure_ascii=False))
    logger.info("SUMMARY: %s", json.dumps(summary, ensure_ascii=False))


if __name__ == "__main__":
    main()
