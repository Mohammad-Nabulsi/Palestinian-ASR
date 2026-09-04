#!/usr/bin/env python3
"""Teacher-forced cross-entropy loss on the val and test blocks, for any LoRA adapter.

The training runs log a training loss every --log-every steps but never computed a
validation loss: run_eval() only decodes with model.generate() and scores WER/CER,
which says nothing about the model's likelihood on held-out audio. Without it there
is no way to separate "WER plateaued because the model stopped learning" from "WER
plateaued while the model kept fitting the training distribution" -- i.e. to see
overfitting. This fills that gap.

Two uses:
  * imported by train_whisper_medium_lora_sequence.py, so every new stage records
    val/test loss alongside its WER/CER; and
  * run standalone (--discover) to backfill the same numbers for adapters whose run
    already finished, at ~2 min per adapter -- a single teacher-forced forward pass
    over val+test, far cheaper than the ~11 min autoregressive decode.

Loss is aggregated per token, not per batch: HF returns a mean over each batch's
non-ignored label positions, so each batch is weighted by its own token count
before averaging. A plain mean of batch losses would silently over-weight short
batches. Perplexity is exp(loss) on that same token-level mean.
"""
from __future__ import annotations

import argparse
import json
import logging
import math
import sys
from pathlib import Path

import torch
from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))

from scripts.train_whisper_medium_lora_progressive import (  # noqa: E402
    BASE_MODEL,
    LANGUAGE,
    TASK,
    ParquetAudioTextDataset,
    make_collate_fn,
)

DEFAULT_VAL = "/workspace/asr/Palestinian-ASR/eval_sets/custom_train_val/val.parquet"
DEFAULT_TEST = "/workspace/asr/Palestinian-ASR/eval_sets/custom_test/masc_qasr_test.parquet"


@torch.no_grad()
def compute_loss(model, processor, dataset, batch_size: int, logger: logging.Logger,
                 tag: str = "") -> dict:
    """Token-weighted mean cross-entropy over `dataset`, with the model's own dtype."""
    was_training = model.training
    model.eval()
    device = next(model.parameters()).device
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=make_collate_fn(processor), num_workers=0)
    total_nll = 0.0
    total_tokens = 0
    for bi, batch in enumerate(loader):
        feats = batch["input_features"].to(device=device, dtype=model.dtype)
        labels = batch["labels"].to(device)
        out = model(input_features=feats, labels=labels)
        n_tok = int((labels != -100).sum().item())
        if n_tok == 0:
            continue
        total_nll += float(out.loss.item()) * n_tok
        total_tokens += n_tok
        if bi % 40 == 0:
            logger.info("[loss:%s] batch %d/%d", tag, bi + 1, len(loader))
    if was_training:
        model.train()

    loss = total_nll / total_tokens if total_tokens else None
    return {
        "loss": loss,
        "perplexity": math.exp(loss) if loss is not None and loss < 20 else None,
        "n_tokens": total_tokens,
        "n_rows": len(dataset),
    }


def build_adapter_model(adapter_dir: Path, logger: logging.Logger):
    """whisper-medium + this adapter, in bfloat16 -- the dtype the LoRA was trained
    in, so the reported loss is the loss the training run itself would have seen."""
    from peft import PeftModel
    from transformers import WhisperForConditionalGeneration, WhisperProcessor

    processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=LANGUAGE, task=TASK)
    dtype = torch.bfloat16 if torch.cuda.is_available() else torch.float32
    base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=dtype)
    base.generation_config.language = LANGUAGE
    base.generation_config.task = TASK
    base.generation_config.forced_decoder_ids = None
    model = PeftModel.from_pretrained(base, str(adapter_dir))
    model.to("cuda" if torch.cuda.is_available() else "cpu")
    logger.info("loaded %s", adapter_dir)
    return model, processor


def discover_adapters() -> list[tuple[str, str, Path]]:
    """Reuse the benchmark script's notion of a completed checkpoint."""
    sys.path.insert(0, str(REPO_ROOT / "scripts"))
    import importlib.util

    spec = importlib.util.spec_from_file_location(
        "_bench", REPO_ROOT / "scripts" / "eval_adapter_benchmarks.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.discover_adapters()


def setup_logger(log_path: Path) -> logging.Logger:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    logger = logging.getLogger("eval_val_loss")
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


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true", help="every completed checkpoint on disk")
    ap.add_argument("--adapter-dir")
    ap.add_argument("--tag")
    ap.add_argument("--val-parquet", default=DEFAULT_VAL)
    ap.add_argument("--test-parquet", default=DEFAULT_TEST)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--out-json", default=str(REPO_ROOT / "outputs" / "experiment_pipeline" / "val_loss.json"))
    args = ap.parse_args()

    if not args.discover and not args.adapter_dir:
        ap.error("pass --discover or --adapter-dir/--tag")

    out_path = Path(args.out_json)
    logger = setup_logger(out_path.parent / "val_loss.log")
    results = json.loads(out_path.read_text()) if out_path.exists() else {}

    adapters = discover_adapters() if args.discover else [("adhoc", args.tag, Path(args.adapter_dir))]
    todo = [a for a in adapters if a[1] not in results]
    logger.info("%d adapter(s) on disk, %d still need a loss number", len(adapters), len(todo))
    if not todo:
        logger.info("nothing to do")
        return

    val_ds = ParquetAudioTextDataset(Path(args.val_parquet))
    test_ds = ParquetAudioTextDataset(Path(args.test_parquet))
    logger.info("val=%d test=%d rows", len(val_ds), len(test_ds))

    for run_name, tag, adapter_dir in todo:
        model, processor = build_adapter_model(adapter_dir, logger)
        entry = {
            "run": run_name,
            "adapter_dir": str(adapter_dir),
            "val": compute_loss(model, processor, val_ds, args.batch_size, logger, f"{tag}/val"),
            "test": compute_loss(model, processor, test_ds, args.batch_size, logger, f"{tag}/test"),
        }
        results[tag] = entry
        out_path.write_text(json.dumps(results, indent=2, ensure_ascii=False))
        logger.info("LOSS %s: val=%.4f (ppl %.2f) test=%.4f (ppl %.2f)", tag,
                    entry["val"]["loss"], entry["val"]["perplexity"],
                    entry["test"]["loss"], entry["test"]["perplexity"])
        del model
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    logger.info("VAL LOSS DONE: %d adapters in %s", len(results), out_path)


if __name__ == "__main__":
    main()
