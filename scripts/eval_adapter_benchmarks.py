#!/usr/bin/env python3
"""Benchmark every fine-tuned LoRA adapter against the out-of-domain eval sets --
Casablanca Palestinian, Casablanca Jordanian, Layla, and Omnilingual APC -- on the
exact code path the zero-shot baselines used, so tuned and zero-shot numbers sit in
one comparable table.

Why these four and not the custom MASC+QASR test block: every adapter already has a
val and test number on that block from its own in-training eval, and no row in
custom val/test exceeds 30s (max 16.9s), so the training script's plain
`model.generate` and this pipeline's 30s sliding-window chunking see byte-identical
audio and identical generate kwargs. Re-decoding 5,733 rows x N adapters would cost
hours to reproduce numbers we already have.

Decode path, deliberately identical to scripts/zero_shot_eval/run_eval.py: the same
HF ASR pipeline, the same chunk_length_s/stride, the same generate kwargs
(max_new_tokens=256, no_repeat_ngram_size=3, repetition_penalty=1.2), the same
reference-column resolution, and the same resumable predictions.jsonl writer -- this
script reuses run_eval.run_dataset() rather than reimplementing it.

Precision: the adapters were trained in bfloat16 but the zero-shot baselines ran in
float16, so the LoRA merge happens in float32 (lossless for both) and the merged
model is then cast to float16 to match the baselines exactly. float16 holds these
small LoRA deltas with more mantissa than bfloat16 did, so nothing is lost.

The base model is reloaded from the local HF cache once per adapter (~20s) instead
of resetting a cached copy in place -- merge_and_unload() mutates the module tree,
and a fresh load is far harder to get subtly wrong than an unmerge/restore dance.

Usage:
    python scripts/eval_adapter_benchmarks.py --discover
    python scripts/eval_adapter_benchmarks.py --adapter-dir outputs/.../ep1_h50/adapter --tag ep1_h50
"""
from __future__ import annotations

import argparse
import gc
import json
import logging
import sys
import time
from pathlib import Path

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
ZS_DIR = REPO_ROOT / "scripts" / "zero_shot_eval"
sys.path.insert(0, str(ZS_DIR))

from eval_sets import discover_eval_targets  # noqa: E402
from run_eval import run_dataset, setup_logger  # noqa: E402
from whisper_predict import WhisperPredictConfig, WhisperPredictor  # noqa: E402

BASE_MODEL = "openai/whisper-medium"
DEFAULT_DATASETS = ("casablanca_palestinian", "casablanca_jordanian", "layla", "omnilingual_apc")

# Run dirs searched by --discover, in report order. Each contributes every
# checkpoints/<tag>/adapter it holds.
RUN_DIRS = (
    ("forward", REPO_ROOT / "outputs" / "whisper_medium_lora_progressive"),
    ("reverse2", REPO_ROOT / "outputs" / "whisper_medium_lora_reverse2"),
    ("reverse_all", REPO_ROOT / "outputs" / "whisper_medium_lora_reverse_all"),
)


class MergedAdapterPredictor(WhisperPredictor):
    """WhisperPredictor over a locally built, LoRA-merged whisper-medium.

    Inherits predict() and the CUDA-OOM batch-size backoff untouched; only the
    model construction differs from the zero-shot path, which loads by model id.
    """

    def __init__(self, adapter_dir: Path, config: WhisperPredictConfig):
        self.adapter_dir = Path(adapter_dir)
        super().__init__(config)

    def _load(self) -> None:
        from peft import PeftModel
        from transformers import WhisperForConditionalGeneration, WhisperProcessor
        from transformers import pipeline as hf_pipeline

        c = self.config
        t0 = time.time()
        processor = WhisperProcessor.from_pretrained(BASE_MODEL, language=c.language, task=c.task)
        base = WhisperForConditionalGeneration.from_pretrained(BASE_MODEL, dtype=torch.float32)
        base.generation_config.language = c.language
        base.generation_config.task = c.task
        base.generation_config.forced_decoder_ids = None

        peft_model = PeftModel.from_pretrained(base, str(self.adapter_dir))
        merged = peft_model.merge_and_unload()
        merged = merged.to(dtype=c.dtype)

        self.pipe = hf_pipeline(
            "automatic-speech-recognition",
            model=merged,
            tokenizer=processor.tokenizer,
            feature_extractor=processor.feature_extractor,
            dtype=c.dtype,
            device=c.device,
            chunk_length_s=c.chunk_length_s,
            stride_length_s=c.stride_length_s,
            ignore_warning=True,
            generate_kwargs={
                "language": c.language,
                "task": c.task,
                "max_new_tokens": c.max_new_tokens,
                "no_repeat_ngram_size": 3,
                "repetition_penalty": 1.2,
            },
        )
        logging.getLogger(__name__).info(
            "loaded %s merged into %s in %.1fs", self.adapter_dir, BASE_MODEL, time.time() - t0
        )

    def release(self) -> None:
        del self.pipe
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def discover_adapters() -> list[tuple[str, str, Path]]:
    """(run_name, tag, adapter_dir) for every completed checkpoint on disk.

    A checkpoint counts as completed only when its summary.json exists -- the
    adapter/ directory alone can be a stage that trained but whose val/test eval
    was still running when the process was last killed.
    """
    found: list[tuple[str, str, Path]] = []
    for run_name, run_dir in RUN_DIRS:
        ckpt_root = run_dir / "checkpoints"
        if not ckpt_root.is_dir():
            continue
        for tag_dir in sorted(ckpt_root.iterdir()):
            adapter = tag_dir / "adapter"
            if adapter.is_dir() and (tag_dir / "summary.json").exists():
                found.append((run_name, tag_dir.name, adapter))
    return found


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--discover", action="store_true", help="benchmark every completed checkpoint on disk")
    ap.add_argument("--adapter-dir", help="single adapter directory (with --tag)")
    ap.add_argument("--tag", help="label for --adapter-dir")
    ap.add_argument("--run-name", default="adhoc", help="run label for --adapter-dir")
    ap.add_argument("--datasets", nargs="*", default=list(DEFAULT_DATASETS))
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--limit", type=int, default=None, help="cap rows per dataset (smoke test)")
    ap.add_argument("--out-dir", default=str(REPO_ROOT / "outputs" / "adapter_benchmarks"))
    args = ap.parse_args()

    if not args.discover and not args.adapter_dir:
        ap.error("pass --discover or --adapter-dir/--tag")
    if args.adapter_dir and not args.tag:
        ap.error("--adapter-dir needs --tag")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    logger = setup_logger(out_dir / "run.log")

    adapters = discover_adapters() if args.discover else [(args.run_name, args.tag, Path(args.adapter_dir))]
    targets = discover_eval_targets()
    keys = [k for k in args.datasets if k in targets]
    missing = [k for k in args.datasets if k not in targets]
    if missing:
        logger.warning("requested datasets not present on disk, skipping: %s", missing)
    if not keys:
        raise SystemExit(f"none of {args.datasets} are present; have {sorted(targets)}")

    logger.info("benchmarking %d adapter(s) x %d dataset(s): %s", len(adapters), len(keys), keys)

    combined_path = out_dir / "all_benchmarks.json"
    combined = json.loads(combined_path.read_text()) if combined_path.exists() else {}

    for run_name, tag, adapter_dir in adapters:
        adapter_out = out_dir / tag
        summary_path = adapter_out / "summary.json"
        if summary_path.exists() and args.limit is None:
            existing = json.loads(summary_path.read_text())
            if all(k in existing.get("datasets", {}) for k in keys):
                logger.info("[%s] already benchmarked on all requested datasets, skipping", tag)
                combined[tag] = existing
                combined_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
                continue

        adapter_out.mkdir(parents=True, exist_ok=True)
        logger.info("=== %s (%s) ===", tag, adapter_dir)
        config = WhisperPredictConfig(model_id=BASE_MODEL, batch_size=args.batch_size)
        predictor = MergedAdapterPredictor(adapter_dir, config)

        per_dataset = {}
        t0 = time.time()
        for key in keys:
            metrics_path = run_dataset(key, targets[key], predictor, adapter_out,
                                       args.batch_size, args.limit, logger)
            metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
            per_dataset[key] = {k: v for k, v in metrics.items() if k != "per_utterance"}

        predictor.release()
        del predictor
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

        summary = {
            "tag": tag, "run": run_name, "adapter_dir": str(adapter_dir),
            "base_model": BASE_MODEL, "elapsed_s": time.time() - t0,
            "datasets": per_dataset,
        }
        summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
        combined[tag] = summary
        combined_path.write_text(json.dumps(combined, indent=2, ensure_ascii=False))
        logger.info("SUMMARY %s: %s", tag, json.dumps(
            {k: {"wer": v["wer"], "cer": v["cer"]} for k, v in per_dataset.items()}, ensure_ascii=False))

    logger.info("BENCHMARKS DONE: %d adapters in %s", len(combined), combined_path)


if __name__ == "__main__":
    main()
