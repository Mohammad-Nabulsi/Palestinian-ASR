#!/usr/bin/env python3
"""Assemble every result on disk into one report: zero-shot baselines, every
fine-tuned adapter's in-domain val/test, and every adapter's out-of-domain
benchmark numbers.

Writes outputs/experiment_pipeline/RESULTS.md and RESULTS.json. Safe to run at any
point -- it reports whatever exists and marks the rest as pending, so it doubles as
a progress view while the pipeline is still running.
"""
from __future__ import annotations

import json
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "outputs"
PIPE = OUT / "experiment_pipeline"

ZERO_SHOT = {
    "FastConformer CTC (NeMo)": "zs_conformer_ctc",
    "Whisper medium (base)": "whisper_medium_zero_shot",
    "Whisper large-v3": "whisper_large_v3_zero_shot",
    "Cohere transcribe-arabic": "cohere_transcribe_zero_shot",
    "omniASR_LLM_1B": "omni_asr_llm_1b_zero_shot",
}

TUNED_RUNS = {
    "forward (1,2,3,4 x2)": "whisper_medium_lora_progressive",
    "reverse2 (4,3,4,3)": "whisper_medium_lora_reverse2",
    "reverse_all (4,3,2,1)": "whisper_medium_lora_reverse_all",
}

OOD = ["casablanca_palestinian", "casablanca_jordanian", "layla", "omnilingual_apc"]
OOD_SHORT = {"casablanca_palestinian": "cas-PAL", "casablanca_jordanian": "cas-JOR",
             "layla": "layla", "omnilingual_apc": "omni"}


def pct(x) -> str:
    return "--" if x is None else f"{100 * x:.2f}"


def load_json(path: Path):
    try:
        return json.loads(path.read_text())
    except (OSError, json.JSONDecodeError):
        return None


def main() -> None:
    PIPE.mkdir(parents=True, exist_ok=True)
    report: dict = {"zero_shot": {}, "tuned": {}, "benchmarks": {}}

    for label, subdir in ZERO_SHOT.items():
        summary = load_json(OUT / subdir / "summary.json")
        if summary:
            report["zero_shot"][label] = {
                k: {"wer": v.get("wer"), "cer": v.get("cer"), "n": v.get("n_scored")}
                for k, v in summary.items()
            }

    # Loss lives in two places by design: the sequence runs record it in their own
    # summaries at eval time, while the forward run -- which never computed a
    # validation loss -- gets it from eval_val_loss.py's backfill artifact. Merge
    # rather than recompute what already exists.
    backfilled_loss = load_json(PIPE / "val_loss.json") or {}
    curves = load_json(PIPE / "train_loss_curves.json") or {}
    curve_by_tag = {tag: c for run in curves.values() for tag, c in run.items()}

    for label, subdir in TUNED_RUNS.items():
        results = load_json(OUT / subdir / "all_results.json") or {}
        entries = {}
        for tag, s in results.items():
            vl = s.get("val_loss") or backfilled_loss.get(tag, {}).get("val") or {}
            tl = s.get("test_loss") or backfilled_loss.get(tag, {}).get("test") or {}
            train = s.get("train_loss") or curve_by_tag.get(tag) or {}
            entries[tag] = {
                "val_wer": s["val"].get("wer"), "val_cer": s["val"].get("cer"),
                "test_wer": s["test"].get("wer"), "test_cer": s["test"].get("cer"),
                "val_loss": vl.get("loss"), "test_loss": tl.get("loss"),
                "train_loss_last_fifth": train.get("mean_last_fifth") or train.get("last"),
            }
        report["tuned"][label] = entries

    bench = load_json(OUT / "adapter_benchmarks" / "all_benchmarks.json") or {}
    for tag, s in bench.items():
        report["benchmarks"][tag] = {
            k: {"wer": v.get("wer"), "cer": v.get("cer"), "n": v.get("n_scored")}
            for k, v in s.get("datasets", {}).items()
        }

    lines: list[str] = ["# All ASR results", ""]

    lines += ["## Zero-shot baselines", "",
              "| model | " + " | ".join(
                  ["custom test"] + [OOD_SHORT[k] for k in OOD]) + " |",
              "|---|" + "---|" * (1 + len(OOD))]
    for label, sets in report["zero_shot"].items():
        cells = []
        for key in ["masc_qasr_custom_test"] + OOD:
            m = sets.get(key)
            cells.append("--" if not m else f"{pct(m['wer'])} / {pct(m['cer'])}")
        lines.append(f"| {label} | " + " | ".join(cells) + " |")
    lines += ["", "WER / CER, percent.", ""]

    lines += ["## Fine-tuned adapters -- in-domain (speaker-disjoint val / test)", "",
              "| run | checkpoint | train loss | val loss | test loss "
              "| val WER | val CER | test WER | test CER |",
              "|---|---|---|---|---|---|---|---|---|"]

    def num(x, nd=4):
        return "--" if x is None else f"{x:.{nd}f}"

    for label, tags in report["tuned"].items():
        for tag, m in sorted(tags.items()):
            lines.append(
                f"| {label} | {tag} | {num(m['train_loss_last_fifth'])} | {num(m['val_loss'])} "
                f"| {num(m['test_loss'])} | {pct(m['val_wer'])} | {pct(m['val_cer'])} "
                f"| {pct(m['test_wer'])} | {pct(m['test_cer'])} |")
    lines += ["", "Train loss is the mean over the final fifth of that stage's steps "
              "(the whole-stage mean is skewed by warmup). Val/test loss is teacher-forced "
              "token-level cross-entropy. WER/CER in percent.", ""]

    lines += ["## Fine-tuned adapters -- out-of-domain benchmarks", "",
              "| checkpoint | " + " | ".join(OOD_SHORT[k] for k in OOD) + " |",
              "|---|" + "---|" * len(OOD)]
    for tag, sets in sorted(report["benchmarks"].items()):
        cells = []
        for key in OOD:
            m = sets.get(key)
            cells.append("--" if not m else f"{pct(m['wer'])} / {pct(m['cer'])}")
        lines.append(f"| {tag} | " + " | ".join(cells) + " |")
    lines += ["", "WER / CER, percent. Same decode path as the zero-shot rows above.", ""]

    (PIPE / "RESULTS.json").write_text(json.dumps(report, indent=2, ensure_ascii=False))
    (PIPE / "RESULTS.md").write_text("\n".join(lines))
    print(f"wrote {PIPE / 'RESULTS.md'} and RESULTS.json")
    print(f"  zero-shot models: {len(report['zero_shot'])}")
    print(f"  tuned checkpoints: {sum(len(v) for v in report['tuned'].values())}")
    print(f"  benchmarked adapters: {len(report['benchmarks'])}")


if __name__ == "__main__":
    main()
