#!/usr/bin/env python3
"""Recover per-checkpoint training-loss curves from the training logs.

The forward progressive run logged a training loss every 20 steps but only ever
wrote WER/CER into its summaries, so the loss curve exists solely as log text.
This parses it back out into a structured artifact.

Restarts complicate it: the log is append-mode across every relaunch, so a chunk
that crashed and resumed has its early steps recorded more than once. The last
record for a given (epoch, hours, step) is the one from the pass that actually
completed, so later lines win.

Writes outputs/experiment_pipeline/train_loss_curves.json.
"""
from __future__ import annotations

import json
import re
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "outputs"
PIPE = OUT / "experiment_pipeline"

LOGS = {
    "forward": OUT / "whisper_medium_lora_progressive" / "train.log",
    "reverse2": OUT / "whisper_medium_lora_reverse2" / "train.log",
    "reverse_all": OUT / "whisper_medium_lora_reverse_all" / "train.log",
}

# "ep2/h200 step 3680/3808 loss=0.3431 lr=3.06e-05"
LINE = re.compile(r"ep(\d+)/h(\d+) step (\d+)/(\d+) loss=([0-9.]+) lr=([0-9.e+-]+)")


def parse(log_path: Path) -> dict:
    if not log_path.exists():
        return {}
    per_tag: dict[str, dict[int, dict]] = {}
    for line in log_path.read_text(errors="replace").splitlines():
        m = LINE.search(line)
        if not m:
            continue
        epoch, hours, step, total, loss, lr = m.groups()
        tag = f"ep{epoch}_h{hours}"
        per_tag.setdefault(tag, {})[int(step)] = {
            "step": int(step), "total_steps": int(total),
            "loss": float(loss), "lr": float(lr),
        }

    out = {}
    for tag, by_step in per_tag.items():
        pts = [by_step[s] for s in sorted(by_step)]
        losses = [p["loss"] for p in pts]
        tail = losses[-len(losses) // 5:] if len(losses) >= 5 else losses
        out[tag] = {
            "n_points": len(pts),
            "total_steps": pts[-1]["total_steps"],
            "first": losses[0],
            "last": losses[-1],
            "mean": sum(losses) / len(losses),
            "mean_last_fifth": sum(tail) / len(tail),
            "min": min(losses),
            "curve": [[p["step"], p["loss"]] for p in pts],
        }
    return out


def main() -> None:
    PIPE.mkdir(parents=True, exist_ok=True)
    all_curves = {run: parse(path) for run, path in LOGS.items()}
    all_curves = {k: v for k, v in all_curves.items() if v}
    (PIPE / "train_loss_curves.json").write_text(json.dumps(all_curves, indent=2))

    print(f"wrote {PIPE / 'train_loss_curves.json'}")
    for run, tags in all_curves.items():
        print(f"\n{run}:")
        print(f"  {'checkpoint':12} {'pts':>5} {'first':>7} {'last':>7} {'mean':>7} {'mean_last_5th':>13} {'min':>7}")
        for tag in sorted(tags):
            c = tags[tag]
            print(f"  {tag:12} {c['n_points']:5d} {c['first']:7.4f} {c['last']:7.4f} "
                  f"{c['mean']:7.4f} {c['mean_last_fifth']:13.4f} {c['min']:7.4f}")


if __name__ == "__main__":
    main()
