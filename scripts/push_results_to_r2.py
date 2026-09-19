#!/usr/bin/env python3
"""Push finished LoRA checkpoints + their results/configs to a fresh directory
under R2's existing `backup/transfer/adapters/` prefix (alongside FINAL_200h and
whisper).

What goes up, per run (forward / reverse2 / reverse_all, whichever exist):
  checkpoints/<tag>/adapter/*        the actual LoRA weights + config
  checkpoints/<tag>/summary.json     that checkpoint's val/test WER/CER + loss
  all_results.json                  every checkpoint's summary, combined
  train.log                          full training log (small: <200KB each)

Plus, once, at the top level of the destination:
  benchmarks/                        out-of-domain WER/CER for every adapter
  pipeline_results/                  RESULTS.md/json, val_loss.json, train_loss_curves.json
  config.json                        run hyperparameters (base model, LoRA config,
                                      seed, chunk definitions, per-run sequence)
  README.md                          what this directory is, one line per run

Deliberately NOT uploaded: resume_state/ (in-flight optimizer/RNG state, only
meaningful for resuming ON THIS BOX -- copying it elsewhere would be misleading,
since a resume needs the exact same process, not just the same weights).

Uses pyarrow.fs.copy_files (recursive local -> S3, multithreaded) rather than a
hand-rolled walk, matching how this repo already reads from R2 elsewhere.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.fs as fs

REPO_ROOT = Path(__file__).resolve().parents[1]
OUT = REPO_ROOT / "outputs"

RUNS = {
    "forward": OUT / "whisper_medium_lora_progressive",
    "reverse2": OUT / "whisper_medium_lora_reverse2",
    "reverse_all": OUT / "whisper_medium_lora_reverse_all",
}

CONFIG = {
    "base_model": "openai/whisper-medium",
    "seed": 42,
    "lora": {"r": 32, "alpha": 32, "dropout": 0.05,
             "target_modules": ["q_proj", "fc1", "v_proj", "k_proj", "fc2", "out_proj"]},
    "batch_size": 8, "eval_batch_size": 32, "lr": 1e-4, "warmup_ratio": 0.1,
    "chunk_hours": 50.0, "n_chunks": 4,
    "chunk_definition": ("speaker-disjoint 200h train block, ranked by "
                          "score_product_align (see SPEAKER_DISJOINT_SELECTION.md), "
                          "sliced into four consecutive ~50h chunks; chunk 1 = most "
                          "Levantine-confident 50h, chunk 4 = least"),
    "runs": {
        "forward": {"sequence": "1,2,3,4,1,2,3,4 (2 epochs, chunk order 1-4)",
                    "note": "N_EPOCHS cut from 3 to 2 mid-run per instruction"},
        "reverse2": {"sequence": "4,3,4,3 (2 epochs over the 2 least-confident chunks)"},
        "reverse_all": {"sequence": "4,3,2,1 (full reverse pass)",
                        "note": "stages 1-2 (chunks 4,3) inherited from reverse2's "
                                "checkpoints/s2_c3 via --init-from, not retrained"},
    },
}


def dest_fs():
    return fs.S3FileSystem(
        access_key=os.environ["R2_ACCESS_KEY_ID"],
        secret_key=os.environ["R2_SECRET_ACCESS_KEY"],
        endpoint_override=os.environ["R2_ENDPOINT"],
        scheme="https",
    )


def push_dir(s3, local_dir: Path, remote_path: str) -> int:
    if not local_dir.exists():
        return 0
    n = sum(1 for p in local_dir.rglob("*") if p.is_file())
    print(f"  {local_dir} -> {remote_path}  ({n} files, {sum(p.stat().st_size for p in local_dir.rglob('*') if p.is_file())/1e6:.1f} MB)")
    fs.copy_files(str(local_dir), remote_path, destination_filesystem=s3)
    return n


def push_file(s3, local_file: Path, remote_path: str) -> bool:
    if not local_file.exists():
        return False
    fs.copy_files(str(local_file), remote_path, destination_filesystem=s3)
    print(f"  {local_file} -> {remote_path}")
    return True


def main() -> None:
    dest_root = sys.argv[1] if len(sys.argv) > 1 else None
    if not dest_root:
        stamp = datetime.now(timezone.utc).strftime("%Y-%m-%d")
        dest_root = f"backup/transfer/adapters/lora_speaker_disjoint_{stamp}"
    print(f"destination: {dest_root}\n")

    s3 = dest_fs()
    manifest = {"destination": dest_root, "runs": {}}

    for run_name, run_dir in RUNS.items():
        if not run_dir.exists():
            print(f"[{run_name}] no output dir, skipping")
            continue
        checkpoints = sorted(p.name for p in (run_dir / "checkpoints").iterdir()
                             if (p / "summary.json").exists()) if (run_dir / "checkpoints").exists() else []
        if not checkpoints:
            print(f"[{run_name}] no completed checkpoints, skipping")
            continue
        print(f"[{run_name}] {len(checkpoints)} completed checkpoint(s): {checkpoints}")
        n_files = push_dir(s3, run_dir / "checkpoints", f"{dest_root}/{run_name}/checkpoints")
        push_file(s3, run_dir / "all_results.json", f"{dest_root}/{run_name}/all_results.json")
        push_file(s3, run_dir / "train.log", f"{dest_root}/{run_name}/train.log")
        manifest["runs"][run_name] = {"checkpoints": checkpoints, "n_files": n_files}

    print("\n[shared] benchmarks + pipeline results + config")
    push_dir(s3, OUT / "adapter_benchmarks", f"{dest_root}/benchmarks")
    pipe = OUT / "experiment_pipeline"
    for name in ("RESULTS.md", "RESULTS.json", "val_loss.json", "train_loss_curves.json"):
        push_file(s3, pipe / name, f"{dest_root}/pipeline_results/{name}")

    readme_lines = [
        f"# LoRA speaker-disjoint experiment -- {dest_root.rsplit('/', 1)[-1]}",
        "",
        "Pushed by scripts/push_results_to_r2.py. See config.json for hyperparameters",
        "and pipeline_results/RESULTS.md for the full comparison table (zero-shot",
        "baselines + every fine-tuned checkpoint's WER/CER/loss + out-of-domain",
        "benchmarks against Casablanca/Layla/Omni).",
        "",
        "Runs included:",
    ]
    for run_name, info in manifest["runs"].items():
        readme_lines.append(f"  - {run_name}: {', '.join(info['checkpoints'])}")
    readme_lines.append("")
    readme_lines.append("resume_state/ (in-flight optimizer/RNG state) is intentionally not")
    readme_lines.append("included -- it is only meaningful for resuming training on the original box.")

    config_path = REPO_ROOT / ".r2_push_config.tmp.json"
    readme_path = REPO_ROOT / ".r2_push_readme.tmp.md"
    config_path.write_text(json.dumps(CONFIG, indent=2))
    readme_path.write_text("\n".join(readme_lines))
    push_file(s3, config_path, f"{dest_root}/config.json")
    push_file(s3, readme_path, f"{dest_root}/README.md")
    config_path.unlink()
    readme_path.unlink()

    print(f"\nDONE. Uploaded to {dest_root}")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
