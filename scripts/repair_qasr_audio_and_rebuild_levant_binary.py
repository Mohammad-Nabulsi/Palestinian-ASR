#!/usr/bin/env python3
from __future__ import annotations

import argparse
import shlex
import subprocess
from pathlib import Path

ROOT = Path("/home/MohammadNabulsi/whisper")
AUDIO_SCAN_SCRIPT = ROOT / "dialect_identifiaction" / "arabic_dialect_scan_badrex_mms300m.py"
SPLIT_SCRIPT = ROOT / "scripts" / "create_levant_non_levant_splits.py"
DEFAULT_DATA_ROOT = ROOT / "data"
DEFAULT_TEXT_ROW_PROBS = (
    ROOT / "Runs" / "text_dialect_scan_marbertv2_written_clean_masc_c_qasr" / "row_probabilities.jsonl"
)
DEFAULT_AUDIO_OUTPUT_DIR = (
    ROOT / "Runs" / "dialect_scan_badrex_mms300m_lev08_text_candidates_masc_c_qasr_qasrfix"
)
DEFAULT_SPLIT_OUTPUT_DIR = ROOT / "data_curated_levant_binary_v2_qasr_audio_fix"


def run_cmd(cmd: list[str]) -> None:
    print("$", " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Repair the QASR audio dialect scan by rerunning audio classification with raw-PCM-aware "
            "loading, then rebuild the Levantine/non-Levantine split using the double 0.8 threshold."
        )
    )
    parser.add_argument("--data-root", type=Path, default=DEFAULT_DATA_ROOT)
    parser.add_argument("--text-row-probs", type=Path, default=DEFAULT_TEXT_ROW_PROBS)
    parser.add_argument("--audio-output-dir", type=Path, default=DEFAULT_AUDIO_OUTPUT_DIR)
    parser.add_argument("--split-output-dir", type=Path, default=DEFAULT_SPLIT_OUTPUT_DIR)
    parser.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    parser.add_argument("--threshold", type=float, default=0.8)
    parser.add_argument("--rows-per-shard", type=int, default=100000)
    parser.add_argument("--batch-size", type=int, default=32768)
    parser.add_argument("--compression", default="snappy")
    parser.add_argument("--parquet-batch-size", type=int, default=64)
    parser.add_argument("--log-every", type=int, default=500)
    parser.add_argument("--min-seconds", type=float, default=2.0)
    parser.add_argument("--resume-audio-scan", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    audio_cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(AUDIO_SCAN_SCRIPT),
        "--data-root",
        str(args.data_root),
        "--output-dir",
        str(args.audio_output_dir),
        "--text-probabilities-path",
        str(args.text_row_probs),
        "--text-target-threshold",
        str(args.threshold),
        "--device",
        args.device,
        "--parquet-batch-size",
        str(args.parquet_batch_size),
        "--log-every",
        str(args.log_every),
        "--min-seconds",
        str(args.min_seconds),
    ]
    if args.resume_audio_scan:
        audio_cmd.append("--resume")

    split_cmd = [
        str(ROOT / ".venv" / "bin" / "python"),
        str(SPLIT_SCRIPT),
        "--overwrite",
        "--data-root",
        str(args.data_root),
        "--text-row-probs",
        str(args.text_row_probs),
        "--audio-row-probs",
        str(args.audio_output_dir / "row_probabilities.jsonl"),
        "--output-root",
        str(args.split_output_dir),
        "--threshold",
        str(args.threshold),
        "--rows-per-shard",
        str(args.rows_per_shard),
        "--batch-size",
        str(args.batch_size),
        "--compression",
        args.compression,
    ]

    print("Stage 1: rerun audio dialect classification with QASR PCM-aware decoding", flush=True)
    run_cmd(audio_cmd)
    print("Stage 2: rebuild Levantine/non-Levantine split from repaired audio predictions", flush=True)
    run_cmd(split_cmd)
    print("Done.", flush=True)
    print(f"Audio scan output: {args.audio_output_dir}")
    print(f"Split output: {args.split_output_dir}")


if __name__ == "__main__":
    main()
