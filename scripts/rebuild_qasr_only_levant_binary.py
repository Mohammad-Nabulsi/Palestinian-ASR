#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shlex
import shutil
import subprocess
from pathlib import Path

ROOT = Path("/home/MohammadNabulsi/whisper")
AUDIO_SCAN_SCRIPT = ROOT / "dialect_identifiaction" / "arabic_dialect_scan_badrex_mms300m.py"
SPLIT_SCRIPT = ROOT / "scripts" / "create_levant_non_levant_splits.py"
DEFAULT_TEXT_ROW_PROBS = ROOT / "Runs" / "text_dialect_scan_marbertv2_written_clean_masc_c_qasr" / "row_probabilities.jsonl"
DEFAULT_AUDIO_OUTPUT_DIR = ROOT / "Runs" / "dialect_scan_badrex_mms300m_lev08_text_candidates_qasr_only_qasrfix"
DEFAULT_TEMP_QASR_DATA_ROOT = ROOT / ".tmp_qasr_only_split_input"
DEFAULT_QASR_ONLY_TEXT_PROBS = ROOT / "Runs" / "text_dialect_scan_marbertv2_written_clean_qasr_only" / "row_probabilities.jsonl"
DEFAULT_QASR_ONLY_SPLIT_ROOT = ROOT / "data_curated_levant_binary_qasr_only_rebuild"
DEFAULT_TARGET_DATASET_ROOT = ROOT / "data_curated_levant_binary_v1"


def run_cmd(cmd: list[str]) -> None:
    print("$", " ".join(shlex.quote(part) for part in cmd), flush=True)
    subprocess.run(cmd, check=True, cwd=ROOT)


def write_qasr_only_text_probs(src: Path, dst: Path) -> int:
    dst.parent.mkdir(parents=True, exist_ok=True)
    count = 0
    with src.open('r', encoding='utf-8') as fin, dst.open('w', encoding='utf-8') as fout:
        for line in fin:
            rec = json.loads(line)
            if rec.get('source') != 'qasr':
                continue
            fout.write(json.dumps(rec, ensure_ascii=False) + '\n')
            count += 1
    return count


def build_qasr_only_input_tree(src_data_root: Path, out_root: Path) -> int:
    if out_root.exists():
        shutil.rmtree(out_root)
    out_root.mkdir(parents=True, exist_ok=True)
    count = 0
    for path in sorted(src_data_root.glob('processed_qasr_segments*.parquet')):
        target = out_root / path.name
        os.symlink(path, target)
        count += 1
    return count


def replace_qasr_dirs(split_root: Path, target_root: Path) -> None:
    for split in ('train', 'val', 'test'):
        src_qasr = split_root / split / 'qasr'
        dst_qasr = target_root / split / 'qasr'
        if dst_qasr.exists():
            shutil.rmtree(dst_qasr)
        shutil.copytree(src_qasr, dst_qasr)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            'Run a QASR-only repaired audio dialect scan, rebuild the QASR Levantine/non-Levantine '
            'split with the double 0.8 threshold, and replace the QASR split inside the main curated dataset.'
        )
    )
    parser.add_argument('--source-data-root', type=Path, default=ROOT / 'data')
    parser.add_argument('--text-row-probs', type=Path, default=DEFAULT_TEXT_ROW_PROBS)
    parser.add_argument('--qasr-only-text-row-probs', type=Path, default=DEFAULT_QASR_ONLY_TEXT_PROBS)
    parser.add_argument('--audio-output-dir', type=Path, default=DEFAULT_AUDIO_OUTPUT_DIR)
    parser.add_argument('--temp-qasr-data-root', type=Path, default=DEFAULT_TEMP_QASR_DATA_ROOT)
    parser.add_argument('--qasr-only-split-root', type=Path, default=DEFAULT_QASR_ONLY_SPLIT_ROOT)
    parser.add_argument('--target-dataset-root', type=Path, default=DEFAULT_TARGET_DATASET_ROOT)
    parser.add_argument('--device', default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--threshold', type=float, default=0.8)
    parser.add_argument('--rows-per-shard', type=int, default=100000)
    parser.add_argument('--batch-size', type=int, default=32768)
    parser.add_argument('--compression', default='snappy')
    parser.add_argument('--parquet-batch-size', type=int, default=64)
    parser.add_argument('--log-every', type=int, default=500)
    parser.add_argument('--min-seconds', type=float, default=2.0)
    parser.add_argument('--resume-audio-scan', action='store_true')
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    qasr_text_count = write_qasr_only_text_probs(args.text_row_probs, args.qasr_only_text_row_probs)
    print(f'Wrote {qasr_text_count} QASR text-candidate rows to {args.qasr_only_text_row_probs}')

    shard_count = build_qasr_only_input_tree(args.source_data_root, args.temp_qasr_data_root)
    print(f'Linked {shard_count} QASR shards into {args.temp_qasr_data_root}')

    audio_cmd = [
        str(ROOT / '.venv' / 'bin' / 'python'),
        str(AUDIO_SCAN_SCRIPT),
        '--data-root', str(args.temp_qasr_data_root),
        '--output-dir', str(args.audio_output_dir),
        '--text-probabilities-path', str(args.qasr_only_text_row_probs),
        '--text-target-threshold', str(args.threshold),
        '--device', args.device,
        '--parquet-batch-size', str(args.parquet_batch_size),
        '--log-every', str(args.log_every),
        '--min-seconds', str(args.min_seconds),
    ]
    if args.resume_audio_scan:
        audio_cmd.append('--resume')

    split_cmd = [
        str(ROOT / '.venv' / 'bin' / 'python'),
        str(SPLIT_SCRIPT),
        '--overwrite',
        '--data-root', str(args.temp_qasr_data_root),
        '--text-row-probs', str(args.qasr_only_text_row_probs),
        '--audio-row-probs', str(args.audio_output_dir / 'row_probabilities.jsonl'),
        '--output-root', str(args.qasr_only_split_root),
        '--threshold', str(args.threshold),
        '--rows-per-shard', str(args.rows_per_shard),
        '--batch-size', str(args.batch_size),
        '--compression', args.compression,
    ]

    print('Stage 1: rerun repaired QASR-only audio dialect classification', flush=True)
    run_cmd(audio_cmd)
    print('Stage 2: rebuild QASR-only Levantine/non-Levantine split', flush=True)
    run_cmd(split_cmd)
    print('Stage 3: replace QASR split inside target curated dataset', flush=True)
    replace_qasr_dirs(args.qasr_only_split_root, args.target_dataset_root)
    print('Done.', flush=True)
    print(f'Audio scan output: {args.audio_output_dir}')
    print(f'QASR-only split output: {args.qasr_only_split_root}')
    print(f'Replaced QASR directories in: {args.target_dataset_root}')


if __name__ == '__main__':
    main()
