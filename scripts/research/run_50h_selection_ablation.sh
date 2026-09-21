#!/bin/bash
# Promoted from the research scratch dir (~/run_all.sh) on the 5090 node.
# The three 50h runs: ascending / shuffled / random.
# See docs/HANDOFF.md for what it produced.
set -u
PY=~/anaconda3/envs/work/bin/python
S=~/sim_sets
cd ~/Palestinian-ASR
common="--batch-size 8 --lr 2e-5 --warmup-ratio 0.1 --n-chunks 1 --chunk-hours 50 \
        --test-eval end --heartbeat-min 20 --audio-dir $HOME/v3_data/audio \
        --test-parquet $S/test.parquet --sequence 1,1"

echo "=== RUN 1: ascending similarity =========================== $(date)"
$PY -u scripts/train_whisper_medium_lora_sequence.py $common \
  --train-parquet $S/train_ascending.parquet --rank-meta $S/train_ascending_rank.json \
  --val-parquet $S/val.parquet --within-chunk-order file \
  --run-name sim_ascending --out-dir ~/runs/sim_ascending 2>&1 | tail -400

echo "=== RUN 2: same set, shuffled ============================= $(date)"
$PY -u scripts/train_whisper_medium_lora_sequence.py $common \
  --train-parquet $S/train_shuffled.parquet --rank-meta $S/train_shuffled_rank.json \
  --val-parquet $S/val.parquet --within-chunk-order shuffle \
  --run-name sim_shuffled --out-dir ~/runs/sim_shuffled 2>&1 | tail -400

echo "=== RUN 3: random 50h ===================================== $(date)"
$PY -u scripts/train_whisper_medium_lora_sequence.py $common \
  --train-parquet $S/random_train.parquet --rank-meta $S/random_train_rank.json \
  --val-parquet $S/random_val.parquet --within-chunk-order shuffle \
  --run-name random_50h --out-dir ~/runs/random_50h 2>&1 | tail -400

echo "=== ALL RUNS DONE $(date)"
