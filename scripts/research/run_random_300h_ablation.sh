#!/bin/bash
# Random-matched 300h ablation. IDENTICAL to the curated 300h run except the data:
# same loop, same 6x50h ascending banding, same 2 epochs, same lr, same val/test.
set -u
cd ~/Palestinian-ASR
exec ~/anaconda3/envs/work/bin/python -u scripts/train_whisper_medium_lora_sequence.py \
  --train-parquet ~/rnd300_data/parquet/train_ordered.parquet \
  --rank-meta     ~/rnd300_data/rank_meta.json \
  --val-parquet   ~/v3_data/parquet/val.parquet \
  --test-parquet  ~/v3_data/parquet/test.parquet \
  --audio-dir     ~/rnd300_data/audio \
  --sequence 1,2,3,4,5,6,1,2,3,4,5,6 --n-chunks 6 --chunk-hours 50 \
  --batch-size 8 --lr 1e-4 --warmup-ratio 0.1 \
  --test-eval end --heartbeat-min 20 \
  --run-name rnd300_matched --out-dir ~/runs/rnd300_matched
