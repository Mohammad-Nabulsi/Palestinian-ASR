#!/usr/bin/env bash
set -uo pipefail
cd /root/Palestinian-ASR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
ROOT=/workspace/asr_env/checkpoints

seed_and_run() {
  local old="$1" new="$2"
  local src_dir="$ROOT/whisper_medium_pal__${old}/openai__whisper-medium/stage1"
  local dst_dir="$ROOT/whisper_medium_pal__${new}/openai__whisper-medium/stage1"
  local latest
  latest=$(ls -d "$src_dir"/ckpt_step* 2>/dev/null | sort | tail -1)
  if [ -z "$latest" ]; then
    echo "ABORTING: no ckpt_step found under $src_dir (expected epoch-1 checkpoint from $old)"
    exit 1
  fi
  local epoch
  epoch=$(python3 -c "import json; print(json.load(open('$latest/trainer_state.json'))['epoch'])")
  if [ "$epoch" != "1" ]; then
    echo "ABORTING: $latest has epoch=$epoch, expected 1 (source run may not have finished stage1 cleanly)"
    exit 1
  fi
  mkdir -p "$dst_dir"
  cp -r "$latest" "$dst_dir/$(basename "$latest")"
  echo "[seed] copied $latest -> $dst_dir/$(basename "$latest") (epoch=1 state, will resume to epoch 2)"

  echo "=== $(date -Is) START $new (seeded from $old, stage1 epoch2 only + fresh stage2) ==="
  PAL_RUN="$new" PAL_STAGE1_EPOCHS=2 PAL_STAGE2_EARLYSTOP=1 \
    /workspace/venv_qwen_gpu/bin/python -u pal_s1e1of2_run.py
  local rc=$?
  echo "=== $(date -Is) END $new rc=$rc ==="
  if [ $rc -ne 0 ]; then
    echo "ABORTING chain: $new failed with rc=$rc"
    exit $rc
  fi
}

seed_and_run run6_qasr_lev_10h                     run6_qasr_lev_10h_ep2
seed_and_run run7_qasr_non_lev_50h                  run7_qasr_non_lev_50h_ep2
seed_and_run run8_qasr_lev_masc_lev                 run8_qasr_lev_masc_lev_ep2
seed_and_run run9_qasr_lev_masc_lev_nonlev50h       run9_qasr_lev_masc_lev_nonlev50h_ep2

echo "=== $(date -Is) ALL SEEDED EP2 RUNS DONE ==="
