#!/usr/bin/env bash
set -uo pipefail
cd /root/Palestinian-ASR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

declare -A CKPTS=(
  [run2_jor_s1e1of2]="/workspace/asr_env/checkpoints/whisper_medium_pal__run2_jor/openai__whisper-medium/stage1/epoch001"
  [run3_omni_s1e1of2]="/workspace/asr_env/checkpoints/whisper_medium_pal__run3_omni/openai__whisper-medium/stage1/epoch001"
  [run4_omni_jor_s1e1of2]="/workspace/asr_env/checkpoints/whisper_medium_pal__run4_omni_jor/openai__whisper-medium/stage1/epoch001"
)

for r in run2_jor_s1e1of2 run3_omni_s1e1of2 run4_omni_jor_s1e1of2; do
  echo "=== $(date -Is) START $r (stage1 ckpt: ${CKPTS[$r]}) ==="
  PAL_RUN="$r" PAL_STAGE1_CKPT="${CKPTS[$r]}" PAL_STAGE2_EPOCHS=1 \
    /workspace/venv_qwen_gpu/bin/python -u pal_s1e1of2_run.py
  rc=$?
  echo "=== $(date -Is) END $r rc=$rc ==="
  if [ $rc -ne 0 ]; then
    echo "ABORTING chain: $r failed with rc=$rc"
    exit $rc
  fi
done
echo "=== $(date -Is) ALL S1E1OF2 RUNS DONE ==="
