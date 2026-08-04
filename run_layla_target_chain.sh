#!/usr/bin/env bash
set -e
cd /root/Palestinian-ASR
PY=/workspace/venv_qwen_gpu/bin/python
ROOT=/workspace/asr_env/checkpoints

echo "$(date -Iseconds) waiting for run12_qasrlev_target (pid 38642) to finish..."
while kill -0 38642 2>/dev/null; do sleep 15; done
echo "$(date -Iseconds) run12 done."
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do
  echo "$(date -Iseconds) waiting for GPU headroom..."
  sleep 10
done

run_bypass() {
  local pal_run="$1" ckpt="$2"
  echo "$(date -Iseconds) === START $pal_run (ckpt=$ckpt) ==="
  PAL_RUN="$pal_run" PAL_STAGE1_CKPT="$ckpt" PAL_STAGE2_EARLYSTOP=1 "$PY" -u pal_layla_target_run.py
  echo "$(date -Iseconds) === END $pal_run rc=$? ==="
}

echo "$(date -Iseconds) === START layla_only (no pretrain) ==="
PAL_RUN=layla_only "$PY" -u pal_layla_target_run.py
echo "$(date -Iseconds) === END layla_only rc=$? ==="

run_bypass run2_1ep_to_layla "$ROOT/whisper_medium_pal__run2_jor/openai__whisper-medium/stage1/epoch001"
run_bypass run2_2ep_to_layla "$ROOT/whisper_medium_pal__run2_jor/openai__whisper-medium/stage1/epoch002"
run_bypass run3_1ep_to_layla "$ROOT/whisper_medium_pal__run3_omni/openai__whisper-medium/stage1/epoch001"
run_bypass run3_2ep_to_layla "$ROOT/whisper_medium_pal__run3_omni/openai__whisper-medium/stage1/epoch002"
run_bypass run4_1ep_to_layla "$ROOT/whisper_medium_pal__run4_omni_jor/openai__whisper-medium/stage1/epoch001"
run_bypass run4_2ep_to_layla "$ROOT/whisper_medium_pal__run4_omni_jor/openai__whisper-medium/stage1/epoch002"
run_bypass run6_1ep_to_layla "$ROOT/whisper_medium_pal__run6_qasr_lev_10h/openai__whisper-medium/stage1/best"
run_bypass run6_2ep_to_layla "$ROOT/whisper_medium_pal__run6_qasr_lev_10h_ep2/openai__whisper-medium/stage1/best"
run_bypass run7_1ep_to_layla "$ROOT/whisper_medium_pal__run7_qasr_non_lev_50h/openai__whisper-medium/stage1/best"
run_bypass run7_2ep_to_layla "$ROOT/whisper_medium_pal__run7_qasr_non_lev_50h_ep2/openai__whisper-medium/stage1/best"
run_bypass run8_1ep_to_layla "$ROOT/whisper_medium_pal__run8_qasr_lev_masc_lev/openai__whisper-medium/stage1/best"
run_bypass run8_2ep_to_layla "$ROOT/whisper_medium_pal__run8_qasr_lev_masc_lev_ep2/openai__whisper-medium/stage1/epoch002"
run_bypass run9_1ep_to_layla "$ROOT/whisper_medium_pal__run9_qasr_lev_masc_lev_nonlev50h/openai__whisper-medium/stage1/best"
run_bypass run9_2ep_to_layla "$ROOT/whisper_medium_pal__run9_qasr_lev_masc_lev_nonlev50h_ep2/openai__whisper-medium/stage1/epoch002"

echo "$(date -Iseconds) === START run11_to_layla (2-step merge) ==="
PAL_RUN=run11_to_layla \
  PAL_STAGE1_CKPT="$ROOT/whisper_medium_pal__run11_qasr_nonlev_then_masc_qasr_lev/stage1_merged_adapter" \
  PAL_STAGE1_CKPT2="$ROOT/whisper_medium_pal__run11_qasr_nonlev_then_masc_qasr_lev/stage2_merged_adapter" \
  PAL_STAGE2_EARLYSTOP=1 "$PY" -u pal_layla_target_run.py
echo "$(date -Iseconds) === END run11_to_layla rc=$? ==="

echo "$(date -Iseconds) ALL LAYLA-TARGET RUNS DONE"
