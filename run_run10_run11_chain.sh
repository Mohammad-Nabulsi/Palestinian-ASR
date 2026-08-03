#!/usr/bin/env bash
set -uo pipefail
cd /root/Palestinian-ASR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

echo "=== $(date -Is) START run10_all_combined ==="
PAL_RUN=run10_all_combined PAL_STAGE2_EARLYSTOP=1 \
  /workspace/venv_qwen_gpu/bin/python -u pal_s1e1of2_run.py
rc=$?
echo "=== $(date -Is) END run10_all_combined rc=$rc ==="
if [ $rc -ne 0 ]; then
  echo "ABORTING chain: run10_all_combined failed with rc=$rc"
  exit $rc
fi

echo "=== $(date -Is) START run11_qasr_nonlev_then_masc_qasr_lev ==="
PAL_RUN=run11_qasr_nonlev_then_masc_qasr_lev \
  /workspace/venv_qwen_gpu/bin/python -u pal_run11_3stage.py
rc=$?
echo "=== $(date -Is) END run11_qasr_nonlev_then_masc_qasr_lev rc=$rc ==="
if [ $rc -ne 0 ]; then
  echo "ABORTING chain: run11 failed with rc=$rc"
  exit $rc
fi

echo "=== $(date -Is) ALL RUN10 RUN11 DONE ==="
