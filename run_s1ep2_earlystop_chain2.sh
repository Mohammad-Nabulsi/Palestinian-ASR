#!/usr/bin/env bash
set -uo pipefail
cd /root/Palestinian-ASR
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

for r in run8_qasr_lev_masc_lev run9_qasr_lev_masc_lev_nonlev50h; do
  echo "=== $(date -Is) START $r ==="
  PAL_RUN="$r" PAL_STAGE2_EARLYSTOP=1 \
    /workspace/venv_qwen_gpu/bin/python -u pal_s1e1of2_run.py
  rc=$?
  echo "=== $(date -Is) END $r rc=$rc ==="
  if [ $rc -ne 0 ]; then
    echo "ABORTING chain: $r failed with rc=$rc"
    exit $rc
  fi
done
echo "=== $(date -Is) ALL QASR_LEV_MASC_LEV RUNS DONE ==="
