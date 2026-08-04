#!/usr/bin/env bash
set -e
cd /root/Palestinian-ASR
PY=/workspace/venv_qwen_gpu/bin/python

echo "$(date -Iseconds) waiting for layla_target_chain (pid 46835) to finish..."
while kill -0 46835 2>/dev/null; do sleep 15; done
echo "$(date -Iseconds) layla_target_chain done."
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do
  sleep 10
done

echo "$(date -Iseconds) === START lev65_to_layla ==="
PAL_RUN=lev65_to_layla PAL_STAGE2_EARLYSTOP=1 "$PY" -u pal_layla_target_run.py
echo "$(date -Iseconds) === END lev65_to_layla rc=$? ==="
echo "$(date -Iseconds) LEV65 FOLLOWUP DONE"
