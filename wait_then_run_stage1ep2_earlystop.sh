#!/usr/bin/env bash
set -uo pipefail
echo "$(date -Is) waiting for orchestrator pid 8087 (run_layla_jor_study.sh, all 3 of its runs) to finish..."
while kill -0 8087 2>/dev/null; do
  sleep 20
done
echo "$(date -Is) orchestrator gone. Double-checking no other pal_*_run.py is still alive..."
while pgrep -f "pal_.*_run\.py" >/dev/null 2>&1; do
  sleep 15
done
echo "$(date -Is) Checking GPU is actually idle before launching..."
sleep 5
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do
  echo "$(date -Is) GPU still has >2GB used, waiting..."
  sleep 10
done
echo "$(date -Is) GPU free. Launching stage1ep2-earlystop chain."
./run_s1ep2_earlystop_chain.sh
