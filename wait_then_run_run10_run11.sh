#!/usr/bin/env bash
set -uo pipefail
echo "$(date -Is) waiting for run8/run9 chain (pid 18871) to finish..."
while kill -0 18871 2>/dev/null; do
  sleep 20
done
echo "$(date -Is) run8/run9 chain gone. Double-checking no other pal_*_run.py is still alive..."
while pgrep -f "pal_.*_run\.py" >/dev/null 2>&1; do
  sleep 15
done
echo "$(date -Is) Checking GPU is actually idle before launching..."
sleep 5
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do
  echo "$(date -Is) GPU still has >2GB used, waiting..."
  sleep 10
done
echo "$(date -Is) GPU free. Launching run10 -> run11 chain."
cd /root/Palestinian-ASR
./run_run10_run11_chain.sh
