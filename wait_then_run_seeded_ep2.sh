#!/usr/bin/env bash
set -uo pipefail
echo "$(date -Is) waiting for run10/run11 chain (pid 20734, the waiter, and its child chain) to finish..."
while pgrep -f "pal_.*_run\.py" >/dev/null 2>&1; do
  sleep 20
done
# also wait for the waiter script itself, in case it's between stages
while kill -0 20734 2>/dev/null; do
  sleep 20
done
echo "$(date -Is) run10/run11 gone. Checking GPU is actually idle before launching..."
sleep 5
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 2000 ]; do
  echo "$(date -Is) GPU still has >2GB used, waiting..."
  sleep 10
done
echo "$(date -Is) GPU free. Launching seeded ep2 chain (run6/7/8/9 epoch-2 continuations)."
cd /root/Palestinian-ASR
./run_seeded_ep2_chain.sh
