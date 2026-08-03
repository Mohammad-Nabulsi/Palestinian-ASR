#!/usr/bin/env bash
set -uo pipefail
echo "$(date -Is) waiting for group A (pid 29485) to finish..."
while kill -0 29485 2>/dev/null; do
  sleep 15
done
echo "$(date -Is) group A done. Checking GPU headroom before resuming group B..."
sleep 5
until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 15000 ]; do
  echo "$(date -Is) GPU still busy, waiting..."
  sleep 10
done
echo "$(date -Is) resuming group B."
cd /root/Palestinian-ASR
/workspace/venv_qwen_gpu/bin/python -u run_generalization_eval.py run2_2ep_alone run2_2ep_stage2final run3_2ep_alone run3_2ep_stage2final run6_2ep_alone run6_2ep_stage2final run7_2ep_alone run7_2ep_stage2final run4_1ep_alone run4_2ep_alone run4_2ep_stage2final run5_2ep_alone run5_2ep_stage2final run8_1ep_stage2final run8_2ep_stage2final run11_stage2_alone run10_alone run10_stage2final
