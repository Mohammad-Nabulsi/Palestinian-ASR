#!/usr/bin/env bash
set -e
cd /root/Palestinian-ASR

echo "$(date -Iseconds) waiting for training (pid 29435) to finish..."
while kill -0 29435 2>/dev/null; do sleep 15; done
echo "$(date -Iseconds) training done."

echo "$(date -Iseconds) waiting for group B (pid 33282) to finish (success or crash)..."
while kill -0 33282 2>/dev/null; do sleep 15; done
echo "$(date -Iseconds) group B done."

until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 3000 ]; do
  echo "$(date -Iseconds) waiting for GPU headroom..."
  sleep 10
done

# re-resolve the matrix (picks up anything that finished since we last ran it) and diff
# against what's already been written to generalization_results/
/workspace/venv_qwen_gpu/bin/python scripts/resolve_generalization_matrix.py

MISSING=$(/workspace/venv_qwen_gpu/bin/python -c "
import json, os
matrix = json.load(open('generalization_matrix.json'))
all_names = [c['name'] for c in matrix]
done = set(f[:-5] for f in os.listdir('generalization_results') if f.endswith('.json'))
print(' '.join(n for n in all_names if n not in done))
")

echo "$(date -Iseconds) recovery batch: $MISSING"
if [ -n "$MISSING" ]; then
  /workspace/venv_qwen_gpu/bin/python -u run_generalization_eval.py $MISSING
fi
echo "$(date -Iseconds) recovery batch done."
