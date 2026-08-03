#!/usr/bin/env bash
set -e
cd /root/Palestinian-ASR

for pid in 33282 35874 35875; do
  echo "$(date -Iseconds) waiting for pid $pid..."
  while kill -0 "$pid" 2>/dev/null; do sleep 15; done
done
echo "$(date -Iseconds) groups B, C, D all finished."

until [ "$(nvidia-smi --query-gpu=memory.used --format=csv,noheader,nounits)" -lt 3000 ]; do
  sleep 10
done

/workspace/venv_qwen_gpu/bin/python scripts/resolve_generalization_matrix.py

MISSING=$(/workspace/venv_qwen_gpu/bin/python -c "
import json, os
matrix = json.load(open('generalization_matrix.json'))
all_names = [c['name'] for c in matrix]
done = set(f[:-5] for f in os.listdir('generalization_results') if f.endswith('.json'))
print(' '.join(n for n in all_names if n not in done))
")

echo "$(date -Iseconds) final recovery batch: $MISSING"
if [ -n "$MISSING" ]; then
  /workspace/venv_qwen_gpu/bin/python -u run_generalization_eval.py $MISSING
fi
echo "$(date -Iseconds) ALL DONE."
