#!/bin/bash
# Promoted from the research scratch dir (~/eval_sweep.sh) on the 5090 node.
# 4 models x 5 test sets sweep.
# See docs/HANDOFF.md for what it produced.
set -u
PY=~/anaconda3/envs/work/bin/python
S=~/sim_sets; N=~/nat_resplit; V=~/v3_data/audio; O=~/eval_sweep
mkdir -p $O
declare -A SETS=(
  [newtest]="$S/test.parquet|$V"
  [casa_pal]="$N/parquet/casa_pal_test.parquet|$N/audio"
  [casa_jor]="$N/parquet/casa_jor_test.parquet|$N/audio"
  [layla_told]="$HOME/layla_told/layla_told_test.parquet|$N/audio"
  [omni_all]="$S/omni_test_all_speakers.parquet|$N/audio"
)
run () {  # name, model, adapter
  for s in newtest casa_pal casa_jor layla_told omni_all; do
    IFS='|' read -r pq ad <<< "${SETS[$s]}"
    out=$O/${1}__${s}.json
    [ -s "$out" ] && { echo "skip $1/$s"; continue; }
    echo "### $1 / $s  $(date +%H:%M:%S)"
    if [ "$2" = "whisper-lora" ]; then
      $PY -u ~/eval_model.py --parquet "$pq" --audio-dir "$ad" --model whisper-lora \
        --adapter "$3" --out "$out" --batch-size 16 --tag "$1/$s" 2>&1 | grep -E "RESULT|rows|Error|error" 
    else
      $PY -u ~/eval_model.py --parquet "$pq" --audio-dir "$ad" --model whisper-base \
        --out "$out" --batch-size 16 --tag "$1/$s" 2>&1 | grep -E "RESULT|rows|Error|error"
    fi
  done
}
run base whisper-base ""
run sim_ascending whisper-lora ~/runs/sim_ascending/checkpoints/s2_c1/adapter
run sim_shuffled  whisper-lora ~/runs/sim_shuffled/checkpoints/s2_c1/adapter
run random_50h    whisper-lora ~/runs/random_50h/checkpoints/s2_c1/adapter
echo "=== SWEEP DONE $(date) ==="
