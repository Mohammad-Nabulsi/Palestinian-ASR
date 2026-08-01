#!/usr/bin/env bash
# The pal study: unified.whisper.run.ipynb executed once per run, selected by PAL_RUN.
#   run1_pal_only  LoRA(r=16) on palTrain (early stopping on palVal WER)      -> eval palVal/palTest
#   run2_jor       stage1 = 1ep on ALL Jordanian Casablanca, merge, eval; stage2 = 1ep palTrain, eval
#   run3_omni      same, stage1 = ALL omni
#   run4_omni_jor  same, stage1 = omni + Jordanian Casablanca
#   run1_pal_only_1ep  run1 with a fixed 1-epoch budget (budget-matched to the stage 2s above)
# Sequential on purpose -- one GPU, and a shared HF/datasets cache.
#
# PAL_STAGE2_EPOCHS=N (default 1) gives the FINAL stage N epochs instead of 1, continuing from
# the saved adapter + optimizer state rather than restarting; results land under
# Runs/whisper_medium_pal/<run>_s2epN/ so each budget is stored separately, while the
# checkpoints stay under the un-suffixed run name (that is what the resume reads).
#
# Usage: scripts/run_pal_whisper_study.sh [run_name ...]
#        PAL_STAGE2_EPOCHS=2 scripts/run_pal_whisper_study.sh run2_jor run3_omni
set -uo pipefail
cd /root/Palestinian-ASR

# This GPU is shared with other training runs (a FastConformer run held 3.65GB of the 23.52GB
# card during the first attempt). expandable_segments keeps the caching allocator from
# fragmenting the smaller slice that leaves us.
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

RUNS=("$@")
if [ ${#RUNS[@]} -eq 0 ]; then
  RUNS=(run1_pal_only run2_jor run3_omni run4_omni_jor)
fi

export PAL_STAGE2_EPOCHS="${PAL_STAGE2_EPOCHS:-1}"

for r in "${RUNS[@]}"; do
  # Must match RUN_TAG in the notebook's Cell 1.
  tag="$r"
  [ "$PAL_STAGE2_EPOCHS" != "1" ] && tag="${r}_s2ep${PAL_STAGE2_EPOCHS}"
  out="Runs/whisper_medium_pal/$tag"
  mkdir -p "$out"
  echo "=== $(date -Is) START $tag (PAL_RUN=$r, stage2 epochs=$PAL_STAGE2_EPOCHS) ==="
  PAL_RUN="$r" jupyter nbconvert --to notebook --execute \
      --ExecutePreprocessor.kernel_name=venv_qwen_gpu \
      --ExecutePreprocessor.timeout=-1 \
      --output-dir "$out" --output executed.ipynb \
      unified.whisper.run.ipynb > "$out/nbconvert.log" 2>&1
  rc=$?
  echo "=== $(date -Is) END $tag rc=$rc ==="
  if [ $rc -ne 0 ]; then
    echo "ABORTING the chain: $tag failed. Tail of $out/nbconvert.log:"
    tail -40 "$out/nbconvert.log"
    exit $rc
  fi
  # Per-run summary, printed into the driver log so the chain's progress is readable in one place.
  [ -f "$out/SUMMARY.json" ] && /workspace/venv_qwen_gpu/bin/python -c "
import json,sys; s=json.load(open('$out/SUMMARY.json'))
print('  stages:', json.dumps(s['stages'], ensure_ascii=False))"
done
echo "=== $(date -Is) ALL RUNS DONE ==="
