#!/usr/bin/env bash
# Bring Layla up to the same state as the rest of the cleaned data:
# land raw zip on network storage -> extract -> build parquet shards -> fast text-clean.
#
# Layla does NOT go through merge_cleaned_outputs_and_report.py: historically it was
# sharded straight into data/ as layla__*.parquet (scripts/finalize_data_with_layla.py)
# and cleaned in place. data/ does not exist yet at this point in the pipeline
# (scripts/create_data_with_final_omnilingual.py creates it, and hard-fails if it already
# exists), so the shards are staged in their own root and copied into data/ at that step.
set -euo pipefail

WS=/workspace/asr/Palestinian-ASR
REPO=/root/Palestinian-ASR
PY=$REPO/.venv_data/bin/python
ZIP_SRC="$REPO/Layla Witheeb Jordanian Arabic Acoustic Dataset.zip"
ZIP_DEST="$WS/Layla/Layla Witheeb Jordanian Arabic Acoustic Dataset.zip"
DATASET_ROOT="$WS/Layla/Layla Witheeb Jordanian Arabic Acoustic Dataset"
SHARDS=$WS/processed_layla_shards_v1
CLEAN_OUT=$WS/data_cleaned_text_layla_v1
STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG=$REPO/.logs/layla_pipeline_$STAMP.log

exec > >(tee -a "$LOG") 2>&1
echo "=== Layla pipeline start $(date -u) ==="

# ---------- Step A: land the zip on network storage ----------
echo "=== STEP A: copy zip -> $ZIP_DEST ==="
if [ -f "$ZIP_DEST" ] && [ "$(stat -c %s "$ZIP_DEST")" = "$(stat -c %s "$ZIP_SRC")" ]; then
  echo "STEP A: already present at matching size, skipping"
else
  cp -v "$ZIP_SRC" "$ZIP_DEST"
fi
echo "STEP A: verifying archive integrity"
unzip -tqq "$ZIP_DEST"
echo "STEP A done: $(du -h "$ZIP_DEST" | cut -f1)"

# ---------- Step B: extract, reconciling with the copy already on disk ----------
echo "=== STEP B: extract -> $DATASET_ROOT ==="
# -n: never overwrite. The 2.3GB extracted copy from 07-18 is already there and verified
# complete (218/218 transcript+audio pairs); this only fills in anything the old copy lacks.
cd "$WS/Layla"
unzip -n -q "$ZIP_DEST"
echo "STEP B done: $(find -L "$DATASET_ROOT" -type f | wc -l) files, $(du -sh "$DATASET_ROOT" | cut -f1)"

# ---------- Step C: build parquet shards ----------
echo "=== STEP C: build shards -> $SHARDS ==="
"$PY" -u "$REPO/scripts/build_layla_shards.py" \
  --dataset-root "$DATASET_ROOT" \
  --output-root "$SHARDS"
echo "STEP C done: $(ls "$SHARDS/layla" | wc -l) shards"

# ---------- Step D: fast text clean ----------
echo "=== STEP D: fast text-clean -> $CLEAN_OUT ==="
"$PY" -u "$REPO/.logs/clean_layla_v1.py"
echo "STEP D done: $(ls "$CLEAN_OUT/clean" | wc -l) clean shards"

# ---------- Step E: export normalization prompt input ----------
echo "=== STEP E: export normalization input batches ==="
"$PY" -u "$REPO/scripts/export_layla_normalization_input.py" \
  --dataset-root "$DATASET_ROOT"

echo "=== ALL DONE $(date -u) ==="
