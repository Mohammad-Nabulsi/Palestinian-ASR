#!/usr/bin/env bash
# Bring the 2,584 newly-extracted QASR recordings up to the same state as the rest of the
# cleaned data (data_cleaned_text_merged_v1/): segment -> fast text-clean -> merge.
#
# The 961 recordings already represented in the merge are excluded via --wav-stem-list, and the
# output roots are named *_part2 so stable_file_id() hashes stay disjoint from the 110 QASR shards
# already merged (merge_cleaned_outputs_and_report.py hard-fails on a destination collision).
set -euo pipefail

WS=/workspace/asr/Palestinian-ASR
REPO=/root/Palestinian-ASR
PY=$REPO/.venv_data/bin/python
SEG_OUT=$WS/processed_qasr_segments_part2
CLEAN_OUT=$WS/data_cleaned_text_qasr_part2_v1
MERGED=$WS/data_cleaned_text_merged_v1
STAMP=$(date -u +%Y%m%d_%H%M%S)
LOG=$REPO/.logs/qasr_part2_pipeline_$STAMP.log

exec > >(tee -a "$LOG") 2>&1
echo "=== QASR part2 pipeline start $(date -u) ==="

# ---------- Step A: segment ----------
echo "=== STEP A: segment 2,584 new recordings -> $SEG_OUT ==="
if [ -d "$SEG_OUT/train" ] && [ -n "$(ls -A "$SEG_OUT/train" 2>/dev/null)" ]; then
  echo "STEP A: shards already present, skipping"
else
  # cwd = $WS so --wav-dir stays relative and original_audio_path is stored relative,
  # matching the QASR rows already in the merge.
  cd "$WS"
  "$PY" -u "$REPO/preprocess/qasr_segment_to_arrow.py" \
    --wav-dir QASR/wav_all/alt/arabic-speech-web/mgb2.1/wav \
    --xml-dir QASR/mgb2.1/release/train_20210109/xml \
    --output-dir "$SEG_OUT" \
    --wav-stem-list "$WS/.logs/qasr_new_stems_2584.json"
fi
echo "STEP A done: $(ls "$SEG_OUT/train" | wc -l) arrow shards"

# ---------- Step B: fast text clean ----------
echo "=== STEP B: fast text-clean -> $CLEAN_OUT ==="
cd "$WS"
"$PY" -u "$REPO/.logs/clean_qasr_part2.py"
echo "STEP B done: $(ls "$CLEAN_OUT/clean" | wc -l) clean shards"

# ---------- Step C: merge ----------
echo "=== STEP C: merge into $MERGED ==="
cp -a "$MERGED/reports/generated/merge_manifest.json" \
      "$MERGED/reports/generated/merge_manifest.pre_qasr_part2.json"
"$PY" -u "$REPO/scripts/merge_cleaned_outputs_and_report.py" \
  --source "$CLEAN_OUT" \
  --dest "$MERGED" \
  --intermediate-root "$WS/intermediate/merged_cleaned_sources"

echo "=== ALL DONE $(date -u) ==="
echo "merged clean shards: $(ls "$MERGED/clean" | wc -l)"
echo "  qasr part1: $(ls "$MERGED/clean" | grep -c '^processed_qasr_segments__' || true)"
echo "  qasr part2: $(ls "$MERGED/clean" | grep -c '^processed_qasr_segments_part2__' || true)"
