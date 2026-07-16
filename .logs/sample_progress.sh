#!/bin/bash
cd /root/Palestinian-ASR
OUT=.logs/progress_samples.csv
if [ ! -f "$OUT" ]; then
  echo "epoch,qasr_audio_files_done,qasr_segments_emitted,masc_shards_done" > "$OUT"
fi
while true; do
  ts=$(date +%s)
  qasr_line=$(grep -o "Progress: [0-9]* audio files, [0-9]* emitted segments" .logs/qasr_segment_to_arrow.log 2>/dev/null | tail -1)
  qasr_files=$(echo "$qasr_line" | grep -o "^Progress: [0-9]*" | grep -o "[0-9]*")
  qasr_segs=$(echo "$qasr_line" | grep -o "[0-9]* emitted" | grep -o "^[0-9]*")
  masc_done=$(find data/masc_c_only/data -maxdepth 1 -iname "*.parquet" 2>/dev/null | wc -l)
  echo "${ts},${qasr_files:-0},${qasr_segs:-0},${masc_done:-0}" >> "$OUT"
  sleep 30
done
