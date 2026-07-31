#!/usr/bin/env bash
# Supervisor: runs the QASR audio-dialect scan (badrex mms300m) on the text-stage
# LEV>=0.80 candidates, auto-restarting with --resume on any crash.
set -u
cd /root/Palestinian-ASR

PY=/workspace/venv_nemo_gpu/bin/python
SCRIPT=dialect_identifiaction/arabic_dialect_scan_badrex_mms300m.py
OUT=Runs/dialect_scan_badrex_mms300m_lev08_text_candidates_qasr_only
TEXT_PROBS=Runs/text_dialect_scan_marbertv2_written_clean_qasr_only/row_probabilities.jsonl

mkdir -p "$OUT"

mapfile -t DATA_ROOTS < <(ls data/clean/processed_qasr_segments*__*.parquet)
ARGS=()
for f in "${DATA_ROOTS[@]}"; do
  ARGS+=(--data-root "$f")
done

attempt=0
while true; do
  attempt=$((attempt + 1))
  echo "[supervisor] attempt $attempt starting at $(date -u +%FT%TZ)" >> "$OUT/supervisor.log"
  "$PY" -u "$SCRIPT" \
    "${ARGS[@]}" \
    --output-dir "$OUT" \
    --text-probabilities-path "$TEXT_PROBS" \
    --text-target-label LEV \
    --text-target-threshold 0.80 \
    --device cuda \
    --memory-fraction 0.35 \
    --log-every 200 \
    --resume \
    >> "$OUT/stdout.log" 2>&1
  ec=$?
  echo "[supervisor] attempt $attempt exited code=$ec at $(date -u +%FT%TZ)" >> "$OUT/supervisor.log"
  if [ "$ec" -eq 0 ]; then
    echo "[supervisor] completed successfully" >> "$OUT/supervisor.log"
    break
  fi
  sleep 10
done
