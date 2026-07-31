#!/usr/bin/env bash
# Supervisor: runs the MASC-C text-dialect scan, auto-restarting with --resume on any crash.
set -u
cd /root/Palestinian-ASR

PY=/workspace/venv_nemo_gpu/bin/python
SCRIPT=dialect_identifiaction/arabic_text_dialect_scan_marbertv2_written.py
OUT=Runs/text_dialect_scan_marbertv2_written_clean_masc_c_only

mkdir -p "$OUT"
mapfile -t DATA_ROOTS < <(ls data/clean/masc_c_only__*.parquet)
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
    --allowed-source masc_c \
    --device cuda \
    --batch-size 2048 \
    --log-every 5000 \
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
