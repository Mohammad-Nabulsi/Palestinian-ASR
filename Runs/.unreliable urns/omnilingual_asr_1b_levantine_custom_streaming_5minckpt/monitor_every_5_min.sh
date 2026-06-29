#!/usr/bin/env bash
set -u

RUN_DIR="/home/MohammadNabulsi/whisper/Runs/omnilingual_asr_1b_levantine_custom_streaming_5minckpt"
LOG_PATH="$RUN_DIR/logs/health_check.log"

mkdir -p "$RUN_DIR/logs"

while true; do
  printf '[%s] ' "$(date -u +%Y-%m-%dT%H:%M:%SZ)" | tee -a "$LOG_PATH"
  python3 "$RUN_DIR/health_check.py" | tee -a "$LOG_PATH"
  status=${PIPESTATUS[0]}
  if [ "$status" -ne 0 ]; then
    echo "Health check failed; stopping monitor." | tee -a "$LOG_PATH"
    exit "$status"
  fi
  sleep 300
done
