#!/usr/bin/env bash
# Full-acoustic v2 supervisor: scan -> score -> extract -> forward_v2 training.
#
# Every phase is independently resumable (the scan keeps a per-shard ledger; the
# trainer checkpoints adapter + optimizer + scheduler + RNG), so a retry continues
# rather than restarting -- and in particular never re-enters the LR schedule at
# the top. Each phase retries up to MAX_RETRIES and then stops with a .stuck
# marker; a persistent bug wants a human, not a hot loop.
#
# Training goes through train_whisper_medium_lora_sequence.py. `--sequence
# 1,2,3,4,1,2,3,4` is exactly the forward 8-stage run that
# train_whisper_medium_lora_progressive.py hardcodes, but the sequence script is
# also the one that accepts --init-from/--init-stage-offset, so continuation runs
# resume at the left-off LR. Keeping every run on one entry point avoids two
# divergent copies of the training loop.
#
# STATE must be durable (/workspace, R2) and must never be a session scratchpad --
# /tmp/claude-*/**/scratchpad is deleted with its session, which is what destroyed
# the v1 dataset. TEXT likewise points at persisted copies of the text scans.
#
# R2 credentials come from the environment / rclone config, never from this file:
#   export RCLONE_CONFIG_R2_TYPE=s3 RCLONE_CONFIG_R2_PROVIDER=Cloudflare
#   export RCLONE_CONFIG_R2_ACCESS_KEY_ID=... RCLONE_CONFIG_R2_SECRET_ACCESS_KEY=...
#   export RCLONE_CONFIG_R2_ENDPOINT=https://<account>.r2.cloudflarestorage.com
set -uo pipefail
ROOT="${ROOT:-/root/Palestinian-ASR}"
STATE="${STATE:-/workspace/asr_env/full_acoustic_v2}"
TEXT="${TEXT:-$STATE/text_scans}"
TMP="$STATE/work"
LOG="${LOG:-$STATE/master.log}"
MAX_RETRIES="${MAX_RETRIES:-8}"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$STATE" "$TMP"
exec >>"$LOG" 2>&1
cd "$ROOT"
echo "$(date -Is) supervisor started"

retry() {
  local name="$1"; shift
  local attempt=1
  until "$@"; do
    echo "$(date -Is) $name FAILED (attempt $attempt/$MAX_RETRIES)"
    if [[ $attempt -ge $MAX_RETRIES ]]; then
      echo "$(date -Is) $name exceeded $MAX_RETRIES retries -- STOPPING, needs manual fix"
      touch "$STATE/$name.stuck"
      return 1
    fi
    attempt=$((attempt+1))
    sleep 60
  done
  touch "$STATE/$name.done"
  rm -f "$STATE/$name.stuck"
  echo "$(date -Is) $name COMPLETE"
  return 0
}

# Phase 1: Badrex acoustic dialect ID over every QASR and MASC row in both leaves.
if [[ ! -e "$STATE/scan.done" ]]; then
  retry scan python3 scripts/full_acoustic_scan.py \
    --output-root "$STATE/acoustic_full_scan" --work-dir "$TMP/scan" || exit 1
fi

# Phase 2: weighted (0.3 text / 0.7 acoustic) speaker-disjoint blocks over every
# speaker with acoustic coverage: top 8h -> test, next 8h -> val, next 200h -> train.
if [[ ! -e "$STATE/score.done" ]]; then
  retry score python3 scripts/build_full_acoustic_speaker_split.py \
    --text-scan "$TEXT/text_dialect_scan_marbertv2_written_clean_qasr_only/row_probabilities.jsonl" \
                "$TEXT/text_dialect_scan_marbertv2_written_clean_masc_c_only/row_probabilities.jsonl" \
    --audio-root "$STATE/acoustic_full_scan" --out-dir "$STATE/split" || exit 1
fi

# Phase 3: pull every sample of the selected speakers out of curated_corpus.
if [[ ! -e "$STATE/extract.done" ]]; then
  rm -f "$STATE/data/train.parquet" "$STATE/data/val.parquet" "$STATE/data/test.parquet"
  retry extract python3 scripts/extract_full_acoustic_speaker_split.py \
    --assignments "$STATE/split/speaker_assignments.json" \
    --out-dir "$STATE/data" --work-dir "$TMP/extract" || exit 1
fi

# Phase 4: the forward experiment on the v2 dataset -- 8 stages of 50h chunks
# (checkpoint + val/test eval after each), one continuous OneCycleLR at max_lr=1e-4.
if [[ ! -e "$STATE/train_v2.done" ]]; then
  retry train_v2 python3 scripts/train_whisper_medium_lora_sequence.py \
    --sequence 1,2,3,4,1,2,3,4 --run-name forward_v2 \
    --train-parquet "$STATE/data/train.parquet" \
    --rank-meta "$STATE/split/train_full_rank.json" \
    --val-parquet "$STATE/data/val.parquet" \
    --test-parquet "$STATE/data/test.parquet" \
    --out-dir "$STATE/whisper_medium_lora_forward_v2" \
    --lr 1e-4 || exit 1
fi

echo "$(date -Is) ALL PHASES COMPLETE"
