#!/bin/bash
# Full QASR audio pipeline: download missing archive parts (part_ac, part_ad),
# then decompress+extract all 4 parts end-to-end into a single output folder.
#
# NOTE on resumability: `wget -c` against this specific Azure Blob SAS endpoint was
# measured (A/B tested, alternating flag on/off, fresh files each run) to be ~30x
# slower than a plain GET (0.3-0.6 MB/s vs 15-17 MB/s) - something about how this
# server handles the Range-request probe -c sends. Since a full restart only costs
# ~45-70 min at full speed (far less than a "resumable" download crawling at 0.5MB/s
# would cost), we deliberately do NOT use -c: on failure we delete the partial file
# and restart from scratch. This is the "skip it if it adds a lot of overhead" case
# the resumability request called out.
set -uo pipefail

QASR_DIR="/workspace/asr/Palestinian-ASR/QASR"
LOG="/root/Palestinian-ASR/.logs/qasr_full_extract_$(date -u +%Y%m%d_%H%M%S).log"
TARGET="${QASR_DIR}/wav_all"
NCPU=4

SAS='sp=rl&st=2025-03-08T09:23:55Z&se=2030-03-08T17:23:55Z&spr=https&sv=2022-11-02&sr=c&sig=D8KB7c2B5f4c6ikXd4nHl8QR620lTk3C0SMaB1rZeN0%3D'
BASE_URL='https://arabicspeechdata.blob.core.windows.net/data'

exec > >(stdbuf -oL tr '\r' '\n' >> "$LOG") 2>&1

log() { echo "[$(date -u +%Y-%m-%dT%H:%M:%S+00:00)] $*"; }

log "=== QASR full pipeline starting ==="
log "Log file: $LOG"
log "CPUs to use: $NCPU"

mkdir -p "$QASR_DIR"
cd "$QASR_DIR" || { log "ERROR: cannot cd to $QASR_DIR"; exit 1; }

declare -A EXPECTED_SIZE=(
  [qasr_wav_v1.0.tar.bz2.part_ac]=48318382080
  [qasr_wav_v1.0.tar.bz2.part_ad]=25648868607
)

for part in qasr_wav_v1.0.tar.bz2.part_ac qasr_wav_v1.0.tar.bz2.part_ad; do
  expected=${EXPECTED_SIZE[$part]}
  attempt=0
  while true; do
    actual=$(stat -c%s "$part" 2>/dev/null || echo 0)
    if [ "$actual" = "$expected" ]; then
      log "OK: $part already complete ($actual bytes)"
      break
    fi
    attempt=$((attempt+1))
    if [ "$attempt" -gt 4 ]; then
      log "ERROR: $part still not complete after $attempt attempts (have $actual, want $expected). Giving up."
      exit 1
    fi
    rm -f "$part"
    log "Downloading $part (attempt $attempt, no -c: see note above on why)..."
    wget -q -O "$part" "${BASE_URL}/${part}?${SAS}" &
    WGET_PID=$!
    START_TS=$(date +%s)
    LAST_SIZE=0
    while kill -0 "$WGET_PID" 2>/dev/null; do
      sleep 120
      CUR_SIZE=$(stat -c%s "$part" 2>/dev/null || echo 0)
      ELAPSED=$(( $(date +%s) - START_TS ))
      RATE_MBPS=$(( (CUR_SIZE - LAST_SIZE) / 120 / 1024 / 1024 ))
      PCT=$(( CUR_SIZE * 100 / expected ))
      log "progress: $part  ${CUR_SIZE}/${expected} bytes (${PCT}%)  last-120s-rate=${RATE_MBPS}MB/s  elapsed=${ELAPSED}s"
      if [ "$RATE_MBPS" -lt 3 ]; then
        log "WARNING: rate looks slow (<3MB/s over last 120s) - flagging for investigation"
      fi
      LAST_SIZE=$CUR_SIZE
    done
    wait "$WGET_PID"
    WSTATUS=$?
    actual=$(stat -c%s "$part" 2>/dev/null || echo 0)
    log "wget exited status=$WSTATUS, final size=$actual (expected $expected)"
  done
done

log "=== All 4 parts present locally ==="
ls -la "$QASR_DIR"/qasr_wav_v1.0.tar.bz2.part_*

TOTAL_BYTES=$(( 48318382080 * 3 + 25648868607 ))
log "Total compressed bytes across 4 parts: $TOTAL_BYTES"

# --- extract, all CPUs, single output folder ---
rm -rf "$TARGET"
mkdir -p "$TARGET"
log "=== BEFORE sizes ==="
df -h /workspace

log "=== Starting parallel decompression (pbzip2 -p${NCPU}) + extraction into $TARGET ==="
log "Progress line every 120s from pv (bytes of compressed input consumed):"

cat "$QASR_DIR"/qasr_wav_v1.0.tar.bz2.part_aa \
    "$QASR_DIR"/qasr_wav_v1.0.tar.bz2.part_ab \
    "$QASR_DIR"/qasr_wav_v1.0.tar.bz2.part_ac \
    "$QASR_DIR"/qasr_wav_v1.0.tar.bz2.part_ad \
  | pv -i 120 -f -s "$TOTAL_BYTES" \
  | pbzip2 -dc -p${NCPU} \
  | tar -xf - -C "$TARGET"

STATUS=$?

if [ $STATUS -ne 0 ]; then
  log "ERROR: extraction pipeline failed, exit status $STATUS"
  exit $STATUS
fi

log "=== Extraction command finished with status 0 ==="

# --- verify ---
WAV_COUNT=$(find "$TARGET" -iname "*.wav" | wc -l)
log "Total wav files extracted into $TARGET: $WAV_COUNT"

XML_COUNT=$(find "$QASR_DIR/mgb2.1/release/train_20210109/xml" -iname "*.xml" | wc -l)
log "Total xml transcripts: $XML_COUNT"

find "$TARGET" -iname "*.wav" -printf "%f\n" | sed 's/\.wav$//' | sort -u > /tmp/qasr_final_wav_names.txt
find "$QASR_DIR/mgb2.1/release/train_20210109/xml" -iname "*.xml" -printf "%f\n" | sed 's/\.xml$//' | sort -u > /tmp/qasr_final_xml_names.txt
MATCHED=$(comm -12 /tmp/qasr_final_wav_names.txt /tmp/qasr_final_xml_names.txt | wc -l)
log "xml transcripts with a matching wav: $MATCHED / $XML_COUNT"

log "=== AFTER sizes ==="
du -sh "$TARGET"
df -h /workspace

log "DONE status=success wav_count=$WAV_COUNT matched=$MATCHED xml_total=$XML_COUNT"
