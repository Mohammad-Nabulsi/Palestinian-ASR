#!/usr/bin/env bash
# Runs the layla+jor pal-study variants sequentially on the single GPU, waiting for any
# currently-running pal_s1e1of2_run.py job to finish first. Each run's own stdout/stderr is
# already teed to Runs/whisper_medium_pal/<RUN_TAG>/logs/ by the script itself; this wrapper's
# log just tracks orchestration (waits, launches, exit codes).
set -uo pipefail
cd /root/Palestinian-ASR
PY=/workspace/venv_qwen_gpu/bin/python
ORCH_LOG=/root/Palestinian-ASR/Runs/whisper_medium_pal/layla_jor_study_orchestrator.log
mkdir -p /root/Palestinian-ASR/Runs/whisper_medium_pal
exec >>"$ORCH_LOG" 2>&1

log() { echo "[$(date '+%Y-%m-%d %H:%M:%S')] $*"; }

log "orchestrator started, waiting for any running pal_*_run.py to finish"
while pgrep -f "pal_.*_run\.py" >/dev/null 2>&1; do
    sleep 20
done
log "GPU free, starting layla_jor study"

run() {
    local tag="$1"; shift
    log "=== launching $tag ($*) ==="
    env "$@" "$PY" -u pal_layla_jor_run.py
    local rc=$?
    log "=== $tag exited rc=$rc ==="
    return $rc
}

# 1) stage1 = 1 epoch on layla+jor, merge, stage2 = early-stopping default (<=50ep, patience 3)
run "run5_layla_jor" PAL_RUN=run5_layla_jor PAL_STAGE1_EPOCHS=1 PAL_STAGE2_EARLYSTOP=1
rc1=$?

# 2) stage1 = 2 epochs on layla+jor, merge LAST epoch, stage2 = 1 fixed epoch
run "run5_layla_jor_s1ep2" PAL_RUN=run5_layla_jor_s1ep2 PAL_STAGE1_EPOCHS=2 PAL_STAGE2_EPOCHS=1
rc2=$?

# 3) reuse run 2's stage-1 (2-epoch) checkpoint, stage2 = 2 fixed epochs
if [ $rc2 -eq 0 ]; then
    S1_CKPT=$($PY -c "
import json
d = json.load(open('/root/Palestinian-ASR/Runs/whisper_medium_pal/run5_layla_jor_s1ep2/SUMMARY.json'))
print(d['train']['stage1']['last_epoch_dir'])
")
    log "reusing stage1 checkpoint: $S1_CKPT"
    run "run5_layla_jor_s1ep2_s2ep2" PAL_RUN=run5_layla_jor_s1ep2_s2ep2 PAL_STAGE1_EPOCHS=2 \
        PAL_STAGE1_CKPT="$S1_CKPT" PAL_STAGE2_EPOCHS=2
    rc3=$?
else
    log "SKIPPING run5_layla_jor_s1ep2_s2ep2: run5_layla_jor_s1ep2 failed (rc=$rc2)"
    rc3=1
fi

log "orchestrator done: run5_layla_jor rc=$rc1 | run5_layla_jor_s1ep2 rc=$rc2 | run5_layla_jor_s1ep2_s2ep2 rc=$rc3"
