#!/usr/bin/env bash
# Polls a progress_latest.json every 2 minutes and appends a one-line summary
# to a report log, so Monitor can surface progress without active polling.
set -u
STATE_JSON="$1"
REPORT_LOG="$2"
LABEL="${3:-run}"

while true; do
  sleep 120
  if [ -f "$STATE_JSON" ]; then
    python3 - "$STATE_JSON" "$LABEL" "$REPORT_LOG" <<'EOF'
import json, sys, datetime
state_path, label, report_log = sys.argv[1], sys.argv[2], sys.argv[3]
try:
    with open(state_path) as f:
        s = json.load(f)
    total = s.get('total_rows') if s.get('total_rows') is not None else s.get('total_candidates')
    completed = s.get('completed_total')
    elapsed = s.get('elapsed_seconds')
    rate = s.get('rows_per_second')
    eta_human = s.get('estimated_remaining_human')
    if rate is None and elapsed and completed:
        rate = round(completed / elapsed, 2) if elapsed > 0 else None
    if eta_human is None and rate and total is not None:
        remaining = max(total - completed, 0)
        eta_sec = remaining / rate if rate else None
        eta_human = f"{eta_sec/60:.1f}min" if eta_sec is not None else None
    line = (
        f"[{datetime.datetime.utcnow().isoformat()}Z] {label} "
        f"progress={s.get('overall_progress_percentage')}% "
        f"completed={completed}/{total} "
        f"rows_per_sec={rate} "
        f"eta={eta_human} "
        f"errors={s.get('errors_total')} "
        f"current_uid={s.get('current_uid')}"
    )
except Exception as e:
    line = f"[{datetime.datetime.utcnow().isoformat()}Z] {label} progress=unavailable ({e})"
with open(report_log, "a") as out:
    out.write(line + "\n")
EOF
  else
    echo "[$(date -u +%FT%TZ)] $LABEL progress=not_started_yet (no state file at $STATE_JSON)" >> "$REPORT_LOG"
  fi
done
