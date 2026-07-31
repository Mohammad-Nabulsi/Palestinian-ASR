#!/usr/bin/env bash
# Stop the QASR part2 pipeline at the Step B/C boundary, so the merge into
# data_cleaned_text_merged_v1/ does not run unattended while other sessions are active.
#
# Trigger is the "STEP B done" line in the pipeline log. Step C then does an echo, a small
# cp, and a python startup (~3s of pyarrow imports) before its first shutil.move, so a 0.2s
# poll has a comfortable margin.
#
# Deliberately NOT using a broad `pkill -f merge_cleaned...`: that pattern also matches any
# shell whose command line merely mentions the merge script, including other sessions' work
# and this agent's own tool shells (it killed one on the first attempt). The only processes
# this script ever kills are the pipeline driver and its own direct children.
LOG=/root/Palestinian-ASR/.logs/qasr_part2_pipeline.nohup.log
PIPE_PID=3856

while true; do
  if grep -q "STEP B done" "$LOG" 2>/dev/null; then
    # Kill the driver first so it cannot advance, then any merge child it already spawned.
    for child in $(pgrep -P "$PIPE_PID" 2>/dev/null); do
      if tr '\0' ' ' < "/proc/$child/cmdline" 2>/dev/null | grep -q merge_cleaned_outputs_and_report; then
        kill -9 "$child" 2>/dev/null
        echo "$(date -u +%H:%M:%S) killed merge child $child"
      fi
    done
    kill -9 "$PIPE_PID" 2>/dev/null
    echo "$(date -u +%H:%M:%S) STEP B done -> pipeline driver killed, Step C held"
    exit 0
  fi
  if ! kill -0 "$PIPE_PID" 2>/dev/null; then
    echo "$(date -u +%H:%M:%S) pipeline driver exited before STEP B done"
    exit 0
  fi
  sleep 0.2
done
