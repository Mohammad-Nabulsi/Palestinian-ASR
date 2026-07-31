"""Watchdog for a running run_omni_eval.py process.

Every 2 minutes: checks the monitored PID is still alive, scans the run.log tail
written since the last check for OOM warnings or a crash traceback, and appends one
status line to watchdog.log. Exits (and logs a final line) once the monitored process
is no longer running.

Usage:
    python watchdog.py --pid 12345 --run-log outputs/omni_asr_llm_300m_zero_shot/run.log \
        --watchdog-log outputs/omni_asr_llm_300m_zero_shot/watchdog.log
"""
from __future__ import annotations

import argparse
import os
import re
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path

CHECK_INTERVAL_S = 120
OOM_PATTERN = re.compile(r"out of memory|OutOfMemoryError|CUDA error", re.IGNORECASE)
CRASH_PATTERN = re.compile(r"Traceback \(most recent call last\)|FATAL:", re.IGNORECASE)


def pid_alive(pid: int) -> bool:
    try:
        os.kill(pid, 0)
    except OSError:
        return False
    return True


def gpu_mem() -> str:
    try:
        out = subprocess.run(
            ["nvidia-smi", "--query-gpu=memory.used,memory.total,utilization.gpu", "--format=csv,noheader"],
            capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip().replace("\n", " | ") or "no gpu data"
    except Exception as exc:
        return f"nvidia-smi failed: {exc}"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--pid", type=int, required=True)
    ap.add_argument("--run-log", required=True)
    ap.add_argument("--watchdog-log", required=True)
    args = ap.parse_args()

    run_log = Path(args.run_log)
    watchdog_log = Path(args.watchdog_log)
    watchdog_log.parent.mkdir(parents=True, exist_ok=True)

    seen_bytes = 0
    oom_total = 0
    crash_detected = False

    def log(line: str) -> None:
        ts = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S")
        entry = f"{ts} | {line}"
        with open(watchdog_log, "a", encoding="utf-8") as f:
            f.write(entry + "\n")
        print(entry, flush=True)

    log(f"watchdog started for pid={args.pid}, watching {run_log}")

    while True:
        time.sleep(CHECK_INTERVAL_S)
        alive = pid_alive(args.pid)

        new_text = ""
        if run_log.exists():
            data = run_log.read_bytes()
            new_text = data[seen_bytes:].decode("utf-8", errors="replace")
            seen_bytes = len(data)

        new_ooms = len(OOM_PATTERN.findall(new_text))
        oom_total += new_ooms
        if CRASH_PATTERN.search(new_text):
            crash_detected = True

        status = "ALIVE" if alive else "DEAD"
        log(
            f"status={status} new_oom_events={new_ooms} total_oom_events={oom_total} "
            f"crash_traceback_seen={crash_detected} gpu=[{gpu_mem()}]"
        )

        if not alive:
            log("monitored process has exited; watchdog stopping")
            break


if __name__ == "__main__":
    main()
