from __future__ import annotations

import json
import sys
from pathlib import Path

RUN_DIR = Path(__file__).resolve().parent
NOTEBOOK_PATH = RUN_DIR / "omnilingual_asr_1b_levantine_custom_streaming_5minckpt_run.ipynb"
SUMMARY_REPORT_PATH = RUN_DIR / "summary_report.json"
TRAINING_SUMMARY_PATH = RUN_DIR / "training_summary.json"
LOG_PATH = RUN_DIR / "logs" / "omnilingual_asr_1b_run.log"
CHECKPOINT_DIR = RUN_DIR / "checkpoints"


def _load_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def notebook_errors(path: Path) -> list[str]:
    if not path.exists():
        return [f"missing notebook: {path}"]

    nb = _load_json(path)
    found: list[str] = []
    for idx, cell in enumerate(nb.get("cells", [])):
        for output in cell.get("outputs", []):
            if output.get("output_type") == "error":
                found.append(
                    f"cell {idx}: {output.get('ename', 'Error')}: {output.get('evalue', '')}".strip()
                )
    return found


def log_errors(path: Path) -> list[str]:
    if not path.exists():
        return [f"missing log: {path}"]

    lines = path.read_text(encoding="utf-8", errors="replace").splitlines()
    bad = []
    needles = ("Traceback", "RuntimeError", "CUDA out of memory", "CalledProcessError", "ERROR")
    for line in lines[-200:]:
        if any(needle in line for needle in needles):
            bad.append(line)
    return bad


def summary_warnings() -> list[str]:
    warnings: list[str] = []
    if not SUMMARY_REPORT_PATH.exists():
        warnings.append(f"missing summary report: {SUMMARY_REPORT_PATH}")
    if not TRAINING_SUMMARY_PATH.exists():
        warnings.append(f"missing training summary: {TRAINING_SUMMARY_PATH}")
    if not CHECKPOINT_DIR.exists():
        warnings.append(f"missing checkpoint dir: {CHECKPOINT_DIR}")
    return warnings


def main() -> int:
    problems = []
    problems.extend(summary_warnings())
    problems.extend(notebook_errors(NOTEBOOK_PATH))
    problems.extend(log_errors(LOG_PATH))

    if problems:
        print("UNHEALTHY")
        for problem in problems:
            print(problem)
        return 1

    print("HEALTHY")
    if SUMMARY_REPORT_PATH.exists():
        summary = _load_json(SUMMARY_REPORT_PATH)
        training = summary.get("training_summary") or {}
        print(f"checkpoint={training.get('best_checkpoint', 'n/a')}")
        print(f"completed_at={training.get('completed_at', 'n/a')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
