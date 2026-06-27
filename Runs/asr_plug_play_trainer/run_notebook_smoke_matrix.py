#!/usr/bin/env python3
"""Execute the universal notebook across a small model/dataset smoke matrix."""

from __future__ import annotations

import json
import os
import subprocess
import sys
from pathlib import Path
from typing import Any

BUNDLE_DIR = Path(__file__).resolve().parent
EXECUTOR = BUNDLE_DIR / "execute_notebook.py"
NOTEBOOK = Path(os.environ.get("ASR_PNP_NOTEBOOK", str(BUNDLE_DIR / "train_asr_plug_play.ipynb"))).resolve()
DATASET_PREP = BUNDLE_DIR / "prepare_smoke_datasets.py"
RUNS_DIR = BUNDLE_DIR / "runs" / "notebook_smoke_matrix"
EXECUTED_DIR = RUNS_DIR / "executed_notebooks"
SUMMARY_PATH = RUNS_DIR / "summary.json"
SUMMARY_MD_PATH = RUNS_DIR / "summary.md"

CASES = [
    {
        "name": "whisper_large_v3_short_train",
        "model_id": "openai/whisper-large-v3",
        "dataset_name": "synthetic_short_1x1x1",
        "stage_plan": "prepare,train,predict,score",
    },
    {
        "name": "whisper_medium_short_train",
        "model_id": "openai/whisper-medium",
        "dataset_name": "synthetic_short_1x1x1",
        "stage_plan": "prepare,train,predict,score",
    },
    {
        "name": "qwen_short_predict",
        "model_id": "Qwen/Qwen3-ASR-0.6B",
        "dataset_name": "synthetic_short_1x1x1",
        "stage_plan": "prepare,predict,score",
    },
    {
        "name": "omni_prepare_only",
        "model_id": "facebook/omniASR-LLM-1B",
        "dataset_name": "synthetic_short_1x1x1",
        "stage_plan": "prepare",
    },
    {
        "name": "whisper_long_filter_probe",
        "model_id": "openai/whisper-large-v3",
        "dataset_name": "synthetic_long_train_probe",
        "stage_plan": "prepare",
    },
    {
        "name": "qwen_long_keep_probe",
        "model_id": "Qwen/Qwen3-ASR-0.6B",
        "dataset_name": "synthetic_long_train_probe",
        "stage_plan": "prepare",
    },
    {
        "name": "whisper_existing_dataset_prepare",
        "model_id": "openai/whisper-large-v3",
        "dataset_name": "whisper_large_v3_levantine_1x1x1",
        "stage_plan": "prepare",
    },
]


def run(cmd: list[str], env: dict[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(cmd, cwd=BUNDLE_DIR, env=env, text=True, capture_output=True)


def read_json(path: Path) -> Any:
    if not path.exists():
        return None
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    RUNS_DIR.mkdir(parents=True, exist_ok=True)
    EXECUTED_DIR.mkdir(parents=True, exist_ok=True)

    notebook_python = sys.executable

    prep = run([notebook_python, str(DATASET_PREP)])
    if prep.returncode != 0:
        raise RuntimeError(f"Dataset prep failed:\nSTDOUT:\n{prep.stdout}\nSTDERR:\n{prep.stderr}")

    results: list[dict[str, Any]] = []
    for case in CASES:
        env = os.environ.copy()
        case_work_dir = RUNS_DIR / case["name"]
        env.update(
            {
                "ASR_PNP_MODEL_ID": case["model_id"],
                "ASR_PNP_DATASET_NAME": case["dataset_name"],
                "ASR_PNP_STAGE_PLAN": case["stage_plan"],
                "ASR_PNP_RUN_NAME": case["name"],
                "ASR_PNP_WORK_DIR": str(case_work_dir),
                "ASR_PNP_SMOKE_EPOCHS": "1",
                "ASR_PNP_SMOKE_BATCH": "1",
            }
        )
        executed_notebook = EXECUTED_DIR / f"{case['name']}.ipynb"
        cmd = [
            notebook_python,
            str(EXECUTOR),
            "--input",
            str(NOTEBOOK),
            "--output",
            str(executed_notebook),
        ]
        proc = run(cmd, env=env)
        run_result = read_json(case_work_dir / "run_result.json")
        prepared_stats = read_json(case_work_dir / "prepared" / "stats.json")
        dataset_resolved = read_json(case_work_dir / "prepared" / "dataset_resolved.json")
        score_files = sorted((case_work_dir / "scores").glob("*.json")) if (case_work_dir / "scores").exists() else []
        prediction_files = sorted((case_work_dir / "predictions").glob("*.jsonl")) if (case_work_dir / "predictions").exists() else []
        results.append(
            {
                **case,
                "returncode": proc.returncode,
                "status": "passed" if proc.returncode == 0 else "failed",
                "stdout_tail": proc.stdout[-4000:],
                "stderr_tail": proc.stderr[-4000:],
                "executed_notebook": str(executed_notebook),
                "work_dir": str(case_work_dir),
                "run_result": run_result,
                "prepared_stats": prepared_stats,
                "dataset_resolved": dataset_resolved,
                "prediction_files": [str(path) for path in prediction_files],
                "score_files": [str(path) for path in score_files],
            }
        )

    summary = {
        "bundle_dir": str(BUNDLE_DIR),
        "notebook": str(NOTEBOOK),
        "results": results,
    }
    SUMMARY_PATH.write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    lines = ["# Notebook Smoke Matrix", ""]
    for item in results:
        lines.append(f"## {item['name']}")
        lines.append(f"- status: {item['status']}")
        lines.append(f"- model_id: {item['model_id']}")
        lines.append(f"- dataset_name: {item['dataset_name']}")
        lines.append(f"- stage_plan: {item['stage_plan']}")
        stats = item.get("prepared_stats") or {}
        if stats:
            lines.append(f"- prepared_splits: {json.dumps(stats.get('splits', {}), ensure_ascii=False)}")
            lines.append(f"- dropped_too_long_train: {stats.get('dropped_too_long_train')}")
        if item.get("score_files"):
            lines.append(f"- score_files: {', '.join(item['score_files'])}")
        lines.append(f"- executed_notebook: {item['executed_notebook']}")
        lines.append("")
    SUMMARY_MD_PATH.write_text("\n".join(lines), encoding="utf-8")
    print(json.dumps({"summary_path": str(SUMMARY_PATH), "markdown_path": str(SUMMARY_MD_PATH)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
