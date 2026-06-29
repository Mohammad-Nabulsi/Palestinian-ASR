#!/usr/bin/env python3
"""Score prediction files against references using model-agnostic metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

from asr_universal_trainer import compute_wer_cer, load_config


def read_prediction_records(path: Path) -> list[dict[str, Any]]:
    if path.suffix.lower() == ".jsonl":
        with path.open("r", encoding="utf-8") as handle:
            return [json.loads(line) for line in handle if line.strip()]
    obj = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(obj, list):
        return obj
    if isinstance(obj, dict) and isinstance(obj.get("records"), list):
        return obj["records"]
    raise ValueError(f"Unsupported prediction file structure: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate prediction files with shared WER/CER logic.")
    parser.add_argument("--predictions", required=True, help="Prediction JSONL/JSON path")
    parser.add_argument("--config", default=None, help="Optional config path for metric normalizer defaults")
    parser.add_argument("--normalizer", default=None, help="Metric normalizer override")
    parser.add_argument("--output", default=None, help="Optional metrics output path")
    args = parser.parse_args()

    prediction_path = Path(args.predictions).resolve()
    records = read_prediction_records(prediction_path)
    if not records:
        raise ValueError(f"No prediction records found in {prediction_path}")

    cfg = load_config(args.config) if args.config else {}
    normalizer = args.normalizer or cfg.get("evaluation", {}).get("metric_normalizer", "arabic_basic")
    predictions = [str(record.get("prediction") or record.get("pred") or "") for record in records]
    references = [str(record.get("reference") or record.get("ref") or "") for record in records]
    metrics = compute_wer_cer(predictions, references, normalizer)
    payload = {
        "prediction_path": str(prediction_path),
        "normalizer": normalizer,
        "n": len(records),
        "split": records[0].get("split"),
        **metrics,
    }

    if args.output:
        output_path = Path(args.output).resolve()
    else:
        output_path = prediction_path.parent.parent / "scores" / f"{prediction_path.stem}_metrics.json"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({**payload, "metrics_path": str(output_path)}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
