#!/usr/bin/env python3
"""
Run batch dataset downloads from a manifest.

This wraps download_hf_dataset.py so you can download many datasets in one command.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path
from typing import Any, Dict, List


def build_cmd(script_path: Path, item: Dict[str, Any]) -> List[str]:
    dataset = item["dataset"]
    mode = item.get("mode", "both")
    output_dir = item.get("output_dir", "./datasets_storage")

    cmd = [
        sys.executable,
        str(script_path),
        "--dataset",
        dataset,
        "--mode",
        mode,
        "--output_dir",
        output_dir,
    ]

    if item.get("config") is not None:
        cmd += ["--config", str(item["config"])]

    if item.get("split") is not None:
        cmd += ["--split", str(item["split"])]

    if item.get("token") is not None:
        cmd += ["--token", str(item["token"])]

    if item.get("export_parquet", False):
        cmd.append("--export_parquet")

    if item.get("parquet_include_audio", False):
        cmd.append("--parquet_include_audio")

    if item.get("stream_include_audio", False):
        cmd.append("--stream_include_audio")

    if item.get("stream_batch_size") is not None:
        cmd += ["--stream_batch_size", str(item["stream_batch_size"])]

    if item.get("max_stream_rows") is not None:
        cmd += ["--max_stream_rows", str(item["max_stream_rows"])]

    return cmd


def main() -> None:
    parser = argparse.ArgumentParser(description="Batch-download HF datasets from manifest JSON.")
    parser.add_argument(
        "--manifest",
        required=True,
        help="Path to JSON manifest. See dataset_tools/datasets_manifest.example.json",
    )
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    payload = json.loads(manifest_path.read_text(encoding="utf-8"))

    if "datasets" not in payload or not isinstance(payload["datasets"], list):
        raise ValueError("Manifest must contain a 'datasets' list")

    script_path = Path(__file__).resolve().parent / "download_hf_dataset.py"

    failures = 0
    for idx, item in enumerate(payload["datasets"], start=1):
        if not item.get("enabled", True):
            print(f"[{idx}] Skipping disabled dataset: {item.get('dataset')}")
            continue

        cmd = build_cmd(script_path, item)
        print(f"\n[{idx}] Running: {' '.join(cmd)}")
        result = subprocess.run(cmd)
        if result.returncode != 0:
            failures += 1
            print(f"[{idx}] FAILED: {item.get('dataset')}")

    if failures:
        print(f"\nCompleted with {failures} failure(s).")
        raise SystemExit(1)

    print("\nAll dataset downloads completed successfully.")


if __name__ == "__main__":
    main()
