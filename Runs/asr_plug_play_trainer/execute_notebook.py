#!/usr/bin/env python3
"""Execute a notebook and persist the executed copy."""

from __future__ import annotations

import argparse
from pathlib import Path

import nbformat
from nbclient import NotebookClient


def main() -> None:
    parser = argparse.ArgumentParser(description="Execute a notebook and save the executed result.")
    parser.add_argument("--input", required=True, help="Input notebook path")
    parser.add_argument("--output", required=True, help="Executed notebook output path")
    parser.add_argument("--kernel-name", default="python3", help="Kernel name to use")
    args = parser.parse_args()

    input_path = Path(args.input).resolve()
    output_path = Path(args.output).resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    nb = nbformat.read(input_path, as_version=4)
    client = NotebookClient(
        nb,
        timeout=None,
        kernel_name=args.kernel_name,
        resources={"metadata": {"path": str(input_path.parent)}},
        allow_errors=False,
    )

    print(f"Executing notebook: {input_path}")
    client.execute()
    nbformat.write(nb, output_path)
    print(f"Wrote executed notebook: {output_path}")


if __name__ == "__main__":
    main()
