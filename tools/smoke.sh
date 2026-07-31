#!/usr/bin/env bash
# End-to-end smoke test: build the tiny fixture set, run all pipeline stages,
# then assert invariants over the result. Takes well under a minute.
#
#   ./tools/smoke.sh
#
# Uses .venv_data (python3.11, requirements-data.txt) -- no torch/transformers
# needed, because configs/sample.yaml uses the offline hash_stub dialect backend.
set -euo pipefail

cd "$(dirname "$0")/.."
PY="${PY:-./.venv_data/bin/python}"

echo "== 1/4  unit tests (text normalization pinned against original scripts)"
"$PY" tests/test_textnorm.py

echo
echo "== 2/4  build sample fixtures"
"$PY" -u tools/make_sample_data.py --output .sample_data

echo
echo "== 3/4  run pipeline"
rm -rf .sample_run
"$PY" -u -m pipeline run --config configs/sample.yaml

echo
echo "== 4/4  verify run"
"$PY" tests/verify_run.py --run-root .sample_run

echo
echo "SMOKE OK"
