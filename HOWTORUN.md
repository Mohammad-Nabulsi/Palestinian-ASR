# HOW TO RUN

This document covers the full workflow in this repo after the requested updates:

- preprocessing into `data/processed`
- model-agnostic utilities in `experiments/utils`
- Whisper-specific utilities in `experiments/whisper/utils`
- executed notebooks
- baseline + fine-tuned Whisper experiments

## 1) Environment Setup

### 1.1 Create and activate venv

```bash
python3 -m venv .venv
source .venv/bin/activate
```

### 1.2 Install dependencies

```bash
pip install --upgrade pip setuptools wheel
pip install -r requirements.txt
```

### 1.3 Verify CUDA visibility

```bash
python - <<'PY'
import torch
print('torch', torch.__version__)
print('cuda_available', torch.cuda.is_available())
print('cuda_version', torch.version.cuda)
print('device_count', torch.cuda.device_count())
print('device0', torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'cpu')
PY
```

Expected on this machine:
- `torch 2.11.0+cu128`
- CUDA available `True`
- GPU `NVIDIA A100-SXM4-80GB`

## 2) Preprocessing to `data/processed`

### 2.1 What preprocessing does

Implemented in `experiments/utils/preprocessing.py`.

Filters applied:
- remove low-volume samples (`rms_db_first_seconds < -35`)
- remove samples containing digits
- remove samples containing English/non-Arabic letters
- remove punctuation from text
- remove samples that become empty after cleaning

### 2.2 Run preprocessing CLI

```bash
PYTHONPATH=. ./.venv/bin/python experiments/run_preprocessing.py \
  --manifest-root data/manifests \
  --audio-root data/local_datasets \
  --out-root data/processed \
  --low-volume-db -35 \
  --max-seconds-for-rms 3 \
  --num-workers 16
```

### 2.3 Key outputs

Under `data/processed/`:
- `processed_all.parquet`
- `preprocessing_full_with_flags.parquet`
- `processed_split_summary.csv`
- `preprocessing_report.json`
- `dataset_tables/*.csv` and `*.parquet`
- `cache/audio_rms_cache.csv`

## 3) Executed Preprocessing Notebook

Notebook:
- `experiments/preprocessing_to_processed.ipynb`

It is already executed and validates preprocessing outputs.

Re-execute if needed:

```bash
PYTHONPATH=. ./.venv/bin/python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
p = Path('experiments/preprocessing_to_processed.ipynb')
nb = nbformat.read(p, as_version=4)
nb = NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(Path('.').resolve())}}).execute()
nbformat.write(nb, p)
print('executed', p)
PY
```

## 4) Whisper Experiments

## 4.1 Script layout

Model-agnostic scripts:
- `experiments/utils/data_io.py`
- `experiments/utils/text_cleaning.py`
- `experiments/utils/audio_quality.py`
- `experiments/utils/preprocessing.py`
- `experiments/utils/metrics.py`

Whisper-specific scripts:
- `experiments/whisper/utils/config.py`
- `experiments/whisper/utils/dataset_build.py`
- `experiments/whisper/utils/inference.py`
- `experiments/whisper/utils/finetune.py`
- `experiments/whisper/utils/run.py`
- CLI: `experiments/whisper/run_whisper_experiments.py`

## 4.2 Smoke test (fast wiring check)

```bash
PYTHONPATH=. ./.venv/bin/python experiments/whisper/run_whisper_experiments.py \
  --processed-root data/processed \
  --output-root outputs/whisper_experiments_smoke \
  --smoke \
  --models openai/whisper-base \
  --train-batch 1 \
  --eval-batch 1 \
  --grad-accum 1 \
  --num-workers 1
```

## 4.3 Full requested run (base + medium, fine-tune on 5k per dataset)

```bash
PYTHONPATH=. ./.venv/bin/python experiments/whisper/run_whisper_experiments.py \
  --processed-root data/processed \
  --output-root outputs/whisper_experiments_full \
  --models openai/whisper-base openai/whisper-medium \
  --max-train-per-dataset 5000 \
  --epochs 1 \
  --max-steps -1 \
  --lr 1e-5 \
  --train-batch 8 \
  --eval-batch 8 \
  --grad-accum 2 \
  --num-workers 4
```

Notes:
- This performs baseline eval on each dataset test split.
- Then fine-tunes each model on each dataset train pool (or validation pool if train split is unavailable).
- Then evaluates fine-tuned model on the same test split.

## 4.4 Outputs

Results are saved under the output root:
- `whisper_experiment_results.csv`
- `whisper_experiment_results.json`
- baseline predictions/metrics files
- fine-tuned model folders and summaries
- fine-tuned evaluation predictions/metrics

## 5) Executed Whisper Notebook

Notebook:
- `experiments/whisper/whisper_base_medium_baseline_finetune.ipynb`

It is already executed in smoke mode and calls the scripts.

Re-execute:

```bash
PYTHONPATH=. ./.venv/bin/python - <<'PY'
from pathlib import Path
import nbformat
from nbclient import NotebookClient
p = Path('experiments/whisper/whisper_base_medium_baseline_finetune.ipynb')
nb = nbformat.read(p, as_version=4)
nb = NotebookClient(nb, timeout=1800, kernel_name='python3', resources={'metadata': {'path': str(Path('.').resolve())}}).execute()
nbformat.write(nb, p)
print('executed', p)
PY
```

## 6) Troubleshooting

### 6.1 Hugging Face auth/rate limits
If you see rate limiting, run:

```bash
huggingface-cli login
```

### 6.2 Slow runs or OOM
- Reduce `--train-batch`
- Increase `--grad-accum`
- Reduce `--max-train-per-dataset`
- Use `--models openai/whisper-base` only first

### 6.3 Start over cleanly

```bash
rm -rf outputs/whisper_experiments_full outputs/whisper_experiments_smoke
```

