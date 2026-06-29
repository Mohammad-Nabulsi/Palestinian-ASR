# Plug-and-play ASR pipeline

This repository is a modular ASR experiment scaffold for comparing multiple ASR model families with the same high-level workflow.

Milestone 7 connects all previous pieces into one end-to-end smoke workflow:

1. load config
2. create fake smoke data
3. dispatch model name to architecture adapter
4. prepare data and verify preparation cache
5. run base prediction and verify prediction cache
6. evaluate base predictions and verify metric cache
7. run smoke LoRA training
8. run tuned prediction
9. evaluate tuned predictions
10. compare base vs tuned metrics
11. write summary and error reports

The current project is still smoke-safe. It does **not** download real models or run full fine-tuning unless future adapter implementations replace the placeholder real-training branches.

---

## Supported models

The registry maps exact model names to architecture families.

| Family | Supported model names | Smoke representative |
|---|---|---|
| Whisper | `openai/whisper-medium`, `openai/whisper-large-v3` | `openai/whisper-medium` |
| Qwen ASR | `Qwen/Qwen3-ASR-0.6B`, `Qwen/Qwen3-ASR-1.7B` | `Qwen/Qwen3-ASR-0.6B` |
| Omni ASR | `Omni ASR 300M`, `Omni ASR 1B` | `Omni ASR 300M` |

Changing `model_name` in the config is enough to switch architecture dispatch, provided the model name exists in `asr_pipeline/registry.py`.

---

## Architecture design

The project separates orchestration from model-family behavior.

### High-level APIs

The notebook calls only high-level APIs:

```python
prepare_data_and_collator(config, adapter, split_paths)
predict(config, adapter, prepared_data, split="test", tuned_adapter_path=None)
evaluate_predictions(config, prediction_path)
train(config, adapter, prepared_data, collator)
compare_base_vs_tuned(base_metrics_path, tuned_metrics_path)
```

### Adapter interface

All model families inherit from `BaseASRAdapter` and implement the same interface:

```python
prepare_dataset(...)
build_collator(...)
load_model(...)
predict(...)
train(...)
```

Architecture-specific logic stays inside:

```text
asr_pipeline/adapters/whisper.py
asr_pipeline/adapters/qwen.py
asr_pipeline/adapters/omni.py
```

The high-level pipeline should not care whether the model is Whisper, Qwen, or Omni.

---

## File structure

```text
asr_pipeline/
    __init__.py
    config.py
    registry.py
    data.py
    collators.py
    predict.py
    evaluate.py
    normalization.py
    train.py
    adapters/
        __init__.py
        base.py
        whisper.py
        qwen.py
        omni.py
    utils/
        __init__.py
        hashing.py
        io.py
        logging.py
        wandb.py
configs/
    smoke_config.yaml
notebooks/
    train_asr_plug_play.ipynb
outputs/
    prepared/
    checkpoints/
    predictions/
        base/
        tuned/
    metrics/
        base/
        tuned/
    reports/
data/
    smoke/
README_ASR_PIPELINE.md
```

---

## Setup

From the project root:

```bash
python -m venv .venv
source .venv/bin/activate
pip install pyyaml nbformat jupyter
```

Optional packages for future real training:

```bash
pip install torch transformers datasets peft wandb
```

Optional where supported later:

```bash
pip install unsloth
```

The smoke workflow is intentionally lightweight and does not require the heavy model packages.

---

## Config

The default smoke config is `configs/smoke_config.yaml`:

```yaml
model_name: openai/whisper-medium
train_path: data/smoke/train.jsonl
val_path: data/smoke/val.jsonl
test_path: data/smoke/test.jsonl
output_dir: outputs
run_name: milestone7_smoke
smoke_mode: true
wandb_mode: disabled
learning_rate: 0.00005
max_epochs: 50
early_stopping_patience: 5
seed: 42
local_model_cache_dir: models/cache
```

Important fields:

| Field | Meaning |
|---|---|
| `model_name` | Exact model name used by the registry to select an adapter |
| `train_path`, `val_path`, `test_path` | JSONL metadata files |
| `output_dir` | Root folder for prepared data, predictions, metrics, checkpoints, reports |
| `run_name` | Human-readable run name included in artifact paths |
| `smoke_mode` | When `true`, uses fake/safe training and prediction paths |
| `wandb_mode` | `disabled`, `offline`, or `online` |
| `local_model_cache_dir` | Future real model cache directory |

---

## Fake smoke data

The notebook creates fake data with:

```python
generate_fake_smoke_dataset(project_root / "data" / "smoke", overwrite=True)
```

It writes:

```text
data/smoke/train.jsonl
data/smoke/val.jsonl
data/smoke/test.jsonl
data/smoke/wav/train_smoke.wav
data/smoke/wav/val_smoke.wav
data/smoke/wav/test_smoke.wav
```

Each split has exactly one tiny 16 kHz WAV file and one JSONL row.

### Dataset schema

Every JSONL row must contain:

| Column | Description |
|---|---|
| `uid` | Unique sample ID |
| `audio_path` | Path to audio file. Relative paths are resolved relative to the JSONL file |
| `text` | Reference transcript |
| `duration` | Audio duration in seconds |
| `sample_rate` | Audio sample rate |
| `source` | Dataset/source label |

Example:

```json
{"uid":"smoke_test_0001","audio_path":"wav/test_smoke.wav","text":"مرحبا هذا مثال اختبار صغير","duration":0.25,"sample_rate":16000,"source":"fake_smoke"}
```

---

## Smoke run

Open and run:

```text
notebooks/train_asr_plug_play.ipynb
```

Or execute from the command line:

```bash
jupyter nbconvert --to notebook --execute notebooks/train_asr_plug_play.ipynb --output train_asr_plug_play_executed.ipynb
```

The notebook is organized into these sections:

1. Imports
2. Config setup
3. Smoke model list
4. Fake smoke data creation
5. Loop over smoke models
6. Final summary table
7. Error report summary
8. Instructions for switching to real training

The smoke loop runs the representative models:

```python
[
    "openai/whisper-medium",
    "Qwen/Qwen3-ASR-0.6B",
    "Omni ASR 300M",
]
```

One architecture failing does not stop the notebook. The failure is stored in `outputs/reports/` and the loop continues.

---

## Real training

Real training is intentionally scaffolded, not fully implemented yet. To switch from smoke to real training later:

1. Create a real config or edit `configs/smoke_config.yaml`.
2. Set `smoke_mode: false`.
3. Point `train_path`, `val_path`, and `test_path` to real JSONL metadata.
4. Keep `local_model_cache_dir` on persistent storage.
5. Implement the real branches in the adapters:
   - `load_model(...)`
   - `predict(...)`
   - `train(...)`
6. Keep the same notebook-level APIs.

The intended real-training behavior is:

- use LoRA fine-tuning
- use Unsloth where supported
- fall back to PEFT where Unsloth is unsupported or unavailable
- use lowest validation WER for best checkpoint selection
- log metrics and artifacts to W&B according to `wandb_mode`

---

## Caching

The pipeline uses cache keys so repeated runs skip duplicate work.

### Preparation cache

Stored under:

```text
outputs/prepared/
```

The cache key depends on:

- model family
- model name
- dataset paths
- preparation settings
- relevant config hash

The notebook verifies:

- first preparation run creates cache
- second preparation run loads cache

### Prediction cache

Base predictions:

```text
outputs/predictions/base/
```

Tuned predictions:

```text
outputs/predictions/tuned/
```

The prediction cache key depends on:

- model family
- model name
- split
- prepared dataset identity
- config hash
- base/tuned state
- tuned adapter fingerprint when applicable
- training config hash when applicable

The notebook verifies base prediction cache by running prediction twice.

### Metric cache

Base metrics:

```text
outputs/metrics/base/
```

Tuned metrics:

```text
outputs/metrics/tuned/
```

The metric cache key depends on:

- prediction file
- prediction file hash
- config hash
- base/tuned state

The notebook verifies base evaluation cache by running evaluation twice.

---

## Prediction

Use the same API for base and tuned prediction.

### Base prediction

```python
base_pred = predict(config, adapter, prepared_data, split="test")
```

Saved under:

```text
outputs/predictions/base/
```

### Tuned prediction

```python
tuned_pred = predict(
    config,
    adapter,
    prepared_data,
    split="test",
    tuned_adapter_path=best_checkpoint,
)
```

Saved under:

```text
outputs/predictions/tuned/
```

### Prediction JSONL schema

Base prediction rows include:

```text
uid
reference
prediction
model_name
model_family
tuned_or_base
run_name
config_hash
```

Tuned prediction rows additionally include:

```text
tuned_adapter_path
training_config_hash
```

---

## Evaluation

Use the same API for base and tuned prediction files:

```python
eval_result = evaluate_predictions(config, prediction_path)
```

Metrics computed:

| Metric | Meaning |
|---|---|
| `wer` | Word error rate on raw text |
| `cer` | Character error rate on raw text |
| `normalized_wer` | WER after basic normalization |
| `normalized_cer` | CER after basic normalization |
| `loose_wer` | WER after stronger loose normalization |
| `loose_cer` | CER after stronger loose normalization |

### Normalization behavior

Basic normalization:

- trims whitespace
- collapses repeated whitespace
- removes common punctuation

Loose normalization:

- removes stronger punctuation sets
- removes Arabic tatweel
- removes Arabic diacritics

The normalization is deliberately conservative for dialectal Arabic:

- it does **not** convert dialect text into MSA
- it does **not** globally convert `أ/إ/آ` to `ا`
- it does **not** globally convert `ى` to `ي`
- it does **not** globally convert `ه` to `ة`

---

## Tuned comparison

Compare base and tuned metrics with:

```python
comparison = compare_base_vs_tuned(base_metrics_path, tuned_metrics_path)
```

Reports are saved under:

```text
outputs/reports/
```

Each comparison includes:

- base WER/CER
- tuned WER/CER
- absolute improvement
- relative improvement
- normalized metric comparisons
- loose metric comparisons

Positive improvement means the tuned metric is lower than the base metric.

---

## W&B

Set `wandb_mode` in the config:

| Mode | Behavior |
|---|---|
| `disabled` | No W&B run is created |
| `offline` | Use local/offline W&B logging if the SDK is installed |
| `online` | Use normal W&B logging if the SDK is installed |

The training scaffold logs:

- model name
- model family
- dataset paths
- config hash
- hyperparameters
- LoRA configuration
- LoRA backend: Unsloth or PEFT
- epoch
- train loss
- validation loss
- WER
- CER
- best epoch
- best checkpoint path
- prediction artifacts if available
- metric artifacts if available

If W&B is unavailable, the wrapper falls back to a no-op logger instead of breaking smoke runs.

---

## Output files

Main generated artifacts:

```text
outputs/prepared/                         # prepared-data caches
outputs/predictions/base/*.jsonl          # base predictions
outputs/predictions/tuned/*.jsonl         # tuned predictions
outputs/metrics/base/*.json               # base metrics
outputs/metrics/tuned/*.json              # tuned metrics
outputs/checkpoints/*/                    # smoke checkpoints and training metadata
outputs/reports/base_vs_tuned__*.json     # comparison report JSON
outputs/reports/base_vs_tuned__*.md       # comparison report Markdown
outputs/reports/milestone7_summary.json   # final notebook summary rows
outputs/reports/milestone7_error_report.json
```

---

## Troubleshooting

### `Unsupported model_name`

The model name must exactly match a registry entry in `asr_pipeline/registry.py`.

### `Audio file does not exist`

For relative `audio_path` values, the path is resolved relative to the JSONL file. Check that the audio file exists at that resolved location.

### First-run cache assertion fails

The notebook clears smoke caches at the start. If you changed output paths or manually reused a different output directory, delete the relevant folder under `outputs/` and rerun.

### W&B import or login issue

Use:

```yaml
wandb_mode: disabled
```

or use offline mode:

```yaml
wandb_mode: offline
```

### Real training raises `NotImplementedError`

That is expected in this scaffold. Real model loading/training branches are intentionally left for later implementation.

### One architecture fails in the notebook

The notebook catches the error, saves a detailed report under `outputs/reports/`, and continues with the next architecture.

---

## How to add another model family

1. Add the model names to `asr_pipeline/registry.py`.
2. Create a new adapter file, for example:

```text
asr_pipeline/adapters/new_family.py
```

3. Inherit from `BaseASRAdapter`.
4. Implement:

```python
prepare_dataset(...)
build_collator(...)
load_model(...)
predict(...)
train(...)
```

5. Add a collator in `asr_pipeline/collators.py` if needed.
6. Add preparation rules in `asr_pipeline/data.py`.
7. Add the family to the smoke model list when ready.
8. Run the notebook and verify the summary table.

Do not duplicate code for different model sizes in the same architecture. Register the larger and smaller sizes to the same adapter.

---

## Acceptance checklist

- [x] Notebook runs top to bottom
- [x] Smoke models are tested
- [x] Cache verification passes
- [x] Predictions are saved
- [x] Metrics are saved
- [x] Training smoke pass runs or logs safe failure
- [x] Tuned comparison is saved
- [x] W&B mode works
- [x] Changing `model_name` switches architecture
