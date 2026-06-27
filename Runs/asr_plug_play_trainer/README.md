# ASR Plug-and-Play Trainer

This package gives you one notebook that calls one Python script. The dataset is normalized once into a common ASR schema, then the script chooses the correct model-family adapter. The notebook now exposes separate `prepare`, `train`, `predict`, and `score` phases so prediction generation is decoupled from metric scoring.

## Files

| File | Purpose |
|---|---|
| `train_asr_plug_play.ipynb` | Notebook front-end. Change model/dataset names, run preparation/training/prediction/scoring, and inspect outputs. |
| `asr_universal_trainer.py` | Actual pipeline: dataset normalization, model registry, collators, training/eval backends. |
| `example_config.yaml` | Main config template. Change `model.model_id`, data path, and column names. |
| `qwen_eval_config.yaml` | Qwen3-ASR eval/data-format example. |
| `requirements.txt` | Suggested Python packages. |

## Environment

Recommended on your A100 VM:

```bash
cd /home/MohammadNabulsi/whisper/asr_plug_play_trainer
python3.12 -m venv .venv
source .venv/bin/activate
pip install -U pip wheel setuptools
# Install torch matching your CUDA first. Example for CUDA 12.8:
pip install torch torchaudio --index-url https://download.pytorch.org/whl/cu128
pip install -r requirements.txt
```

Optional extras:

```bash
# Qwen3-ASR inference/eval
pip install -U qwen-asr

# Omnilingual ASR inference tools
pip install -U omnilingual-asr

# NVIDIA NeMo backend
pip install 'nemo_toolkit[asr]'
```

## Data format

Your dataset can be JSONL, JSON, CSV, Parquet, NeMo-style manifest, or Hugging Face dataset. The script converts everything into:

```json
{
  "uid": "utt_001",
  "audio_path": "/abs/or/relative/audio.wav",
  "text": "transcript here",
  "duration": 4.2,
  "split": "train",
  "language": "ar",
  "speaker_id": "optional"
}
```

Minimum required columns are audio path and text. Duration is recommended but can be inferred for WAV files.

## How to run

For everyday use, start from `train_asr_plug_play.ipynb` and change the model/dataset names there.

## Current Verification Status

As of 2026-06-25, the notebook-driven smoke matrix was executed successfully and saved under `runs/notebook_smoke_matrix/`.

Verified cases:

- Whisper large-v3: `prepare -> train -> predict -> score` on `synthetic_short_1x1x1`
- Whisper medium: `prepare -> train -> predict -> score` on `synthetic_short_1x1x1`
- Qwen3-ASR-0.6B: `prepare -> predict -> score` on `synthetic_short_1x1x1`
- Omnilingual 1B: `prepare` on `synthetic_short_1x1x1`
- Omnilingual 300M/1B: local recipe-backed `prepare -> train -> eval` is now wired through `third_party/omnilingual-asr` when that repo and its env are available
- Dataset-name switching: `whisper_large_v3_levantine_1x1x1` prepared successfully by changing only the dataset name
- Long-audio policy: Whisper dropped the over-limit train sample; Qwen kept it and produced chat-style rows

Summary artifacts:

- `runs/notebook_smoke_matrix/summary.md`
- `runs/notebook_smoke_matrix/summary.json`
- executed notebooks under `runs/notebook_smoke_matrix/executed_notebooks/`

### 1. Smoke test

```bash
python asr_universal_trainer.py --smoke-test --stage all --work-dir ./runs/smoke
```

This creates 1 train, 1 validation, and 1 test WAV file, runs the mock trainer, prints train loss, validation loss, WER, CER, and writes logs.

### 2. Prepare your real dataset

Edit `example_config.yaml`:

```yaml
model:
  model_id: openai/whisper-large-v3

data:
  format: jsonl
  path: /home/MohammadNabulsi/whisper/your_data.jsonl
  columns:
    audio: audio_path
    text: text
    duration: duration
    split: split
```

Then run:

```bash
python asr_universal_trainer.py --config example_config.yaml --stage prepare
```

Outputs go to `output.work_dir/prepared/`:

- `train.jsonl`
- `validation.jsonl`
- `test.jsonl`
- `stats.json`
- `nemo_manifests/train.json`, `validation.json`, `test.json`

### 3. Train and evaluate

```bash
python asr_universal_trainer.py --config example_config.yaml --stage all
```

For Hugging Face Whisper/CTC backends:

- Evaluation runs once per epoch.
- Printed metrics include WER and CER.
- Best checkpoint is selected by lowest WER.
- Early stopping patience is controlled by `training.early_stopping_patience` and defaults to 5.

## Model-family behavior

| Model family | Backend | Training in this script | Eval in this script | Important note |
|---|---:|---:|---:|---|
| Whisper medium/large-v3 | HF Seq2SeqTrainer | Yes | Yes after training | 30s receptive field; prepare segments <=30s. |
| Generic CTC: wav2vec2, WavLM, HuBERT | HF Trainer | Yes | Yes after training | Uses CTC collator with separate audio/label padding. |
| Qwen3-ASR | qwen-asr | Data export + eval | Yes if `qwen-asr` installed | Official card exposes inference/chat/vLLM usage. Training is not treated as stable in this generic script. |
| Omnilingual ASR LLM/CTC | local omnilingual-asr recipe | Yes | Yes | Uses the local `third_party/omnilingual-asr` recipe and inference pipeline; model IDs like `facebook/omniASR-LLM-300M-v2` are mapped automatically. |
| Cohere Transcribe 03-2026 | Transformers pipeline | No | Yes | Public card exposes inference; do not assume fine-tuning. |
| NVIDIA Arabic FastConformer hybrid | NeMo | External command generated | External command generated | NeMo manifest files are created automatically. |

## Changing models

Usually you only change:

```yaml
model:
  model_id: openai/whisper-large-v3
```

The registry will infer the backend. If it guesses wrong, force it:

```yaml
model:
  model_id: your/model
  spec_overrides:
    backend: hf_ctc
    family: hf_ctc
    max_train_seconds: 20
```

## Duration policy

For models with a known training duration cap, the script filters train rows using `model.max_train_seconds`.

```yaml
data:
  long_audio_policy: drop
```

Options:

- `drop`: safest; removes over-limit train samples.
- `error`: fail loudly if any train sample is too long.
- `keep`: keep over-limit samples, only if you know the model supports it.
- `first_chunk`: writes the first N seconds as a WAV. This is technically runnable but can be label-wrong unless the transcript matches that chunk.

## Logs

Everything logs to:

```text
<work_dir>/run.log
<work_dir>/metrics.csv
<work_dir>/run_result.json
```

The notebook tails and displays these files after each command.

## Notes on guarantees

The old mock-only smoke test still exists, but the current notebook verification is stronger: the saved smoke matrix confirms real notebook execution for Whisper training plus separate prediction/scoring, Qwen prepare plus prediction/scoring, Omnilingual prepare/export, dataset-name switching, and long-audio policy behavior. Real model work still requires the relevant packages, GPU, and access tokens where applicable. The script intentionally does not claim that Cohere, Qwen, or Omnilingual all share the same fine-tuning API as Whisper/CTC models; instead it keeps one notebook interface while making the backend-specific supported phases explicit.


For Omnilingual runs, set `model.model_id` to something like `facebook/omniASR-LLM-300M-v2`, keep audio at 16 kHz, and run `--stage all`. The trainer will prepare the universal JSONL, convert it into the local Omnilingual parquet/card format under `<work_dir>/omnilingual_recipe/`, launch the official recipe from `third_party/omnilingual-asr`, then evaluate the tuned checkpoint if one exists.
