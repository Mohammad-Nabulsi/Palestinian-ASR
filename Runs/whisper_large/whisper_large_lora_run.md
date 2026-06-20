# Notebook: `whisper_large_lora_run.ipynb`

This notebook is the user-facing entry point for the Whisper Large experiment. It is stage-driven: the notebook sets configuration, optionally creates smoke data, loads the model, generates baseline predictions, evaluates saved predictions, trains LoRA, generates tuned predictions, evaluates those saved predictions, and writes shared results.

## Configure Paths

In the first code cell, set:

- `train_manifest`
- `validation_manifest`
- `test_manifest`
- `output_dir`
- `model_cache_dir`
- `smoke_mode`

Leave `smoke_mode=True` for a wiring check. Set `smoke_mode=False` for the real dataset run.

## Outputs

The notebook writes:

- `outputs/predictions/baseline_test_predictions.jsonl`
- `outputs/predictions/tuned_test_predictions.jsonl`
- `outputs/metrics/baseline_metrics.json`
- `outputs/metrics/tuned_metrics.json`
- `outputs/checkpoints/whisper_large/...`
- `outputs/training_summary.json`
- `outputs/results.json`

## Cell Order

1. Imports and `WhisperLargeRunConfig`.
2. Smoke data creation, or real manifest path confirmation.
3. Baseline model load from local cache or Hugging Face.
4. Baseline prediction generation, cached after the first run.
5. Baseline evaluation from saved JSONL predictions.
6. LoRA training or checkpoint resume.
7. Tuned prediction generation from the best checkpoint.
8. Tuned evaluation from saved JSONL predictions.
9. Final summary.

Each stage logs what it is doing. Prediction generation uses a tqdm progress bar when `tqdm` is installed.

## Resume Training

Set `resume_from_checkpoint` in the config to a checkpoint directory, or leave it as `None` for a fresh run. Hugging Face Trainer checkpoints are saved under `outputs/checkpoints/whisper_large/`.

## Evaluation Policy

Baseline evaluation is cached after the first run. During training, evaluation is configured for the end of each epoch only. The default eval manifest is the test manifest because the request explicitly asked for end-of-epoch test-set evaluation. If you want a stricter research split, change the training utility to use the validation manifest for early stopping and reserve test for final reporting.

## Metrics

The notebook reports:

- WER
- CER
- RTF, when both inference time and audio duration are available
