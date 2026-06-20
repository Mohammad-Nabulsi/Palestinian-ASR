# Whisper Large LoRA Run

This directory contains the first model-specific run: `openai/whisper-large` without a version suffix.

## Layout

- `utils/`: Whisper-specific high-level APIs and helper modules.
- `whisper_large_lora_run.ipynb`: thin notebook that defines config, then calls high-level APIs.
- `whisper_large_lora_run.md`: notebook companion notes and run instructions.
- `outputs/`: generated predictions, metrics, checkpoints, smoke data, and `results.json`.
- `models/`: local Hugging Face cache target for `openai/whisper-large` when a real run is used.

## Real Run Behavior

When `smoke_mode=False`, the workflow:

1. Loads `openai/whisper-large` from `models/openai_whisper-large` if already cached.
2. Downloads from Hugging Face and saves locally if the model is not cached.
3. Runs baseline predictions on the test manifest once, then reuses the saved predictions on later notebook runs.
4. Evaluates baseline WER, CER, and RTF when timing and duration data are available.
5. Attaches LoRA with:
   - `r=32`
   - `target_modules=["q_proj", "k_proj", "v_proj", "fc1", "fc2"]`
   - `lora_alpha=32`
   - `lora_dropout=0.05`
   - `bias="none"`
   - `use_gradient_checkpointing="unsloth"` when Unsloth is available
   - `random_state=3407`
6. Trains for up to 10 epochs with epoch-only evaluation on the test set, early stopping patience 3, checkpointing every epoch, and best-model loading.
7. Generates tuned model predictions on the test set.
8. Writes both baseline and tuned model results into `outputs/results.json`.

## Smoke Run Behavior

The notebook defaults to `smoke_mode=True`. Smoke mode creates three one-second WAV files and one-row train, validation, and test manifests. It does not download Whisper Large or train a real model. It verifies that imports, config, manifest loading, prediction saving, metric calculation, checkpoint writing, and result writing work end to end.

## Hardware Note

This local machine currently reports a GTX 1650 with 4 GB VRAM and CPU-only PyTorch installed. That is enough for smoke validation, but not enough for practical Whisper Large LoRA training. Use a larger CUDA GPU for the real run.
