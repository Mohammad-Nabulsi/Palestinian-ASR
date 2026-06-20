# Runs

`Runs/` contains reusable training and evaluation workflows. The structure is split into:

- `utils/`: model-agnostic helpers for manifests, metrics, prediction files, and shared result writing.
- one directory per model family. For now this repo includes `whisper_large/`.

The model notebooks are intentionally thin. They define configuration and then call high-level APIs from the corresponding model utilities.

## Expected Dataset Inputs

The real train, validation, and test paths are configured inside each notebook. Manifest files can be CSV, JSON, JSONL, or Parquet.

Each manifest should include:

- an audio path column: one of `audio_filepath`, `audio_path`, `path`, `file`, or `audio`
- a transcript column: one of `text`, `transcript`, `sentence`, or `normalized_text`

Relative audio paths are resolved relative to the manifest file.

## Shared Outputs

Every model writes to a shared `results.json` file under that model run output directory. Both baseline and tuned model entries include WER, CER, optional RTF, prediction file paths, and run metadata.

## Smoke Mode

`whisper_large/whisper_large_lora_run.ipynb` starts in `smoke_mode=True`. This creates one train, one validation, and one test WAV sample and exercises the full API flow without downloading or training Whisper Large. Set `smoke_mode=False` and update the manifest paths for a real run.
