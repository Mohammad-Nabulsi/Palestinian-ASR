# Model-Agnostic Utilities

This directory contains reusable helpers that should not know about Whisper or any other model architecture.

## Modules

- `data.py`: manifest loading, manifest normalization, WAV duration reading, and smoke ASR sample generation.
- `metrics.py`: dependency-light CER, WER, and optional real-time factor (RTF) calculation from saved prediction records.
- `results.py`: shared `results.json` update logic used by all model directories.

## Manifest Contract

The loader accepts CSV, JSON, JSONL, NDJSON, and Parquet. Each row must expose an audio file path and a transcript. Supported audio columns are `audio_filepath`, `audio_path`, `path`, `file`, and `audio`. Supported text columns are `text`, `transcript`, `sentence`, and `normalized_text`.

## Prediction Contract

Prediction files are JSONL. Each line should include:

- `sample_id`
- `audio_filepath`
- `reference`
- `prediction`
- optional `audio_seconds`
- optional `inference_seconds`

When `audio_seconds` and `inference_seconds` are present, metrics include RTF.
