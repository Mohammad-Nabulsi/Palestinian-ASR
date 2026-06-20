# Dataset Tools

This folder is a reusable pipeline for Hugging Face dataset downloads based on dataset cards.

## Why there is a single script
- `download_hf_dataset.py` is the core engine.
- You change `--dataset` (and optional config/split) per dataset.
- This avoids duplicating code per dataset.

If you want to run many datasets together, use manifest mode with `download_from_manifest.py`.

## Files
- `dataset_tools/download_hf_dataset.py`: single-dataset downloader (stream + non-stream)
- `dataset_tools/download_from_manifest.py`: batch runner for many datasets
- `dataset_tools/datasets_manifest.example.json`: manifest template
- `dataset_tools/cards/`: per-dataset markdown docs

## Install
```bash
pip install -U datasets pyarrow pandas
```

## Single dataset usage
Inspect configs/splits first:
```bash
python3 dataset_tools/download_hf_dataset.py \
  --dataset <owner/dataset> \
  --inspect_only
```

Download full non-stream copy + parquet export:
```bash
python3 dataset_tools/download_hf_dataset.py \
  --dataset <owner/dataset> \
  --mode non_stream \
  --output_dir ./datasets_storage \
  --export_parquet
```

Stream to sharded parquet:
```bash
python3 dataset_tools/download_hf_dataset.py \
  --dataset <owner/dataset> \
  --mode stream \
  --split train \
  --output_dir ./datasets_storage
```

## Batch mode (many datasets)
1. Copy and edit manifest:
```bash
cp dataset_tools/datasets_manifest.example.json dataset_tools/datasets_manifest.json
```
2. Set each dataset entry (`dataset`, `split`, options, enabled true/false).
3. Run:
```bash
python3 dataset_tools/download_from_manifest.py \
  --manifest dataset_tools/datasets_manifest.json
```

## Storage recommendation for large ASR datasets
- Primary storage: HF Arrow (`save_to_disk`) for full fidelity and training stability.
- Secondary storage: parquet for analytics and tabular workflows.
- Default parquet behavior is metadata-first (audio excluded unless you pass include-audio flags).

## Levant routing
For configured datasets (currently Omnilingual preset), rows matching Palestinian/Jordanian/Lebanese/Syrian are additionally saved under `levant/`.

When Casablanca exact HF dataset ID is provided, add it to the preset list in `download_hf_dataset.py` to auto-enable same routing.

## Dataset docs currently present
- `dataset_tools/cards/INDEX.md`
- `dataset_tools/cards/facebook_omnilingual_asr_corpus.md`
- `dataset_tools/cards/mohamedrashad_sada22.md`
- `dataset_tools/cards/casablanca_pending_card.md`
- `dataset_tools/cards/TEMPLATE_dataset_card.md`
