# MohamedRashad/SADA22

## What this dataset is
- Name: SADA (Saudi Audio Dataset for Arabic)
- HF Dataset ID: `MohamedRashad/SADA22`
- Modalities: Audio + Text
- HF format: parquet (on hub files)
- License: CC BY-NC-SA 4.0

## Dataset card summary
- Large Arabic speech corpus for Arabic speech AI.
- Card states over 667 hours of transcribed Arabic audio, mostly Saudi dialects.
- Card notes data sourced from 57+ TV shows; includes metadata for age/gender/dialect.
- Intended tasks in card: ASR, TTS, diarization, dialect ID, gender/age classification.

## Schema (from card/viewer)
- `audio`
- `text`
- `cleaned_text`
- `speaker_age`
- `speaker_gender`
- `speaker_dialect`

## Splits and sizes (from HF page/card)
- Subset: `default`
- Splits:
  - `train` (~242k rows)
  - `validation` (~5.14k rows)
  - `test` (~6.19k rows)
- Total rows shown: ~253,166
- Total file size shown: ~50.4 GB

## Processing notes from card context
- Keep both `text` and `cleaned_text`.
- For ASR training, choose one canonical target text policy:
  - `text` for raw fidelity
  - `cleaned_text` for normalized training targets
- Keep `speaker_dialect` metadata for dialect-aware analysis/sampling.

## Recommended storage choice
- Primary: HF Arrow (`save_to_disk`) for reproducible training.
- Secondary: parquet for EDA/analytics.
- For this dataset, metadata-first parquet is usually more efficient unless you explicitly need embedded audio payloads.

## Inspect configs/splits
```bash
python3 dataset_tools/download_hf_dataset.py \
  --dataset MohamedRashad/SADA22 \
  --inspect_only
```

## Download commands
### Non-stream (full)
```bash
python3 dataset_tools/download_hf_dataset.py \
  --dataset MohamedRashad/SADA22 \
  --mode non_stream \
  --output_dir ./datasets_storage \
  --export_parquet
```

### Stream (sharded parquet)
```bash
python3 dataset_tools/download_hf_dataset.py \
  --dataset MohamedRashad/SADA22 \
  --mode stream \
  --split train \
  --output_dir ./datasets_storage \
  --stream_batch_size 2000
```

### Both
```bash
python3 dataset_tools/download_hf_dataset.py \
  --dataset MohamedRashad/SADA22 \
  --mode both \
  --split train \
  --output_dir ./datasets_storage \
  --export_parquet
```

## Start working with it
### Non-stream
```python
from datasets import load_from_disk

ds = load_from_disk("datasets_storage/MohamedRashad__SADA22/non_stream/train/hf_arrow")
print(ds)
print(ds.column_names)
```

### Stream direct
```python
from datasets import load_dataset

stream_ds = load_dataset("MohamedRashad/SADA22", split="train", streaming=True)
for i, row in enumerate(stream_ds):
    if i == 3:
        break
    print(row["speaker_dialect"], row["cleaned_text"][:80])
```

## Citation (from card)
```bibtex
@misc{SADA2022,
  title={SADA: Saudi Audio Dataset for Arabic},
  author={SDAIA and Saudi Broadcasting Authority},
  year={2022},
  howpublished={\url{https://www.kaggle.com/datasets/sdaiancai/sada2022}},
  note={CC BY-NC-SA 4.0}
}
```
