# MohamedRashad/mgb2-arabic

## What this dataset is
- Name: `mgb2-arabic`
- HF Dataset ID: `MohamedRashad/mgb2-arabic`
- Task: ASR
- Modalities: Audio + Text
- Hub format: parquet
- Language: Arabic

## Dataset card / viewer summary
- Subset: `default`
- Split counts shown on HF viewer:
  - `train`: ~376k rows
  - `validation`: ~5k rows
  - `test`: ~5.37k rows
  - total subset: ~386k rows
- Viewer fields shown:
  - `audio`
  - `transcript`
  - `duration`
  - `quality`
  - `segment_id`
  - `recording_id`
  - `show_title`
  - `broadcast_date`
  - `genre`
  - `service`

## Processing notes
- Keep `transcript` exactly as label text for ASR.
- Keep `duration` for filtering out very short/long utterances.
- Keep program metadata (`show_title`, `broadcast_date`, `service`) for domain balancing and diagnostics.

## Recommended storage
- Primary: HF Arrow (`save_to_disk`) for training fidelity.
- Secondary: parquet for fast analytics and filtering.

## Inspect configs/splits
```bash
python3 dataset_tools/download_hf_dataset.py \
  --dataset MohamedRashad/mgb2-arabic \
  --inspect_only
```

## Download
### Non-stream
```bash
python3 dataset_tools/download_hf_dataset.py \
  --dataset MohamedRashad/mgb2-arabic \
  --mode non_stream \
  --output_dir ./datasets_storage \
  --export_parquet
```

### Stream
```bash
python3 dataset_tools/download_hf_dataset.py \
  --dataset MohamedRashad/mgb2-arabic \
  --mode stream \
  --split train \
  --output_dir ./datasets_storage \
  --stream_batch_size 2000
```

## Start working with it
### Non-stream
```python
from datasets import load_from_disk

ds = load_from_disk("datasets_storage/MohamedRashad__mgb2-arabic/non_stream/train/hf_arrow")
print(ds.column_names)
```

### Stream direct
```python
from datasets import load_dataset

stream_ds = load_dataset("MohamedRashad/mgb2-arabic", split="train", streaming=True)
print(next(iter(stream_ds)))
```

## Source
- https://huggingface.co/datasets/MohamedRashad/mgb2-arabic
