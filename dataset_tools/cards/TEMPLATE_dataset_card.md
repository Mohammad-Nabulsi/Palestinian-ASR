# <owner/dataset>

## What this dataset is
- Name:
- Source:
- License:
- Task type:

## Dataset card summary
- Short description:
- Intended use:
- Known limitations:

## Schema
- List fields exactly as in card.

## Configs and splits
Use:
```bash
python3 dataset_tools/download_hf_dataset.py --dataset <owner/dataset> --inspect_only
```
Then record:
- Configs:
- Splits per config:

## Recommended storage for this dataset
- Primary:
- Secondary:
- Why:

## Download commands
### Non-stream
```bash
python3 dataset_tools/download_hf_dataset.py --dataset <owner/dataset> --mode non_stream --output_dir ./datasets_storage --export_parquet
```

### Stream
```bash
python3 dataset_tools/download_hf_dataset.py --dataset <owner/dataset> --mode stream --split train --output_dir ./datasets_storage
```

### Batch manifest mode
1) Add dataset to `dataset_tools/datasets_manifest.example.json`
2) Run:
```bash
python3 dataset_tools/download_from_manifest.py --manifest dataset_tools/datasets_manifest.example.json
```

## Output structure
- Show expected folders and files for this dataset.

## Start working with it
### Non-stream (HF Arrow)
```python
from datasets import load_from_disk

# update split path
# ds = load_from_disk("datasets_storage/<dataset_slug>/non_stream/train/hf_arrow")
```

### Stream (direct from HF)
```python
from datasets import load_dataset

# stream_ds = load_dataset("<owner/dataset>", split="train", streaming=True)
```

### Stream parquet shards
```python
import glob
import pandas as pd

# files = sorted(glob.glob("datasets_storage/<dataset_slug>/stream/train/parquet/*.parquet"))
# df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
```

## Notes
- Add any preprocessing and text normalization from the card.
