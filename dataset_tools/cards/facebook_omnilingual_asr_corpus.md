# facebook/omnilingual-asr-corpus

## What this dataset is
- Name: Meta Omnilingual ASR Corpus
- Source: Meta FAIR Omnilingual ASR project
- Goal: spontaneous speech + transcripts for under-served languages
- License: CC-BY-4.0

## Schema (from your provided card)
- `language` (format `{iso639_3}_{iso15924}` like `lij_Latn`)
- `iso_639_3`
- `iso_15924`
- `glottocode`
- `prompt_id`
- `prompt`
- `speaker_id`
- `segment_id`
- `audio` (FLAC)
- `raw_text`

## Transcription conventions to keep
- Special tags:
  - `<laugh>`
  - `<hesitation>`
  - `<unintelligible>`
  - `<noise>`
- Disfluencies and false starts are intentional data, keep as-is.

## Recommended storage choice
- Best primary storage for this size/type: **HF Arrow (`save_to_disk`)**.
  - Keeps full feature typing and audio references safely.
  - Better for training pipelines in `datasets`.
- Parquet is still useful for analytics/inspection.
  - In this setup, parquet defaults to metadata-first (without `audio`) to keep files manageable.

## Check configs/splits first
```bash
python3 dataset_tools/download_hf_dataset.py \
  --dataset facebook/omnilingual-asr-corpus \
  --inspect_only
```

## Non-stream download (full)
```bash
python3 dataset_tools/download_hf_dataset.py \
  --dataset facebook/omnilingual-asr-corpus \
  --mode non_stream \
  --output_dir ./datasets_storage \
  --export_parquet
```

## Stream mode (sharded parquet)
```bash
python3 dataset_tools/download_hf_dataset.py \
  --dataset facebook/omnilingual-asr-corpus \
  --mode stream \
  --split train \
  --output_dir ./datasets_storage \
  --stream_batch_size 2000
```

## Both modes together
```bash
python3 dataset_tools/download_hf_dataset.py \
  --dataset facebook/omnilingual-asr-corpus \
  --mode both \
  --split train \
  --output_dir ./datasets_storage \
  --export_parquet
```

## Levant routing behavior
For this dataset preset, the script auto-routes matched rows for:
- Palestinian
- Jordanian
- Lebanese
- Syrian

into:
- `.../levant/non_stream/...`
- `.../levant/stream/...`

instead of leaving them only under the main root.

## Output structure
```text
datasets_storage/
  facebook__omnilingual-asr-corpus/
    download_report.json
    non_stream/
      <split>/
        hf_arrow/
        parquet/<split>.parquet
    stream/
      <split>/
        parquet/<split>.part-00000.parquet
    levant/
      non_stream/
        <split>/
          hf_arrow/
          <split>.parquet
      stream/
        <split>/
          parquet/<split>.part-00000.parquet
```

## Start working with it
### Non-stream read
```python
from datasets import load_from_disk

train_ds = load_from_disk("datasets_storage/facebook__omnilingual-asr-corpus/non_stream/train/hf_arrow")
print(train_ds)
print(train_ds.column_names)
```

### Stream read directly from HF
```python
from datasets import load_dataset

stream_ds = load_dataset("facebook/omnilingual-asr-corpus", split="train", streaming=True)
for i, row in enumerate(stream_ds):
    if i == 3:
        break
    print(row["language"], row["raw_text"][:80])
```

### Read stream parquet shards later
```python
import glob
import pandas as pd

files = sorted(glob.glob("datasets_storage/facebook__omnilingual-asr-corpus/stream/train/parquet/*.parquet"))
df = pd.concat((pd.read_parquet(f) for f in files), ignore_index=True)
print(df.head())
```

## Citation
Use the citation in the dataset card:
`@misc{omnilingualasr2025, ... url={https://arxiv.org/abs/2511.09690}}`
