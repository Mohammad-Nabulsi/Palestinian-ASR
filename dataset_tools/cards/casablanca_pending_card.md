# Casablanca Dataset (Pending Exact HF ID/Card)

This file is pre-created so we can fill the full per-dataset guide as soon as you share:
- exact Hugging Face dataset ID (`owner/name`)
- full dataset card text (schema, splits, notes)

## Planned rules for this dataset
- Download through `dataset_tools/download_hf_dataset.py`.
- Keep Palestinian, Jordanian, Lebanese, and Syrian dialect rows in `levant/` subdirectory.

## Temporary command skeleton
```bash
python3 dataset_tools/download_hf_dataset.py \
  --dataset <CASABLANCA_OWNER/CASABLANCA_DATASET> \
  --mode both \
  --split train \
  --output_dir ./datasets_storage \
  --export_parquet
```

## Next step
After you send the exact card, I will replace this placeholder with:
- exact schema mapping
- exact configs/splits
- exact preprocessing notes
- stream/non-stream usage examples
- dialect routing checks
