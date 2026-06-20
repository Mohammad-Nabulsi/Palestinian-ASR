# MohamedRashad/MASC-Arabic

## What this dataset is
- Name: MASC Arabic
- HF Dataset ID: `MohamedRashad/MASC-Arabic`
- Tasks: ASR, TTS (as tagged on page)
- Modalities: Audio + Text
- Hub format: parquet
- Language: Arabic
- License on page: `cc-by-4.0`

## Dataset card summary (from HF page)
- MASC is described as ~1,000 hours of 16kHz Arabic speech crawled from 700+ YouTube channels.
- Multi-regional, multi-genre, multi-dialect.
- Intended for advancing Arabic speech technology, especially ASR.

## Fields (from card section)
- `video_id`: source video id
- `start`: chunk start
- `end`: chunk end
- `duration`: chunk duration
- `text`: transcript
- `type`: clean/noisy indicator (`c` clean, `n` noisy)
- `audio`: chunk audio

## Splits (viewer)
- Subset: `default` (~913k rows)
- `train`: ~876k
- `validation`: ~19.5k
- `test`: ~18k

## Processing notes
- Use `type` for optional clean-only training or clean/noisy curriculum.
- Keep `start/end/duration` for timing consistency checks.
- If normalizing text, preserve original `text` in parallel.

## Recommended storage
- Primary: HF Arrow for training reproducibility.
- Secondary: parquet for EDA and tabular workflows.

## Inspect configs/splits
```bash
python3 dataset_tools/download_hf_dataset.py \
  --dataset MohamedRashad/MASC-Arabic \
  --inspect_only
```

## Download
### Non-stream
```bash
python3 dataset_tools/download_hf_dataset.py \
  --dataset MohamedRashad/MASC-Arabic \
  --mode non_stream \
  --output_dir ./datasets_storage \
  --export_parquet
```

### Stream
```bash
python3 dataset_tools/download_hf_dataset.py \
  --dataset MohamedRashad/MASC-Arabic \
  --mode stream \
  --split train \
  --output_dir ./datasets_storage \
  --stream_batch_size 2000
```

## Citation (from card)
```bibtex
@INPROCEEDINGS{10022652,
  author={Al-Fetyani, Mohammad and Al-Barham, Muhammad and Abandah, Gheith and Alsharkawi, Adham and Dawas, Maha},
  booktitle={2022 IEEE Spoken Language Technology Workshop (SLT)},
  title={MASC: Massive Arabic Speech Corpus},
  year={2023},
  pages={1006-1013},
  doi={10.1109/SLT54892.2023.10022652}
}
```

## Source
- https://huggingface.co/datasets/MohamedRashad/MASC-Arabic
