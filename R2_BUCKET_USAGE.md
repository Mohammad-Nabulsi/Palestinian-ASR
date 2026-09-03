# R2 Bucket Usage (S3-compatible)

Backup/transfer storage for this project lives in a Cloudflare R2 bucket, accessed
through the standard S3 API. This doc covers what's in it and how to connect.

**Credentials are never committed to this repo.** See "Credentials" below for how
to get them and where to put them locally.

## Bucket layout

Bucket name: `backup`

```
transfer/
├── curated_corpus/                # final multi-source ASR corpus (see DATA_CURATION.md, PIPELINE.md)
│   ├── train/{casa,layla,masc,omni,qasr}/[lev|non_lev]/*.parquet.zst
│   ├── val/...
│   ├── test/...
│   └── reports/                   # duration_stats.json, summary.json, per-source progress/checkpoints
├── data/
│   ├── data_lev_custom_split_v1/  # standalone qasr+masc_c experiment (train/val/test/mix200h/flat12h.parquet)
│   │                              # NOTE: its train/val/test are NOT speaker-disjoint --
│   │                              # see SPEAKER_DISJOINT_SELECTION.md
│   └── speaker_disjoint_split_v1/ # the split *definition* that replaces it: speaker_assignments.json
│                                  # (speaker -> split), selection.csv, the full ranking, and 150
│                                  # inspection samples. Uploaded 2026-09-03; see its README.md
└── adapters/                      # LoRA / Whisper fine-tuning checkpoints (training runs, not data)
    ├── FINAL_200h/
    └── whisper/
```

`curated_corpus` shards are zstd-wrapped parquet (`*.parquet.zst`) — decompress the
whole file before reading (`unzstd`), since it's a single compressed stream, not
per-page parquet compression. `data_lev_custom_split_v1` files are plain parquet
and support column-projection / range reads directly.

## Credentials

You need three values, none of which belong in this repo or any commit:

| Value | Used for |
|---|---|
| Access Key ID | S3 auth |
| Secret Access Key | S3 auth |
| Account/jurisdiction endpoint | e.g. `https://<account-id>.r2.cloudflarestorage.com` |

Get the current values from wherever the team keeps shared credentials (password
manager / whoever set up the bucket) and export them locally — do not paste them
into files that get committed:

```bash
export R2_ACCESS_KEY_ID="..."
export R2_SECRET_ACCESS_KEY="..."
export R2_ENDPOINT="https://<account-id>.r2.cloudflarestorage.com"
```

If you keep them in a local `.env` file instead, make sure it's covered by
`.gitignore` before touching git at all.

## Connecting with rclone

No config file needed — pass everything via env vars on an ad hoc remote name (`R2:`):

```bash
export RCLONE_CONFIG_R2_TYPE=s3
export RCLONE_CONFIG_R2_PROVIDER=Cloudflare
export RCLONE_CONFIG_R2_ACCESS_KEY_ID="$R2_ACCESS_KEY_ID"
export RCLONE_CONFIG_R2_SECRET_ACCESS_KEY="$R2_SECRET_ACCESS_KEY"
export RCLONE_CONFIG_R2_ENDPOINT="$R2_ENDPOINT"

rclone lsd R2:backup                              # list top-level dirs
rclone lsl R2:backup/transfer/curated_corpus/reports
rclone copy R2:backup/transfer/curated_corpus/reports/summary.json .
```

## Connecting with Python (pyarrow)

Preferred for parquet work — supports column projection and range reads directly
against S3, so you don't have to download whole files to inspect schema/columns:

```python
import os
import pyarrow.fs as fs
import pyarrow.parquet as pq

s3 = fs.S3FileSystem(
    access_key=os.environ["R2_ACCESS_KEY_ID"],
    secret_key=os.environ["R2_SECRET_ACCESS_KEY"],
    endpoint_override=os.environ["R2_ENDPOINT"],
    scheme="https",
)

with s3.open_input_file("backup/transfer/data/data_lev_custom_split_v1/val.parquet") as fh:
    pf = pq.ParquetFile(fh)
    table = pf.read(columns=["uid", "source", "text_lev", "audio_lev", "duration"])
```

For `curated_corpus`'s `.parquet.zst` shards, download + decompress first (they're
not column-readable in place):

```bash
rclone copy R2:backup/transfer/curated_corpus/train/qasr/lev/data-00000.parquet.zst .
unzstd data-00000.parquet.zst
```

## Connecting with boto3

```python
import boto3

s3 = boto3.client(
    "s3",
    aws_access_key_id=os.environ["R2_ACCESS_KEY_ID"],
    aws_secret_access_key=os.environ["R2_SECRET_ACCESS_KEY"],
    endpoint_url=os.environ["R2_ENDPOINT"],
)
s3.download_file("backup", "transfer/curated_corpus/reports/summary.json", "summary.json")
```

## Note on the separate Cloudflare API token

There's also a Cloudflare account-level API token (distinct from the S3 access
key/secret above) used for managing the R2 bucket itself via Cloudflare's API
(e.g. bucket settings, lifecycle rules) rather than for reading/writing objects.
Same rule applies: keep it out of the repo, export as an env var
(e.g. `CLOUDFLARE_API_TOKEN`) when a script actually needs it.
