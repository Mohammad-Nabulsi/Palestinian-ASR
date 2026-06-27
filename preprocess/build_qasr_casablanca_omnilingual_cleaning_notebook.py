from __future__ import annotations

import json
from pathlib import Path


BASE_NOTEBOOK = Path("preprocess/fast_asr_data_cleaning_text_only_arrow_parquet.ipynb")
OUTPUT_NOTEBOOK = Path(
    "preprocess/fast_asr_data_cleaning_text_only_arrow_parquet_qasr_casablanca_omni.ipynb"
)


def clear_outputs(nb: dict) -> None:
    for cell in nb.get("cells", []):
        if cell.get("cell_type") == "code":
            cell["execution_count"] = None
            cell["outputs"] = []


def set_markdown(cell: dict, text: str) -> None:
    cell["source"] = [line for line in text.splitlines(keepends=True)]


def set_code(cell: dict, text: str) -> None:
    cell["source"] = [line for line in text.splitlines(keepends=True)]
    cell["execution_count"] = None
    cell["outputs"] = []


def main() -> None:
    nb = json.loads(BASE_NOTEBOOK.read_text(encoding="utf-8"))
    clear_outputs(nb)

    set_markdown(
        nb["cells"][0],
        """# Fast ASR Data Cleaning Pass (QASR + Casablanca + Omnilingual)

This notebook performs the **same fast first-pass cleaning** as the base notebook, but it only targets these datasets:

- `processed_qasr_segments`
- `casablanca_jordanian`
- `casablanca_palestinian`
- `omnilingual_apc`

It keeps the same cleaning behavior:

- Drop transcripts containing English letters.
- Drop transcripts containing numbers.
- Report transcripts containing bracket/angle tokens like `[laugh]` or `<noise>`, but **keep them**.
- Drop rows with `duration < 0.5s` using the existing duration metadata.
- Create `manual_normalized_transcript`.
- Write dropped rows into folders named by reason.
- Write per-dataset and global reports.

It does **not**:

- Decode audio.
- Check RMS/silence.
- Check corrupt audio.
- Resample audio.
- Convert mono/stereo.
- Perform PCM or loudness normalization.

Those audio safeguards should happen defensively inside the training loader/collator.
""",
    )

    set_markdown(
        nb["cells"][3],
        """## 1. Configuration

Change only these paths if needed.

For your current VM layout, the defaults assume:

```text
/home/MohammadNabulsi/whisper/data
```

as the shared data root, and then select only:

- `processed_qasr_segments/train/*.arrow`
- `casablanca_jordanian/*.parquet`
- `casablanca_palestinian/*.parquet`
- `omnilingual_apc/data-*.arrow`

This intentionally skips QASR index JSONL files and Omnilingual cache Arrow files.
""",
    )

    set_code(
        nb["cells"][4],
        """# =========================
# Main paths
# =========================

INPUT_ROOT = Path("/home/MohammadNabulsi/whisper/data")
OUTPUT_ROOT = Path("/home/MohammadNabulsi/whisper/data_cleaned_text_qasr_casablanca_omni_v1")

# Target only the requested datasets.
# This keeps the cleaning logic identical while avoiding obvious
# non-training artifacts like QASR index JSONL files and HF cache Arrow files.
DISCOVERY_GLOBS = [
    "processed_qasr_segments/train/*.arrow",
    "casablanca_jordanian/*.parquet",
    "casablanca_palestinian/*.parquet",
    "omnilingual_apc/data-*.arrow",
]

# =========================
# Cleaning behavior
# =========================

MIN_DURATION_SEC = 0.5
BATCH_SIZE = 20_000
LOG_EVERY_ROWS = 50_000
COMPRESSION = "zstd"

# If True, output folder is deleted before running.
OVERWRITE_OUTPUT = True

# If set to an integer, only process the first N discovered files.
# Use this for a tiny test first, e.g. DRY_RUN_MAX_FILES = 2
DRY_RUN_MAX_FILES = None

# If duration column is missing, we cannot apply the <0.5s rule without decoding.
# Recommended for fast pass: keep rows with missing duration but report them.
DROP_MISSING_DURATION = False

TEXT_COLUMN_CANDIDATES = [
    "transcript",
    "transcription",
    "text",
    "raw_text",
    "sentence",
    "normalized_text",
]

DURATION_COLUMN_CANDIDATES = [
    "duration",
    "duration_sec",
    "duration_seconds",
    "audio_duration",
    "length_sec",
]

DATASET_COLUMN_CANDIDATES = [
    "dataset",
    "source",
    "corpus",
    "config",
    "dialect",
    "language",
]

print("INPUT_ROOT:", INPUT_ROOT)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("DISCOVERY_GLOBS:")
for pat in DISCOVERY_GLOBS:
    print(" -", pat)
""",
    )

    set_markdown(
        nb["cells"][9],
        """## 4. Discover input files

This notebook only scans the requested dataset shards via explicit glob patterns, so it excludes:

- `processed_qasr_segments/index/*.jsonl`
- `omnilingual_apc/cache-*.arrow`

Everything else about the cleaner stays the same.
""",
    )

    set_code(
        nb["cells"][10],
        """pattern_matches = {}
input_files = []

for pattern in DISCOVERY_GLOBS:
    matches = sorted(INPUT_ROOT.glob(pattern))
    pattern_matches[pattern] = matches
    input_files.extend(matches)

# Preserve discovery order while removing accidental duplicates.
input_files = list(dict.fromkeys(input_files))

if DRY_RUN_MAX_FILES is not None:
    input_files = input_files[:DRY_RUN_MAX_FILES]

parquet_files = [p for p in input_files if p.suffix == ".parquet"]
arrow_files = [p for p in input_files if p.suffix == ".arrow"]
jsonl_files = [p for p in input_files if p.suffix == ".jsonl"]

print("Discovery summary:")
for pattern, matches in pattern_matches.items():
    print(f"- {pattern}: {len(matches):,} file(s)")

print(f"Found parquet files: {len(parquet_files):,}")
print(f"Found arrow files  : {len(arrow_files):,}")
print(f"Found jsonl files  : {len(jsonl_files):,}")
print(f"Will process files : {len(input_files):,}")

for p in input_files[:10]:
    print("-", p)

if not input_files:
    raise FileNotFoundError(
        "No .parquet, .arrow, or .jsonl files matched DISCOVERY_GLOBS under "
        f"{INPUT_ROOT}"
    )
""",
    )

    set_markdown(
        nb["cells"][13],
        """## 6. Run cleaning

This cell performs the actual cleaning.

For a tiny test first, set:

```python
DRY_RUN_MAX_FILES = 2
OUTPUT_ROOT = Path("/home/MohammadNabulsi/whisper/data_cleaned_text_qasr_casablanca_omni_test")
```

Then rerun from the config cell.
""",
    )

    cell14 = "".join(nb["cells"][14]["source"])
    cell14 = cell14.replace(
        '"input_root": str(INPUT_ROOT),\n',
        '"input_root": str(INPUT_ROOT),\n'
        '    "discovery_globs": DISCOVERY_GLOBS,\n',
    )
    set_code(nb["cells"][14], cell14)

    OUTPUT_NOTEBOOK.write_text(
        json.dumps(nb, ensure_ascii=False, indent=1) + "\n",
        encoding="utf-8",
    )
    print(f"Wrote {OUTPUT_NOTEBOOK}")


if __name__ == "__main__":
    main()
