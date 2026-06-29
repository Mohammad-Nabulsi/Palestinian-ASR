#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("/home/MohammadNabulsi/whisper/DATA_CURATION.md")
MARKER = "## Related Utility Scripts"

SECTION = """
## Omnilingual Recleaning and Recovery

After the first merged audit, the Omnilingual APC subset was re-cleaned in explicit Python steps.

### Step 1: Omnilingual `v2` reclean

- Script:
  - [scripts/reclean_omnilingual_v2.py](/home/MohammadNabulsi/whisper/scripts/reclean_omnilingual_v2.py)
- Input root:
  - `.intermediate_data/omnilingual_selected/apc_north_levantine_all_splits/`
- Output root:
  - `data_cleaned_text_omnilingual_v2/`

What `v2` does before the English check:

1. Removes placeholder terms such as:
   - `hesitation`
   - `noise`
   - `unintelligible`
   - `unintelligable`
   - `unitlegable`
2. Strips lone bracket markers:
   - `<`
   - `>`
   - `[`
   - `]`
3. Preserves the original transcript in:
   - `raw_text`
4. Saves the pre-check text in:
   - `precheck_text_v2`
5. Saves punctuation-removed Arabic normalization in:
   - `manual_normalized_transcript`

Saved `v2` report summary:

- `Total rows: 517`
- `Kept rows: 406`
- `Dropped contains_english: 111`

### Step 2: Omnilingual token-span recovery

- Script:
  - [scripts/recover_omnilingual_token_span_rows_v3.py](/home/MohammadNabulsi/whisper/scripts/recover_omnilingual_token_span_rows_v3.py)
- Input root:
  - `data_cleaned_text_omnilingual_v2/dropped/contains_english/`
- Output root:
  - `data_cleaned_text_omnilingual_v3_recovered_from_v2/`

What this recovery step does:

1. Removes full token spans before re-checking English:
   - `(...)`
   - `[...]`
   - `<...>`
2. Recomputes normalized text on the span-stripped version.
3. Returns rows that no longer contain English into:
   - `recovered_clean/`
4. Leaves still-English rows in:
   - `still_contains_english/`

Saved recovery counts:

- `Input dropped-English rows from v2: 111`
- `Recovered rows after removing full token spans: 10`
- `Saved still_contains_english rows currently materialized on disk: 50`

Key Omnilingual output locations:

- `data_cleaned_text_omnilingual_v2/clean/`
- `data_cleaned_text_omnilingual_v2/dropped/contains_english/`
- `data_cleaned_text_omnilingual_v2/reports/summary.txt`
- `data_cleaned_text_omnilingual_v3_recovered_from_v2/recovered_clean/`
- `data_cleaned_text_omnilingual_v3_recovered_from_v2/still_contains_english/`
- `data_cleaned_text_omnilingual_v3_recovered_from_v2/reports/summary.txt`

## Working `data/` Copy With Final Kept Omnilingual Rows

To create a working copy of the merged cleaned dataset and replace its Omnilingual clean subset with the final kept Omnilingual rows, use:

- [scripts/create_data_with_final_omnilingual.py](/home/MohammadNabulsi/whisper/scripts/create_data_with_final_omnilingual.py)

What this script does:

1. Creates a new top-level `data/` directory as a hard-linked copy of:
   - `data_cleaned_text_merged_v1/`
2. Removes the original Omnilingual clean shards from:
   - `data/clean/`
3. Replaces them with the final kept Omnilingual rows from:
   - `data_cleaned_text_omnilingual_v2/clean/`
   - `data_cleaned_text_omnilingual_v3_recovered_from_v2/recovered_clean/`
4. This yields `416` kept Omnilingual rows inside `data/clean/`.
5. Writes a manifest to:
   - `data/reports/generated/omnilingual_final_kept_manifest.json`
6. Deletes the top-level intermediate directory:
   - `intermediate/`

Important note:

- This `data/` copy replaces the Omnilingual clean data only.
- The saved `still_contains_english/` materialization currently contains `50` rows on disk.
- Because of that mismatch, this step does not replace Omnilingual dropped-English shards in `data/`.
""".strip()


def main() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    if "## Omnilingual Recleaning and Recovery" in text:
        start = text.index("## Omnilingual Recleaning and Recovery")
        end = text.index(MARKER, start)
        updated = text[:start] + SECTION + "\n\n" + text[end:]
    else:
        updated = text.replace(MARKER, SECTION + "\n\n" + MARKER, 1)
    DOC_PATH.write_text(updated, encoding="utf-8")
    print(f"Updated {DOC_PATH}")


if __name__ == "__main__":
    main()
