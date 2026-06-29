#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

DOC_PATH = Path("/home/MohammadNabulsi/whisper/DATA_CURATION.md")

MARKER = "## Related Utility Scripts"

SECTION = """
## Omnilingual Recleaning and Recovery

After the first merged audit, the Omnilingual APC subset was re-cleaned in two explicit Python steps.

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

Saved recovery summary:

- `Input dropped-English rows from v2: 111`
- `Recovered rows after removing full token spans: 10`
- `Still containing English after span removal: 101`

Key Omnilingual output locations:

- `data_cleaned_text_omnilingual_v2/clean/`
- `data_cleaned_text_omnilingual_v2/dropped/contains_english/`
- `data_cleaned_text_omnilingual_v2/reports/summary.txt`
- `data_cleaned_text_omnilingual_v3_recovered_from_v2/recovered_clean/`
- `data_cleaned_text_omnilingual_v3_recovered_from_v2/still_contains_english/`
- `data_cleaned_text_omnilingual_v3_recovered_from_v2/reports/summary.txt`

## Replacing Omnilingual Inside the Merged Dataset

To swap the original merged Omnilingual shards with the newer Omnilingual `v2 + recovered` result, use:

- [scripts/replace_merged_omnilingual_with_recovered.py](/home/MohammadNabulsi/whisper/scripts/replace_merged_omnilingual_with_recovered.py)

What this script does:

1. Reads:
   - `data_cleaned_text_omnilingual_v2/clean/`
   - `data_cleaned_text_omnilingual_v3_recovered_from_v2/recovered_clean/`
   - `data_cleaned_text_omnilingual_v3_recovered_from_v2/still_contains_english/`
2. Backs up the current Omnilingual shards from:
   - `data_cleaned_text_merged_v1/clean/`
   - `data_cleaned_text_merged_v1/dropped/contains_english/`
3. Replaces the merged Omnilingual clean shards with:
   - all `v2` Omnilingual clean rows
   - plus recovered rows from the token-span pass
4. Replaces the merged Omnilingual dropped-English shards with the remaining still-English rows.
5. Refuses to proceed if the materialized `still_contains_english/` rows do not match the saved recovery summary count.
6. Writes a manifest to:
   - `data_cleaned_text_merged_v1/reports/generated/omnilingual_replacement_manifest.json`

This replacement step is the final step that makes `data_cleaned_text_merged_v1/` reflect the newer Omnilingual filtering rather than the original Omnilingual rows from `data_cleaned_text_qasr_casablanca_omni_v1/`.
""".strip()


def main() -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    if "## Omnilingual Recleaning and Recovery" in text:
        start = text.index("## Omnilingual Recleaning and Recovery")
        end = text.index(MARKER, start)
        updated = text[:start] + SECTION + "\n\n" + text[end:]
    else:
        if MARKER not in text:
            raise SystemExit(f"Could not find insertion marker: {MARKER}")
        updated = text.replace(MARKER, SECTION + "\n\n" + MARKER, 1)
    DOC_PATH.write_text(updated, encoding="utf-8")
    print(f"Updated {DOC_PATH}")


if __name__ == "__main__":
    main()
