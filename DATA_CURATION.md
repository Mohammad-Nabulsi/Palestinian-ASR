# Data Curation Map

This file is the single place that explains how the speech datasets in this repo were staged, processed, and cleaned.

It answers four questions:

1. Which raw datasets were involved?
2. Which script or notebook created each intermediate directory?
3. What is intermediate vs final?
4. How were `data_cleaned_text_v1` and `data_cleaned_text_qasr_casablanca_omni_v1` created?

## High-Level Pipeline

The data flow in this repo is best understood as:

`raw source datasets` -> `source-specific preprocessing` -> `staging under data/` -> `text-only cleaning outputs`

## Raw Source Datasets

These are the source datasets that appear in the repo and scripts:

- `QASR/`
- `casablanca/`
- `omnilingual_selected/`
- `Layla/`
- `MASC-Arabic2/` or another local MASC source root used with `scripts/filter_masc_c_only.py`

The older download/prep notes are still in [data.md](/home/MohammadNabulsi/whisper/data.md) history, but this file is the cleaned-up operational summary.

## Directory Meaning

### `processed_qasr_segments/`

What it is:
- A QASR-only segmented dataset written as Arrow shards plus indexes.

What created it:
- [preprocess/qasr_segment_to_arrow.py](/home/MohammadNabulsi/whisper/preprocess/qasr_segment_to_arrow.py)

What that script does:
- Reads QASR WAV files from `QASR/.../wav`
- Matches them with QASR XML timing/transcript files
- Cuts audio into utterance segments
- Normalizes transcript text
- Writes Arrow shards under `processed_qasr_segments/train/`
- Writes index/log files under `processed_qasr_segments/index/` and `processed_qasr_segments/logs/`

Why it matters:
- This is an intermediate dataset used later by the targeted cleaning notebook.

### `data/masc_c_only/`

What it is:
- A filtered MASC dataset containing only rows where `type == "c"`.

What created it:
- [scripts/filter_masc_c_only.py](/home/MohammadNabulsi/whisper/scripts/filter_masc_c_only.py)

What that script does:
- Reads MASC parquet shards from a source dataset root
- Keeps only rows where column `type` equals `c`
- Writes filtered parquet shards to `data/masc_c_only/data/`

Why it matters:
- This is an intermediate dataset.
- It was later picked up by the broad cleaning notebook and is, in practice, what produced `data_cleaned_text_v1`.

### `data/`

What it is:
- A staging directory that groups several datasets under one root so downstream code can scan them consistently.

Verified source of the current `data/` layout:
- [scripts/stage_raw_datasets.py](/home/MohammadNabulsi/whisper/scripts/stage_raw_datasets.py)

Why this is verified:
- The current `data/` root contains exactly the top-level entries that `scripts/stage_raw_datasets.py` creates:
  - symlink `data/processed_qasr_segments` -> `../processed_qasr_segments`
  - symlink `data/omnilingual_apc` -> `../omnilingual_selected/apc_north_levantine_all_splits`
  - symlink `data/casablanca_palestinian` -> `../casablanca/levant/Palestine`
  - symlink `data/casablanca_jordanian` -> `../casablanca/levant/Jordan`
  - copied/staged `data/layla/`
  - pre-existing `data/masc_c_only/`
- The current `data/layla/` contents also match `scripts/stage_raw_datasets.py` behavior:
  - WAV files copied in place
  - TXT transcript files copied in place
  - DOCX transcript files converted to TXT in the same relative folder structure
- The current `data/` root does not contain the directories that [preprocess/unify.py](/home/MohammadNabulsi/whisper/preprocess/unify.py) would create, such as:
  - `audio/raw/`
  - `annotations/raw/`
  - `metadata/raw/`
  - `manifests/`
  - `stats/`
  - `logs/`

What `scripts/stage_raw_datasets.py` does:
- Verifies that `data/masc_c_only/` already exists
- Creates symlinks for QASR, Omnilingual APC, and Casablanca subsets
- Copies Layla WAV/TXT files into `data/layla/`
- Converts Layla `.docx` annotations to `.txt` when needed

What this means:
- The current `data/` directory was created as a staging step for downstream processing.
- It was not produced by the heavier `preprocess/unify.py` pipeline.
- `data/` is an intermediate staging root, not the final cleaned training output.

How to recreate the current `data/` layout:
1. Create the filtered MASC subset first with [scripts/filter_masc_c_only.py](/home/MohammadNabulsi/whisper/scripts/filter_masc_c_only.py) so `data/masc_c_only/` exists.
2. Make sure these source roots exist in the repo root:
   - `processed_qasr_segments/`
   - `omnilingual_selected/apc_north_levantine_all_splits/`
   - `casablanca/levant/Palestine/`
   - `casablanca/levant/Jordan/`
   - `Layla/Layla Witheeb Jordanian Arabic Acoustic Dataset/`
3. Run [scripts/stage_raw_datasets.py](/home/MohammadNabulsi/whisper/scripts/stage_raw_datasets.py) from the repo root.
4. That script recreates the current staging layout by:
   - symlinking QASR, Omnilingual APC, and Casablanca into `data/`
   - copying Layla WAV/TXT files into `data/layla/`
   - converting Layla `.docx` transcripts into `.txt` when no TXT already exists

Command:
```bash
cd /home/MohammadNabulsi/whisper
python3 scripts/stage_raw_datasets.py --output data
```

### Layla Transcript Cleaning Prompt

After Layla files were staged under `data/layla/`, a later text-normalization step was used for Layla transcript cleaning. The prompt used for that cleaning/normalization step was:

```text
You are cleaning Arabic dialect ASR transcripts. Your task is to find and normalize **phonetic transcription artifacts only**, not to convert the dialect into MSA. Input: I will provide transcript text, usually grouped by source file. Output a JSON object with exactly these arrays: 1. records Each item must contain: * source * original * normalized 2. word_conversions Each item must contain: * original * normalized * reason * confidence: one of high, medium, low Rules: * Keep the dialect as dialect. * Do **not** translate to MSA. * Do **not** modernize dialect words. * Do **not** normalize Arabic letters globally. * Do **not** convert أ/إ/آ to ا. * Do **not** convert ى to ي. * Do **not** convert ه to ة. * Do **not** remove hamza unless the specific word is clearly a phonetic artifact and the normalized dialect spelling requires it. * Do **not** remove dialect vocabulary such as إلها, راح, إجا, طخ, ستي, تيتا, خشمك, منخارك, بدها, بده, لقت, حكالها. * Do **not** normalize words just because they are not MSA. Normalize only cases where the written word is clearly a pronunciation-spelling artifact, typo-like phonetic spelling, or inconsistent ASR/transcriber representation of the same dialect word. Look for all types of phonetic artifacts, including but not limited to: * ك written as تش, such as عليتش → عليك, صوتش → صوتك, عيونتش → عيونك. * لك written as لتش, such as أسرعلتش → أسرعلك. * كيف written as تشيف. * qaf/hamza phonetic spellings when they are not intended dialect orthography, such as ئال/أل when the intended written dialect word is قال, only if context proves it. * ذ written as ز, such as أنقز → أنقذ, only when the intended word is obvious. * ذ written as د, such as أدنيك/دنيكي when the intended word is أذنيك. * dropped initial letters caused by pronunciation or ASR artifacts, such as ذنيك → أذنيك, only when context is clear. * malformed possessive suffixes, such as أذنيكي → أذنيك, جدتكي → جدتك. * fused or broken forms caused by pronunciation transcription, such as دارستها → دار ستها, only if clearly a spacing artifact. * repeated/partial ASR fragments, stutters, or cut words, but mark these separately as artifact_fragment in the reason. Use one consistent normalization every time: * If the same original word appears multiple times, normalize it the same way. * If multiple original spellings map to the same normalized word, include each original spelling in word_conversions. Important distinction: * dialect word = keep. * phonetic artifact = normalize. * MSA conversion = forbidden. * letter-wide normalization = forbidden. * uncertain case = include in word_conversions with confidence: "low" and explain why, but do not silently change it in records unless the context strongly supports it. For every changed transcript, preserve punctuation and word order as much as possible. Only change the specific artifact words. Also provide a short summary after the JSON: * number of sources processed * number of unique word conversions * examples of high-confidence conversions * examples of uncertain cases * list of dialect words intentionally left unchanged
```

What this prompt was for:
- Cleaning Layla transcript text after staging, especially phonetic spelling artifacts in dialect writing.
- Preserving Jordanian/Levantine dialect wording rather than converting it into MSA.
- Producing normalized transcript records plus a conversion audit trail.

### Layla In-Place Shard Cleaning

A Layla-specific notebook now exists at:
- [preprocess/fast_asr_data_cleaning_text_only_arrow_parquet_layla.ipynb](/home/MohammadNabulsi/whisper/preprocess/fast_asr_data_cleaning_text_only_arrow_parquet_layla.ipynb)

What it targets:
- staged Layla parquet shards directly under `data/`:
  - `data/layla__data-00000-of-00004.parquet`
  - `data/layla__data-00001-of-00004.parquet`
  - `data/layla__data-00002-of-00004.parquet`
  - `data/layla__data-00003-of-00004.parquet`

How it was used:
- The notebook was configured to discover only `layla__*.parquet` under `data/`.
- It wrote temporary clean shards and reports to an intermediate root `L/`.
- After verification, the cleaned Layla parquet shards were copied back in place over the original `data/layla__*.parquet` files.
- The intermediate `L/` directory was then deleted.

What changed in the staged Layla shards:
- The staged Layla parquet files in `data/` now contain the original rows plus cleaning metadata columns such as:
  - `manual_normalized_transcript`
  - `flag_contains_bracket_token`
  - `flag_contains_english`
  - `flag_contains_number`
  - `flag_audio_too_short`
  - `flag_missing_duration`
- This was an in-place staged-data refresh, not a separate long-lived cleaned-output directory like `data_cleaned_text_v1/`.

## Cleaning Outputs

Both cleaned directories are text-cleaning outputs, not raw or audio-standardization outputs.

Shared cleaning behavior in both notebooks:

- Drop transcripts containing English letters
- Drop transcripts containing numbers
- Report bracket/angle tokens like `[laugh]` or `<noise>` but keep them unless another drop rule fires
- Drop rows where duration is below `0.5` seconds
- Create `manual_normalized_transcript`
- Do not decode audio
- Do not run silence/RMS checks
- Do not resample audio
- Do not do loudness or PCM normalization

That means these are fast text-first cleaning passes.

### `data_cleaned_text_v1/`

What it is:
- The output of the broad cleaning notebook:
  - [preprocess/fast_asr_data_cleaning_text_only_arrow_parquet.ipynb](/home/MohammadNabulsi/whisper/preprocess/fast_asr_data_cleaning_text_only_arrow_parquet.ipynb)

Configured input/output:
- input root: `/home/MohammadNabulsi/whisper/data`
- output root: `/home/MohammadNabulsi/whisper/data_cleaned_text_v1`

How discovery worked:
- The notebook recursively scanned `data/` for:
  - `*.parquet`
  - `*.arrow`
  - `*.jsonl`

What it produced:
- `clean/`: cleaned output shards
- `dropped/`: rows dropped by reason
- `reports/`: config, manifest, per-dataset report, log, cleaning summary

What it actually processed in practice:
- According to `data_cleaned_text_v1/reports/per_dataset_report.csv`, this run effectively processed only `masc_c_only`.

Observed result:
- total rows: `369,243`
- kept: `369,082`
- dropped as `audio_too_short`: `161`
- no English-letter drops
- no numeric drops

Interpretation:
- Even though the notebook was generic and scanned all of `data/`, the actual resulting cleaned dataset is effectively a cleaned `masc_c_only` dataset.

### `data_cleaned_text_qasr_casablanca_omni_v1/`

What it is:
- The output of the targeted cleaning notebook:
  - [preprocess/fast_asr_data_cleaning_text_only_arrow_parquet_qasr_casablanca_omni.ipynb](/home/MohammadNabulsi/whisper/preprocess/fast_asr_data_cleaning_text_only_arrow_parquet_qasr_casablanca_omni.ipynb)

Where that notebook came from:
- It was generated from the base notebook by:
  - [preprocess/build_qasr_casablanca_omnilingual_cleaning_notebook.py](/home/MohammadNabulsi/whisper/preprocess/build_qasr_casablanca_omnilingual_cleaning_notebook.py)

Configured input/output:
- input root: `/home/MohammadNabulsi/whisper/data`
- output root: `/home/MohammadNabulsi/whisper/data_cleaned_text_qasr_casablanca_omni_v1`

Targeted discovery globs:
- `processed_qasr_segments/train/*.arrow`
- `casablanca_jordanian/*.parquet`
- `casablanca_palestinian/*.parquet`
- `omnilingual_apc/data-*.arrow`

Important exclusions:
- QASR index files under `processed_qasr_segments/index/*.jsonl`
- Omnilingual cache Arrow files such as `cache-*.arrow`

What it produced:
- `clean/`: cleaned output shards
- `dropped/contains_english/`
- `dropped/contains_number/`
- `dropped/audio_too_short/`
- `reports/`: config, manifest, per-dataset report, log, cleaning summary

What it actually processed in practice:
- `qasr`
- `casablanca_jordanian`
- `casablanca_palestinian`
- `apc_Arab` from the Omnilingual APC data

Observed result from `reports/cleaning_report.json`:
- total rows: `1,197,781`
- kept: `1,128,111`
- dropped for English letters: `5,607`
- dropped for numbers: `58,053`
- dropped for short duration: `6,010`

Per-dataset highlights:
- `qasr`: main volume of the run, `1,194,234` rows total
- `casablanca_jordanian`: `1,696` rows total
- `casablanca_palestinian`: `1,334` rows total
- `apc_Arab`: `517` rows total, with heavy English/bracket-token content

Interpretation:
- This is the real cleaned multi-dataset ASR text set among the two outputs.

## Step-by-Step Provenance

### Path to `data_cleaned_text_v1`

This appears to have been:

1. Prepare a filtered MASC subset with [scripts/filter_masc_c_only.py](/home/MohammadNabulsi/whisper/scripts/filter_masc_c_only.py)
2. Stage datasets under `data/` with [scripts/stage_raw_datasets.py](/home/MohammadNabulsi/whisper/scripts/stage_raw_datasets.py)
3. Run [preprocess/fast_asr_data_cleaning_text_only_arrow_parquet.ipynb](/home/MohammadNabulsi/whisper/preprocess/fast_asr_data_cleaning_text_only_arrow_parquet.ipynb)
4. In practice, that run produced a cleaned `masc_c_only` output at `data_cleaned_text_v1/`

### Path to `data_cleaned_text_qasr_casablanca_omni_v1`

This appears to have been:

1. Segment QASR into Arrow with [preprocess/qasr_segment_to_arrow.py](/home/MohammadNabulsi/whisper/preprocess/qasr_segment_to_arrow.py), producing `processed_qasr_segments/`
2. Stage datasets under `data/` with [scripts/stage_raw_datasets.py](/home/MohammadNabulsi/whisper/scripts/stage_raw_datasets.py)
3. Generate the narrowed notebook with [preprocess/build_qasr_casablanca_omnilingual_cleaning_notebook.py](/home/MohammadNabulsi/whisper/preprocess/build_qasr_casablanca_omnilingual_cleaning_notebook.py)
4. Run [preprocess/fast_asr_data_cleaning_text_only_arrow_parquet_qasr_casablanca_omni.ipynb](/home/MohammadNabulsi/whisper/preprocess/fast_asr_data_cleaning_text_only_arrow_parquet_qasr_casablanca_omni.ipynb)
5. This produced `data_cleaned_text_qasr_casablanca_omni_v1/`

## Intermediate vs Final

Treat these as intermediate:

- `processed_qasr_segments/`
- `data/`
- `data/masc_c_only/`

Treat these as final outputs of the fast text-cleaning pass:

- `data_cleaned_text_v1/`
- `data_cleaned_text_qasr_casablanca_omni_v1/`
- `data_cleaned_text_merged_v1/`

But note:
- `data_cleaned_text_v1/` is final only for the broad notebook run, and that run effectively cleaned only `masc_c_only`
- `data_cleaned_text_qasr_casablanca_omni_v1/` is the clearer multi-source cleaned dataset output
- `data_cleaned_text_merged_v1/` is the post-merge consolidated root that combines both cleaned outputs into one `clean/`, one `dropped/`, and one `reports/` tree

## Folder Cheat Sheet

- `QASR/`: raw QASR source
- `casablanca/`: raw/restructured Casablanca source
- `omnilingual_selected/`: selected Omnilingual source
- `Layla/`: raw Layla source
- `processed_qasr_segments/`: QASR segmented Arrow intermediate
- `data/`: staging root combining selected datasets
- `data_cleaned_text_v1/`: broad fast text-cleaning output, effectively for `masc_c_only`
- `data_cleaned_text_qasr_casablanca_omni_v1/`: targeted fast text-cleaning output for QASR + Casablanca + Omnilingual APC
- `data_cleaned_text_merged_v1/`: merged cleaned-data root combining both cleaned outputs
- `intermediate/merged_cleaned_sources/`: archived empty source roots after the merge step

## Merge and Omnilingual English Audit

After both cleaning runs existed, they were consolidated with:

- [scripts/merge_cleaned_outputs_and_report.py](/home/MohammadNabulsi/whisper/scripts/merge_cleaned_outputs_and_report.py)

What this script does:

1. Moves the contents of:
   - `data_cleaned_text_v1/`
   - `data_cleaned_text_qasr_casablanca_omni_v1/`
2. Merges them into:
   - `data_cleaned_text_merged_v1/clean/`
   - `data_cleaned_text_merged_v1/dropped/`
   - `data_cleaned_text_merged_v1/reports/source_reports/<original_source_name>/`
3. Archives the now-empty original source roots under:
   - `intermediate/merged_cleaned_sources/`
4. Scans dropped Omnilingual APC shards from:
   - `data_cleaned_text_merged_v1/dropped/contains_english/omnilingual_apc*.parquet`
5. Writes detailed reports of exact English tokens found to:
   - `data_cleaned_text_merged_v1/reports/generated/omnilingual_contains_english_report.json`
   - `data_cleaned_text_merged_v1/reports/generated/omnilingual_contains_english_report.md`
   - `data_cleaned_text_merged_v1/reports/generated/merge_manifest.json`

Important note on the Omnilingual English report:

- English tokens are not only from transcript text.
- They can also come from fields like `prompt`, `prompt_id`, `source_file`, and other metadata carried into dropped shards.
- The generated report records token counts by field and row-level examples.

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

## Layla Prompt Merge and Sharding

For the Layla normalization pass, the prompted outputs were manually uploaded and merged into these four JSON files:

- `normalized_output_appended.json`
- `normalized_layla_batch_130_appended_131.json`
- `normalized_pasted_132_133_appended.json`
- `normalized_pasted_text_134_135_136_appended.json`

These JSON files contain the normalized text under `normalized`, and that text was written into the training column named `transcription`.

The Layla sharding step then:

1. Read the four merged JSON files.
2. Matched each JSON `source` entry to the corresponding Layla audio file by removing `_Arabic_transcription` from the transcript filename and resolving the paired `.WAV` or `.wav`.
3. Built Parquet training shards with:
   - `audio`
   - `seg_id`
   - `transcription`
   - `duration`
   - `source_file`
4. Wrote the Layla shards directly into `data/`.
5. Moved the original `Layla/` source directory into:
   - `.intermediate_data/Layla/`

Layla shards written into `data/`:

- `layla__data-00000-of-00004.parquet`
- `layla__data-00001-of-00004.parquet`
- `layla__data-00002-of-00004.parquet`
- `layla__data-00003-of-00004.parquet`

## Flattening `data/clean` Into `data/`

After preparing the working dataset, the shard files under `data/clean/` were unpacked into `data/` directly so the training shards now live at the top level of `data/`.

Moved shard files from `data/clean/` into `data/`:

- `casablanca_jordanian__test-00000-of-00001.parquet__349f3208cf__clean.parquet`
- `casablanca_jordanian__validation-00000-of-00001.parquet__4882d7d03b__clean.parquet`
- `casablanca_palestinian__test-00000-of-00002.parquet__106b14e6b8__clean.parquet`
- `casablanca_palestinian__test-00001-of-00002.parquet__6275edc7fa__clean.parquet`
- `casablanca_palestinian__validation-00000-of-00002.parquet__b3a00f1492__clean.parquet`
- `casablanca_palestinian__validation-00001-of-00002.parquet__a4a279ac0d__clean.parquet`
- `masc_c_only__data__test-00000-of-00009.parquet__5ab0b5b7dc__clean.parquet`
- `masc_c_only__data__test-00001-of-00009.parquet__2b38af7d8d__clean.parquet`
- `masc_c_only__data__test-00002-of-00009.parquet__efb5ed1f66__clean.parquet`
- `masc_c_only__data__test-00003-of-00009.parquet__133ba07b54__clean.parquet`
- `masc_c_only__data__test-00004-of-00009.parquet__1defcdef6f__clean.parquet`
- `masc_c_only__data__test-00005-of-00009.parquet__37f2aee764__clean.parquet`
- `...`

## Binary Levantine Split Curation

To prepare the binary Levantine-vs-non-Levantine training layout, dialect identification was applied in two stages on the cleaned `masc_c` and `qasr` shards:

1. Text dialect identification was run on the full cleaned MASC-C and QASR shards with:
   - `Runs/text_dialect_scan_marbertv2_written_clean_masc_c_qasr/row_probabilities.jsonl`
2. Audio dialect identification was then run only on the subset whose text-stage `LEV` probability was at least `0.80`, with:
   - `Runs/dialect_scan_badrex_mms300m_lev08_text_candidates_masc_c_qasr/row_probabilities.jsonl`

The binary rule used for the new shards is:

- `lev`: rows from `masc_c` or `qasr` where text `LEV >= 0.80` and audio `Levantine >= 0.80`
- `non_lev`: every other `masc_c` or `qasr` row

The script that materializes this layout is:

- [scripts/create_levant_non_levant_splits.py](/home/MohammadNabulsi/whisper/scripts/create_levant_non_levant_splits.py)

What this script does:

1. Reads the text-stage and audio-stage row probability files.
2. Uses the audio-stage rows to mark the final accepted `lev` rows for `masc_c` and `qasr`.
3. Treats all remaining `masc_c` and `qasr` rows as `non_lev`.
4. Re-splits every source into fresh `train`, `val`, and `test` partitions with ratio `0.70 / 0.15 / 0.15`.
5. Writes a new shard tree under:
   - `data_curated_levant_binary_v1/`
6. Saves a generation summary to:
   - `data_curated_levant_binary_v1/reports/summary.json`

The output directory layout is:

- `train/masc/lev/`
- `train/masc/non_lev/`
- `train/qasr/lev/`
- `train/qasr/non_lev/`
- `train/omni/`
- `train/layla/`
- `train/casa/pal/`
- `train/casa/jor/`
- `val/...` with the same leaf directories
- `test/...` with the same leaf directories

Temporary QASR note:

- The currently in-place `qasr` split inside `data_curated_levant_binary_v1/` was rebuilt from the audio dialect predictions available at rebuild time.
- Under the same binary rule, rows with confirmed text `LEV >= 0.80` and confirmed audio `Levantine >= 0.80` are written to `qasr/lev/`.
- All remaining QASR rows are currently written to `qasr/non_lev/`, including rows that were not yet audio-classified in the partial repair run.
- This is a temporary operational choice made because fully re-running QASR audio dialect identification is resource-intensive on the currently available compute setup.
- Once resources allow a full repaired QASR audio pass to completion, the QASR `lev/non_lev` split should be regenerated from the complete repaired audio-stage output.

Run command:

```bash
cd /home/MohammadNabulsi/whisper
./.venv/bin/python scripts/create_levant_non_levant_splits.py --overwrite
```

## QASR Audio Classification Repair

After the first binary Levantine split run, the QASR audio-stage results were audited and the failure mode was identified.

What went wrong in the first QASR audio-stage run:

- The QASR segment builder stores per-segment audio as raw PCM `int16` bytes in the `audio` field, alongside a separate `sampling_rate` column.
- The original audio dialect scan logic attempted to open byte-valued audio with `torchaudio.load(BytesIO(...))`, which expects an encoded audio file stream such as WAV/FLAC/OGG.
- Because of that mismatch, most QASR candidate rows failed during audio loading with `LibsndfileError` before a dialect prediction could be produced.

What was changed to fix it:

- [dialect_identifiaction/arabic_dialect_scan_badrex_mms300m.py](/home/MohammadNabulsi/whisper/dialect_identifiaction/arabic_dialect_scan_badrex_mms300m.py) was updated so byte-valued audio can be decoded in two modes:
  - encoded audio bytes when the payload looks like WAV/FLAC/OGG metadata
  - raw PCM `int16` bytes when a `sampling_rate` is present and the payload is not an encoded audio file stream
- This specifically repairs the QASR case while preserving the working MASC byte-decoding path.

The repair-and-rebuild orchestration script is:

- [scripts/repair_qasr_audio_and_rebuild_levant_binary.py](/home/MohammadNabulsi/whisper/scripts/repair_qasr_audio_and_rebuild_levant_binary.py)

What this repair script does:

1. Re-runs audio dialect classification with the PCM-aware loader and writes a repaired audio-stage output under:
   - `Runs/dialect_scan_badrex_mms300m_lev08_text_candidates_masc_c_qasr_qasrfix/`
2. Rebuilds the binary Levantine split using the same double-threshold rule:
   - text `LEV >= 0.80`
   - audio `Levantine >= 0.80`
3. Writes the rebuilt dataset under:
   - `data_curated_levant_binary_v2_qasr_audio_fix/`

Current limitation:

- A full repaired QASR audio rerun is still the intended final path, but it is currently constrained by available compute resources and runtime.
- Because of that, the working `data_curated_levant_binary_v1/` QASR replacement may temporarily rely on the repaired audio predictions completed so far, with the remainder routed to `non_lev` until the full rerun is completed.

Run command:

```bash
cd /home/MohammadNabulsi/whisper
./.venv/bin/python scripts/repair_qasr_audio_and_rebuild_levant_binary.py
```

## Related Utility Scripts

- [scripts/compute_data_duration_stats.py](/home/MohammadNabulsi/whisper/scripts/compute_data_duration_stats.py): computes duration stats for staged datasets
- [outputs/compute_data_duration_stats.py](/home/MohammadNabulsi/whisper/outputs/compute_data_duration_stats.py): copied next to `outputs/data_duration_stats.json`
- [preprocess/unify.py](/home/MohammadNabulsi/whisper/preprocess/unify.py): older/alternate raw unification pipeline, different output layout from current `data/`

