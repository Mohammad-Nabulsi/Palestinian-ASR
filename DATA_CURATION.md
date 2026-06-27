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

But note:
- `data_cleaned_text_v1/` is final only for the broad notebook run, and that run effectively cleaned only `masc_c_only`
- `data_cleaned_text_qasr_casablanca_omni_v1/` is the clearer multi-source cleaned dataset output

## Folder Cheat Sheet

- `QASR/`: raw QASR source
- `casablanca/`: raw/restructured Casablanca source
- `omnilingual_selected/`: selected Omnilingual source
- `Layla/`: raw Layla source
- `processed_qasr_segments/`: QASR segmented Arrow intermediate
- `data/`: staging root combining selected datasets
- `data_cleaned_text_v1/`: broad fast text-cleaning output, effectively for `masc_c_only`
- `data_cleaned_text_qasr_casablanca_omni_v1/`: targeted fast text-cleaning output for QASR + Casablanca + Omnilingual APC

## Related Utility Scripts

- [scripts/compute_data_duration_stats.py](/home/MohammadNabulsi/whisper/scripts/compute_data_duration_stats.py): computes duration stats for staged datasets
- [outputs/compute_data_duration_stats.py](/home/MohammadNabulsi/whisper/outputs/compute_data_duration_stats.py): copied next to `outputs/data_duration_stats.json`
- [preprocess/unify.py](/home/MohammadNabulsi/whisper/preprocess/unify.py): older/alternate raw unification pipeline, different output layout from current `data/`

