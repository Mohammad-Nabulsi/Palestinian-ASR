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

### MASC source repo, confirmed

The dataset card saved locally at `.intermediate_data/MASC-Arabic2/README.md` credits
`https://huggingface.co/datasets/pain/MASC` as the "Original Dataset Repo" in its prose, but that
is misleading: the card's own usage example calls `load_dataset("MohamedRashad/MASC-Arabic", ...)`,
and a byte-for-byte comparison against both repos' live READMEs (via the HF Hub API) confirms the
local card is an exact copy of `MohamedRashad/MASC-Arabic`'s card, not `pain/MASC`'s.

**The correct Hugging Face repo for MASC in this project is `MohamedRashad/MASC-Arabic`.**

Recorded splits from that same local dataset card (`dataset_info` block):

- `train`: 875,873 examples
- `validation`: 19,521 examples
- `test`: 18,006 examples
- `download_size`: ~184.87 GB
- `dataset_size` (decompressed): ~209.19 GB

Download notebook: [downlaod_notebooks/MASC/masc.ipynb](/home/MohammadNabulsi/whisper/downlaod_notebooks/MASC/masc.ipynb).

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
- `segmented/`: sibling to `data/` (not nested inside it), long-audio (>30s) segmentation
  artifacts — `flagged_over30s.json` (scan output) and `v1/` (segmented shards, reports,
  pre-overwrite backups of the `data/` originals, replacement manifest/checkpoint)

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

## Post-Merge Cleanup: Deleting Superseded Raw/Intermediate Directories

Once `data_cleaned_text_merged_v1/` existed and was verified (119/119 + 417/417 shards present,
`run.log` ended with `DONE rows=...` matching the report totals, no zero-byte files), several
raw/intermediate directories became redundant: their entire content was already captured, in
cleaned form, inside `data_cleaned_text_merged_v1/clean/`. They were deleted directly on network
storage (`/workspace/asr/Palestinian-ASR/...`) to free space:

- `MASC-Arabic2/` (raw, ~173GB) — only the `type=='c'` subset was ever used (via
  `filter_masc_c_only.py`), and that filtered+cleaned subset is fully captured in
  `data_cleaned_text_merged_v1/clean/masc_c_only__*`.
- `casablanca/` (raw, ~8.6GB) — fully captured in `data_cleaned_text_merged_v1/clean/casablanca_*`;
  row counts matched the historical counts in this file exactly.
- `processed_qasr_segments/` (intermediate, ~55GB) — the QASR-segments-as-Arrow intermediate from
  `preprocess/qasr_segment_to_arrow.py`; fully captured in
  `data_cleaned_text_merged_v1/clean/processed_qasr_segments__train__*`. Note: this only reflects
  the 961/3,545 QASR transcripts that had matched audio at the time — see the QASR Gap 2 discussion
  below before assuming this is the full QASR dataset.
- `data/` (staging root from `scripts/stage_raw_datasets.py`, ~80GB) — every one of its five
  entries (`casablanca_jordanian`, `casablanca_palestinian`, `masc_c_only/`, `omnilingual_apc`,
  `processed_qasr_segments`) is covered by `data_cleaned_text_merged_v1/clean/`, except `layla/`
  (Layla was never run through the text-cleaning notebooks — see the Layla sections above). The raw
  Layla source of truth stays independently safe in `Layla/`, so `data/layla/` was not a unique copy.
- `.venv/` (broken pre-existing venv, ~2.4GB) — had a broken `pyarrow.parquet` import and a `pip`
  with a dead shebang; fully superseded by a freshly built `.venv_data` (python3.11, from
  `requirements-data.txt`).

**Explicitly NOT deleted, and why:**

- `QASR/` raw (~150GB, including the un-extracted `qasr_wav_v1.0.tar.bz2.part_aa`/`part_ab`) — see
  "QASR Gap 2" below. Only 961/3,545 QASR transcripts have matched audio in
  `processed_qasr_segments/`; the two un-extracted archives (96.6GB compressed) are suspected to
  hold the other ~2,584 transcripts' audio (their compressed size alone exceeds the uncompressed
  size of everything already extracted, and their mtime is ~4 years newer than the already-extracted
  "alt" wav folder — the two were never the same data).
  **Update 2026-07-26: this suspicion was confirmed and the gap is now fully closed — see
  "QASR Full Extraction" below. `QASR/` raw should still not be deleted (annotations still live
  there), but the audio gap itself is resolved.**
- `Layla/` raw — needed for steps 10-11 (Layla normalize + shard), not yet run.
- `omnilingual_selected/` raw — needed as the input for step 7 (Omnilingual reclean v2).
- `Runs/` — do **not** assume this is safe to delete just because it also holds stale dialect-scan
  summary/log leftovers from the old VM. It also contains `.unreliable urns/` (~95MB of unrelated
  training-run notebooks/configs for whisper_medium, whisper_large_v3, qwen3_asr_0_6b,
  omnilingual_asr_1b) and a `README.md` describing it as the general training/eval workflow
  directory — unrelated to data curation.

Net effect: freed ~317GB (project usage 618GB → 301GB on `/workspace/asr/Palestinian-ASR/`).

## QASR Full Extraction (2026-07-26)

The QASR audio gap described above (and in "QASR Gap 2" / the QASR Audio Classification Repair
section below) turned out to be a genuinely incomplete archive, not corruption: the source archive
on `arabicspeechdata.blob.core.windows.net` has **4** split parts
(`qasr_wav_v1.0.tar.bz2.part_a{a,b,c,d}`, confirmed by listing the container directly with the
existing SAS token — `sig=D8KB7c2B5f4c6ikXd4nHl8QR620lTk3C0SMaB1rZeN0%3D`, valid until 2030-03-08),
totaling ~159GiB compressed. This box only ever had `part_aa`+`part_ab` (~90GB, 56.7% of the total)
downloaded, which is exactly why extraction stopped at 2,021/3,545 matched wav files (57%) — the
`pbzip2`/`tar` "Unexpected EOF" error from the 2026-07-18 extraction attempt was the decompressor
correctly running out of genuinely-missing input, not disk/stream corruption.

What was done to close the gap:

1. Downloaded the missing `part_ac` (48.3GB) and `part_ad` (23.9GB) with plain `wget` (no `-c` — see
   the gotcha in `data.md`'s QASR section, `wget -c` against this specific endpoint measured ~30x
   slower than a plain GET).
2. Extracted all 4 parts in one pass: `cat part_aa part_ab part_ac part_ad | pv | pbzip2 -dc -p4 |
   tar -xf - -C QASR/wav_all/`. Note `pbzip2 -p4` did not actually parallelize decompression — CPU
   usage stayed at ~100% (one core) throughout, because this archive was originally compressed with
   plain single-threaded `bzip2`, not `pbzip2` (parallel bzip2 *decompression* only works on files
   that were themselves compressed as multiple independent streams). Sustained throughput was
   ~15-20MB/s the whole way through, consistent with single-core bzip2 decompression speed.
3. Verified: 3,545 wav files extracted, matching all 3,545 xml transcripts exactly (100%, up from
   57%). Total pipeline time ~4h10m (downloads ~85min + extraction ~2h44m).

Result: `QASR/wav_all/` (220GB) on network storage now contains the **complete** QASR audio set.
`QASR/wav_extracted/` (125GB) and `QASR/alt/` (59GB) are now redundant strict subsets of `wav_all/`
and can be deleted to reclaim ~184GB (not yet done as of this writing — pending confirmation, since
their historical row counts were already baked into the now-superseded
`processed_qasr_segments/` numbers referenced elsewhere in this file).

**Update 2026-07-30:** the four split archive parts (`qasr_wav_v1.0.tar.bz2.part_a{a,b,c,d}`,
170.6 GB) were deleted after verifying all 3,545 extracted wav open cleanly. `wav_extracted/` and
`alt/` were sampled and confirmed byte-identical (md5) to their counterparts in `wav_all/`, so they
remain safe to delete, but as of this writing they still exist. Note `QASR/alt/...` is the
`original_audio_path` recorded in the 110 part-1 QASR shards; deleting it only makes that provenance
string stale, since the audio bytes live inside the parquet and the identical files remain under
`wav_all/`. The annotation archive `qasr_annotation_v1.0.tar.bz2` (77 MB) was deliberately kept —
the XML transcripts it holds are the only copy of the text.

Practical implication for the rest of the pipeline: any future re-run of QASR segmentation
(`preprocess/qasr_segment_to_arrow.py`), the QASR fast-cleaning pass, or the QASR audio dialect scan
should now point at `QASR/wav_all/` to get the full 3,545-transcript set, rather than the old
`QASR/wav_extracted/`/`QASR/alt/` (961- or 2,021-file partial sets) that earlier pipeline runs used.

## QASR Part 2: Closing the Segmentation Gap (2026-07-30)

The 2026-07-26 extraction closed the *audio* gap, but the cleaned dataset still only contained the
961 recordings that the original segmentation run had covered. This step segmented and cleaned the
remaining 2,584 recordings and merged them in, so `data_cleaned_text_merged_v1/` now reflects the
complete 3,545-recording QASR set.

Measured starting point (scan of all 440 QASR shards in the merge, clean + dropped):

- 424,086 rows from exactly **961 distinct `recording_id`s**, all under `QASR/alt/...`
- `QASR/wav_all/` holds 3,545 wav; the XML dir holds 3,545 transcripts
- **2,584 recordings uncovered**, all 2,584 with a matching XML

Audio integrity was verified before any archive was deleted: all 3,545 wav open cleanly, 0 zero-byte,
0 truncated, 235.2 GB, **2,041.9 hours** of audio.

### Step A: segment the uncovered recordings

```bash
cd /workspace/asr/Palestinian-ASR
/root/Palestinian-ASR/.venv_data/bin/python -u preprocess/qasr_segment_to_arrow.py \
  --wav-dir QASR/wav_all/alt/arabic-speech-web/mgb2.1/wav \
  --xml-dir QASR/mgb2.1/release/train_20210109/xml \
  --output-dir /workspace/asr/Palestinian-ASR/processed_qasr_segments_part2 \
  --wav-stem-list /workspace/asr/Palestinian-ASR/.logs/qasr_new_stems_2584.json
```

`--wav-stem-list` is a new optional flag on `preprocess/qasr_segment_to_arrow.py`: it restricts
segmentation to WAV stems listed in a JSON array. Default behaviour is unchanged when it is omitted.
It exists so the already-covered 961 recordings can be excluded without building a symlink farm,
which keeps `original_audio_path` pointing at the real `wav_all/` location. Running from the
`/workspace/...` root keeps that path relative, matching the rows already in the merge.

Result: **2,584/2,584 recordings → 1,177,332 segments, 304 Arrow shards** (151 GB), ~47 min.

### Step B: fast text clean

`.logs/clean_qasr_part2.py` is a copy of the step-5 cleaning script with only `INPUT_ROOT`,
`OUTPUT_ROOT`, and the discovery glob changed — the cleaning logic is untouched, so the new rows are
in exactly the same state as every other dataset in the merge. Output root
`data_cleaned_text_qasr_part2_v1/`; glob `processed_qasr_segments_part2/train/*.arrow`.

The `_part2` root name is load-bearing: `stable_file_id()` hashes the path relative to `INPUT_ROOT`,
so a distinct root guarantees output filenames disjoint from the 110 QASR shards already merged.
`merge_cleaned_outputs_and_report.py` hard-fails on a destination collision, and a re-run of the
original `processed_qasr_segments/` path would have collided on all 110.

Result: **1,177,332 rows, 304 clean shards**, ~29 min at ~690 rows/sec.

### Step C: merge

```bash
/root/Palestinian-ASR/.venv_data/bin/python -u scripts/merge_cleaned_outputs_and_report.py \
  --source /workspace/asr/Palestinian-ASR/data_cleaned_text_qasr_part2_v1 \
  --dest /workspace/asr/Palestinian-ASR/data_cleaned_text_merged_v1 \
  --intermediate-root /workspace/asr/Palestinian-ASR/intermediate/merged_cleaned_sources
```

~3 min: source and dest share a filesystem, so `shutil.move` is a rename and the ~150 GB never moves.
The pre-merge `merge_manifest.json` was preserved as `merge_manifest.pre_qasr_part2.json`.

### Verified end state of `data_cleaned_text_merged_v1/`

| dataset | clean shards | clean rows |
|---|---|---|
| **QASR combined** | **414** | **1,508,531** |
| ├ `processed_qasr_segments` (part 1, 961 recordings) | 110 | 399,633 |
| └ `processed_qasr_segments_part2` (part 2, 2,584 recordings) | 304 | 1,108,898 |
| `masc_c_only` | 417 | 373,464 |
| `casablanca_jordanian` | 2 | 1,695 |
| `casablanca_palestinian` | 4 | 1,328 |
| `omnilingual_apc` | 3 | 43 |
| **total** | **840** | **1,885,061** |

Dropped tree: 550 `audio_too_short` / 417 `contains_english` / 414 `contains_number` shards,
93,528 rows. Grand total 1,978,589 rows, **0 corrupt files** across all 2,221 shards.

QASR grew **399,633 → 1,508,531 clean rows (3.8x)**. Note this exceeds the ~1.19M "historical"
QASR figure cited elsewhere in this file — that number came from a partial run and should no longer
be treated as the target.

### Gotchas from this run

- **The project directory has a ~800 GB quota**, entirely invisible to `df` (which reports the
  cluster's 419 TB free). The first Step B died at 283/304 shards with
  `OSError [Errno 122] Disk quota exceeded`. Check headroom with a `dd` test write, not `df`.
- **A quota death corrupts far more than the file being written.** After the failure, 57 of 283
  clean shards *and* 392 of the dropped shards had unreadable parquet footers, scattered throughout
  the run rather than confined to the tail. Sampling a few shards gave a false all-clear; only a
  footer read of *every* file found them.
- **Do not trust a resumed run's report.** A resume that validates only `clean/` will happily leave a
  corrupt `dropped/` tree behind, and its report covers only the resumed subset. After the quota was
  lifted, the whole 304-shard pass was re-run from scratch rather than patched, and correctness was
  established by counting rows in the files: clean + dropped must equal Step A's segment count
  (1,177,332 = 1,177,332).
- **Concurrent sessions are a real hazard.** Another session re-ran the pipeline script mid-flight;
  its Step A skipped (shards present) and its Step B started cleaning partial Arrow output. Any
  re-runnable driver script here should take a lockfile.

### Ordering constraint with step 9

`scripts/create_data_with_final_omnilingual.py` hardlink-copies `data_cleaned_text_merged_v1/` into
`data/`, i.e. it **snapshots the merge at the moment it runs**, and refuses to run if `data/` already
exists. It must therefore run *after* this merge; running it before would silently produce a `data/`
containing only the old 399,633 QASR rows. Steps 7 and 8 (Omnilingual v2/v3) are unaffected — they
touch only `omnilingual_selected/` and their own output roots.

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

**Superseded 2026-07-31** — the original four-JSON, direct-into-`data/` flow described
below never actually ran end-to-end on this box: the raw `Layla/` source was lost and
had to be re-sourced, and the normalization pass grew to
six batches, closing a 42-row gap the first four left uncovered. Re-run instructions:

**Raw source, as it actually exists on network storage:**

- `Layla/Layla Witheeb Jordanian Arabic Acoustic Dataset/` — nested `region/speaker-code/`
  (e.g. `Amman/LA8F/`), not flat: 874 files total (218 `.txt` + 218 `.docx` +
  218 `.TextGrid` + 218 `.wav`/`.WAV`).
- `Layla/Layla Witheeb Jordanian Arabic Acoustic Dataset.zip` — the original archive.
  Deleted 2026-07-31 after verifying it held exactly the same 874 files as the
  extraction (`zipfile.testzip()` clean, member count matched) — a byte-for-byte
  redundant copy, same reasoning as the deleted QASR split archives above.
- `Layla/normalized_json/` — the hand-normalized phonetic-artifact-correction pass,
  now complete at 218/218 sources across six batch files (see that directory's own
  `README.md` for full provenance and a documented gotcha: speaker `AH25M` uses a
  lowercase `_arabic_transcription` suffix where every other speaker uses
  `_Arabic_transcription`).

**Step 1 — merge the six normalized batches into one file:**

```bash
python3 scripts/merge_layla_normalized_json.py
```

Reads the six `normalized_*.json` files, fails if the merged set isn't exactly 218
unique `source` keys, and writes `Layla/normalized_json/normalized_all.json`.

**Step 2 — build shards from the raw dataset, with the normalized text layered on:**

```bash
python3 scripts/build_layla_shards.py \
    --normalized-json Layla/normalized_json/normalized_all.json
```

This is the re-runnable replacement for the shard-writing half of
`finalize_data_with_layla.py` (see the script's own docstring). It matches each
`*_Arabic_transcription.docx` to its paired `.WAV`/`.wav` case-insensitively (so
`AH25M` isn't silently dropped), prefers the normalized text over the raw `.docx`
text per-row, and fills `gender` from the speaker-directory code. Output:
`processed_layla_shards_v1/layla/*.parquet` (nested under a `layla/` dir so the
`clean` stage's `dataset_from_file()` fallback labels every row `layla`), 218 rows,
6.70 hours audio, columns `audio`, `seg_id`, `transcription`, `gender`, `duration`,
`source_file`, `transcript_source` (`normalized_json` vs `raw_docx`, for traceability).
Verified 218/218 rows used the normalized text.

**Step 3 — run the actual `clean` + `assemble` stages (same code every other dataset
goes through — see `PIPELINE.md`), scoped to just Layla:**

```bash
python3 -m pipeline run --config configs/layla_only.yaml
```

`configs/layla_only.yaml` points `clean` at `processed_layla_shards_v1/` (which is
already ingest-stage-shaped) and `assemble` at an isolated scratch root — deliberately
*not* `data/` directly, since `assemble`'s stage runner `rmtree`s a pre-existing,
non-symlinked `output_root`, and `data/` holds 843 other datasets' shards that must
not be touched. Result: 218/218 rows kept (0 dropped — no English, no digits, no
sub-0.5s clips), with the standard `flag_contains_english` / `flag_contains_number` /
`flag_contains_bracket_token` / `flag_audio_too_short` / `flag_missing_duration` /
`manual_normalized_transcript` columns every other `data/` dataset carries.

**Step 4 — hardlink the assembled shards into `data/` by hand** (not via the config's
`assemble` stage, for the `rmtree`-safety reason above), stripping the redundant
`layla__layla__` prefix that falls out of `build_layla_shards.py` already naming its
own files `layla__data-*.parquet` inside a directory already called `layla/`:

Layla shards now living at the top level of `data/` (matching the historical
convention, alongside `data/clean/`, `data/dropped/`, `data/reports/`):

- `layla__data-00000-of-00004.parquet__b22390b869__clean.parquet`
- `layla__data-00001-of-00004.parquet__511d86b91d__clean.parquet`
- `layla__data-00002-of-00004.parquet__eb54c50e07__clean.parquet`
- `layla__data-00003-of-00004.parquet__3aa60d9c2a__clean.parquet`

Note the other 843 shards under `data/clean/` (QASR, MASC, Casablanca, Omnilingual)
have **not** been flattened to the top level on this box — that's the same
"Flattening `data/clean` into `data/`" operation described later in this file, just
not yet re-run here. Layla is flat because the steps above put it there directly;
it does not imply the rest of `data/clean/` has been migrated.

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

> **Superseded for train/val/test assignment (2026-09-03).** Everything below about the
> *lev / non_lev* labelling still stands, but the `0.70 / 0.15 / 0.15` partition in step 4
> is assigned per row and is **not speaker-disjoint**: ~100% of val's and ~99% of test's
> QASR recordings also have rows in train, and ~99% of both's MASC videos do. The pipeline
> now assigns whole speakers instead, via the `speaker_select` stage — see
> [SPEAKER_DISJOINT_SELECTION.md](SPEAKER_DISJOINT_SELECTION.md). Re-generated trees should
> come from that path; `data_curated_levant_binary_v1` predates it.

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

## Long-Audio Segmentation (VAD + Forced Alignment) — 2026-07-31

After dialect identification (text + audio stages above), the working `data/` root was
scanned for rows whose `duration` exceeds Whisper's 30s hard positional-embedding limit,
using headroom of 28s. This is a two-part stage: Part A segments the flagged audio
offline into a new versioned artifact without touching `data/`; Part B then splices those
segments into `data/` in place of the original long rows. Both parts are one-time,
re-runnable, resumable (checkpoint-file) scripts under `scripts/segmentation/`.

### Flagging pass

A cheap duration-only scan (no audio decode) of every `data/clean/*.parquet` and
top-level `data/layla__*.parquet` shard found:

| group | rows scanned | rows > 30s | max duration |
|---|---|---|---|
| `layla` | 218 | **218 (100%)** | 253.3s |
| `omnilingual_apc` | 416 | **375 (90%)** | 98.1s |
| `masc_c_only` | 373,464 | 15 | 43.0s |
| `casablanca_jordanian` | 1,695 | 0 | 17.8s |
| `casablanca_palestinian` | 1,328 | 0 | 29.5s |
| `processed_qasr_segments` (part 1) | 399,633 | 0 | 10.9s |
| `processed_qasr_segments_part2` | 1,108,898 | 0 | 10.9s |

608 flagged rows total. QASR (both parts) and Casablanca needed no segmentation — they
are already pre-segmented at the utterance level from forced alignment upstream. Output:
`segmented/flagged_over30s.json` (one record per flagged row: `file`, `row_index`,
`duration`, `group`).

### Part A — segment (`scripts/segmentation/segment_whisperx.py`)

Run in a dedicated venv (`/workspace/venv_whisperx`, `pip install whisperx silero-vad
soundfile hf_transfer`; needs `HF_HUB_ENABLE_HF_TRANSFER`-compatible `hf_transfer`
installed or the WhisperX align-model download fails). WhisperX primarily for its
packaged alignment model loader (`jonatasgrosman/wav2vec2-large-xlsr-53-arabic`), used
purely as an alignment tool, not for transcription — the existing transcript text
(`manual_normalized_transcript`, falling back to the group's raw text column) is what
gets aligned and sliced, never re-decoded from scratch.

Per flagged row:

1. Decode audio to mono float32 @16kHz.
2. **Silero VAD** → speech regions (silence gaps are the only legal cut points).
3. **WhisperX forced alignment** of the existing transcript against the audio
   (`whisperx.align`, `language_code="ar"`) → per-word `(start, end, score)`.
4. **Cut-point selection**: walk forward accumulating audio, cutting at the VAD silence
   gap nearest to (but not past) 28s; falls back to the largest inter-word pause in the
   window if VAD found no gap in range.
5. **Transcript slicing**: each transcript word is assigned to a segment by whether its
   alignment-timestamp midpoint falls before/after the cut point.
6. **Quality gate**: a segment is dropped (not down-weighted) if its text is empty, its
   duration is below 0.5s, its mean word alignment score is below `0.45`, or fewer than
   half its words got a timestamp at all (majority-untimed segments are almost always
   transcript/audio mismatch — the dominant failure mode noted in the build spec).

Run command:

```bash
/workspace/venv_whisperx/bin/python scripts/segmentation/segment_whisperx.py --device cuda
```

Resumable via `segmented/v1/checkpoint.txt` (one source shard per line). Output:

- `segmented/v1/<orig_file>__segmented.parquet` — one shard per flagged source file, one
  row per kept segment, columns: `audio` (struct, WAV PCM16 bytes), `text`, `group`,
  `orig_file`, `orig_row_index`, `orig_id`, `segment_idx`, `start_offset`, `end_offset`,
  `duration`, `align_score`, `n_words`.
- `segmented/v1/segmentation_report.json` — per-group totals (samples, segments kept /
  dropped, hours in / kept, errors).
- `segmented/v1/dropped_segments.json` — every dropped segment with its drop reason, for
  audit.

Result: **608/608 flagged rows processed, 0 errors, 2,322 segments kept, 7 dropped**
(6 `empty_text`, 1 `low_align_score`) — a ~0.3% loss, well within the "drop rather than
inject noise" policy. Alignment scores in spot checks were 0.71–0.79. Per-group:

| group | samples segmented | segments kept | dropped | hours in | hours kept |
|---|---|---|---|---|---|
| `layla` | 218 | 1,022 | 4 | 6.70 | 6.70 |
| `omnilingual_apc` | 375 | 1,270 | 3 | 7.71 | 7.70 |
| `masc_c_only` | 15 | 30 | 0 | 0.14 | 0.14 |

`segmented/` is a **new top-level directory, sibling to `data/`, not nested inside it** —
`data/` was not written to by Part A. The raw/staged corpora these rows came from remain
untouched and un-tagged, as does everything else in `data/`.

### Part B — splice into `data/` (`scripts/segmentation/replace_long_audio_in_data.py`)

Part A only produces a derived artifact; this script performs the actual "use these in
place of the corresponding data" replacement inside `data/`, one source shard at a time:

1. Back up the untouched original shard to `segmented/v1/backup_originals/<relative
   path under data/>` (4.2GB total across the 20 affected shards — cheap insurance
   before an in-place overwrite of real training data).
2. Remove exactly the flagged (> 30s) rows.
3. Insert the corresponding segment rows, mapped into **that shard's own column
   schema** — same `audio`/`duration`/id/text columns every other row in the file uses,
   so nothing downstream needs to special-case segmented rows. Every non-text, non-flag
   column (`gender`, `speaker_id`, `language`, `type`, `iso_639_3`, ...) is carried over
   from the original row onto every sub-segment it produced.
4. Recompute `flag_contains_english` / `flag_contains_number` / `flag_contains_bracket_token`
   / `flag_audio_too_short` / `flag_missing_duration` and `manual_normalized_transcript`
   with `pipeline/textnorm.py` (`has_english`, `has_number`, `has_bracket_token`,
   `normalize_arabic_transcript`) — the same functions the `clean` stage uses — instead
   of carrying stale flags computed against the pre-split, much longer original text.
5. Append 6 provenance columns to **every** row in the shard (null on untouched rows):
   `segment_source_file`, `segment_source_row_index`, `segment_idx`,
   `segment_start_offset`, `segment_end_offset`, `segment_align_score` — so segmented
   rows stay identifiable inside `data/` itself, not just in `segmented/v1/`.
6. Write to a staging copy under `segmented/v1/staged_replacement/`, verify the parquet
   footer's row count, only then overwrite the original in place.

New row IDs are `{orig_id}__seg{NN}` (e.g. `layla_Ajlun_JS1M_JS1M_Laylawetheeb_read__seg00`,
`iXuQlqAYIJY__seg00`, `s01__seg00`).

Run command:

```bash
/workspace/venv_omni_gpu/bin/python scripts/segmentation/replace_long_audio_in_data.py
```

(Any Python with `pyarrow` works — Part B never decodes audio, so it doesn't need the
WhisperX venv.) Resumable via `segmented/v1/replacement_checkpoint.txt`. Manifest at
`segmented/v1/replacement_manifest.json` (per-shard rows before/removed/added/after).

Result, cross-checked exactly against Part A's segmentation report:

| group | shards touched | rows removed (>30s) | rows added (segments) |
|---|---|---|---|
| `layla` | 4/4 | 218 | 1,022 |
| `masc_c_only` | 10/417 | 15 | 30 |
| `omnilingual_apc` | 6/6 | 375 | 1,270 |
| **total** | **20 shards** | **608** | **2,322** |

`data/` row count across the touched shards: 9,702 → 11,416. Every other shard in
`data/` (the other 408 `masc_c_only` shards, all QASR, all Casablanca) is byte-identical
to before this stage — Part B only opens shards that appear in `flagged_over30s.json`.

### Known ordering gap: stale dialect-ID rows for the 15 replaced `masc_c_only` rows

Text + audio dialect identification (see "Binary Levantine Split Curation" /
"QASR Audio Classification Repair" above) already ran against `masc_c_only` *before*
this segmentation stage, keyed by `video_id`. The 15 `masc_c_only` rows this stage
removed no longer exist under their original `video_id`s (their 30 replacement
sub-segments have new `{video_id}__segNN` IDs that were never dialect-scored), so:

- `Runs/text_dialect_scan_marbertv2_written_clean_masc_c/row_probabilities.jsonl` and
  `Runs/dialect_scan_badrex_mms300m_lev08_text_candidates_masc_c/row_probabilities.jsonl`
  each carry 15 now-orphaned `video_id` rows.
- The 30 new segment rows have no dialect-ID score at all.

This is a **15-row / 373,464-row (~0.004%) gap** in a stage (14, the Levant binary
split) that had not started as of this writing. `layla`
and `omnilingual_apc` were never part of the dialect-ID scan (only `masc_c` and `qasr`
are), so they're unaffected. If step 14 is built before this gap is closed, either
re-score just these 30 new rows or accept the negligible loss — do not assume the
existing `row_probabilities.jsonl` files already cover them.

**Update 2026-07-31, later same day:** step 14 was built (see "QASR and MASC-C Dialect
Scans, Fresh Full Runs" below) without closing this gap — the 30 new segment rows scored
`non_lev` by default (no score present, so they fail the `>= 0.80` thresholds). Accepted
as the documented negligible loss, not re-scored.

## QASR and MASC-C Dialect Scans, Fresh Full Runs — 2026-07-31

Both the "Temporary QASR note" and "Current limitation" text above (under "Binary
Levantine Split Curation" and "QASR Audio Classification Repair") describe a partial,
resource-constrained state that no longer applies. What actually happened, same day as
the segmentation work above:

- **The PCM-aware decode fix is not a "repair mode"** — it's the permanent, unconditional
  behavior of `dialect_identifiaction/arabic_dialect_scan_badrex_mms300m.py` (see
  `_decode_pcm16le_bytes` / the `sampling_rate` + `_looks_like_encoded_audio` branch in
  `predict_one`). Every invocation of that script gets it; there is no separate "fixed"
  vs "unfixed" mode to choose between anymore.
- **`scripts/repair_qasr_audio_and_rebuild_levant_binary.py` was never run and is now
  moot.** It was always a thin subprocess wrapper with no logic of its own (reruns the
  same audio-dialect script, then the same `create_levant_non_levant_splits.py`) — calling
  the two underlying scripts directly, as done below, is equivalent and is what actually
  happened. Its named output paths
  (`Runs/dialect_scan_badrex_mms300m_lev08_text_candidates_masc_c_qasr_qasrfix/`,
  `data_curated_levant_binary_v2_qasr_audio_fix/`) were never created and don't exist.
- **QASR**, full 1,508,531-row set, scanned end to end:
  - Text: `Runs/text_dialect_scan_marbertv2_written_clean_qasr_only/row_probabilities.jsonl`
    — 122,348 rows with text `LEV >= 0.80`.
  - Audio (PCM-aware decode, 0 errors, 108,453/122,348 successfully decoded — the
    remainder were <2s clips, skipped by design): `Runs/dialect_scan_badrex_mms300m_lev08_text_candidates_qasr_only/row_probabilities.jsonl`.
  - Final split: `data_curated_levant_binary_qasr_only_v1/` —
    `qasr/lev` 31,178 rows (21,824/4,676/4,678), `qasr/non_lev` 1,477,353 rows
    (1,034,147/221,602/221,604).
- **MASC-C**, full 373,464-row cleaned set (the earlier 2026-07-30 attempt only covered
  85.2% of rows before stopping — treat that one as superseded, not resumed from):
  - Text: `Runs/text_dialect_scan_marbertv2_written_clean_masc_c_only/row_probabilities.jsonl`
    — 48,455 rows with text `LEV >= 0.80`.
  - Audio: `Runs/dialect_scan_badrex_mms300m_lev08_text_candidates_masc_c_only/row_probabilities.jsonl`
    — 45,099/48,455 successfully decoded, 0 errors.
  - Final split: `data_curated_levant_binary_masc_c_only_v1/` —
    `masc/lev` 8,214 rows (5,749/1,232/1,233), `masc/non_lev` 365,265 rows
    (255,685/54,789/54,791). Total 373,479 (373,464 pre-segmentation + 15 removed − 15
    + 30 added by the Long-Audio Segmentation splice above; the 30 new rows carry no
    dialect score and default to `non_lev` — see the ordering-gap note above).

**Superseded the same day — see "Final Combined Build" below.** At the time this was
written, the above was two separate scoped outputs, not the single combined tree the
"Binary Levantine Split Curation" section describes. Both `_qasr_only_v1` and
`_masc_c_only_v1` were deleted after the combined build below was verified — they added
nothing not already reflected there.

## Final Combined Build, With Native Test-Split Preservation — 2026-07-31

The two scoped runs above were merged into the single `data_curated_levant_binary_v1/`
this document was always meant to describe, with one deliberate change to the splitting
rule for three leaves: **`masc`, `casa/pal`, and `casa/jor` now keep each source
dataset's own native `test-NNNNN-of-NNNNN.parquet` shard(s) as the final `test` split
verbatim**, instead of resampling test membership randomly. Every other row for those
three leaves (originally `train-*`/`validation-*` shards) is pooled and re-split 80/20
into train/val. `qasr`, `omni`, and `layla` are untouched — same global random
train/val/test ratio split (0.70/0.15/0.15) as before, no native-test carving (QASR's
own shards carry no test/train split to begin with; omni/layla were never shipped with
one either).

Native test/remainder shard counts, confirmed by matching the `test-NNNNN-of-NNNNN.parquet`
filename segment:

| leaf | native test shards | remainder shards (re-split 80/20) |
|---|---|---|
| `masc` | 9 (of 417) | 408 (398 `train-*` + 10 `validation-*`) |
| `casa/pal` | 2 (of 4) | 2 (`validation-*`) |
| `casa/jor` | 1 (of 2) | 1 (`validation-*`) |

Script: [scripts/rebuild_levant_binary_parallel_resumable.py](/root/Palestinian-ASR/scripts/rebuild_levant_binary_parallel_resumable.py)
(a from-scratch rewrite of `create_levant_non_levant_splits.py`'s logic, not an edit to
that file — it stays as documentation of the original combined-random-split design).
Combined data-root: `.logs/full_combined_data_root/` (847 symlinks spanning all six
sources). Combined dialect-probability inputs: `.logs/combined_row_probs/{text,audio}_row_probabilities.jsonl`
(the QASR-only and MASC-C-only scan outputs from the section above, concatenated —
1,881,995 text rows, 170,803 audio rows, both counts matching sums exactly).

**Parallel + resumable design.** Six independent worker processes (`--worker
{qasr,masc,omni,layla,casa_pal,casa_jor}`), one per leaf, each touching only its own
exclusive output subdirectories — no two workers ever write into the same directory, so
there is no repeat of the concurrent-write corruption risk documented elsewhere in this
file. Each worker checkpoints how many of its fixed-order source files it has fully
committed (closing all its parquet shard writers at each checkpoint, every 20 files or
at the end), so a crash only ever costs re-processing files since the last checkpoint,
never a full restart. `ShardWriter` self-heals on startup: it scans its output directory
for the highest existing shard index, validates the last shard's footer, and deletes it
if truncated (crash mid-write) before continuing.

**A real bug was caught and fixed during this run**, worth recording: the first version
reused the 3-way `split_counts()` helper (which always carves a `test` bucket from
flooring leftover, e.g. `floor(847*0.8) + floor(847*0.2) = 846`, leaving `1` row assigned
to `test`) to implement the 80/20 remainder split — but the remainder path explicitly
discards anything routed to `test`. Result: exactly 1 row silently vanished from each of
`casa_pal`'s and `casa_jor`'s remainder pools before the fix (caught by summing
train+val+test against the known input total and finding a 1-row gap in both). Fixed
with a dedicated `make_two_way_position_set()` that computes `val_count =
round(total * val_ratio)`, `train_count = total - val_count` — every row lands in
exactly one bucket, nothing is ever discarded. `masc`'s remainder phase was killed and
restarted with the fix before it had checkpointed (so no data loss there); `casa_pal`/
`casa_jor` were deleted and re-run from scratch (cheap — under 1,700 rows each).

**A second bug, in reporting only (not the data), was caught and fixed the same way:**
`masc`'s native-test phase (9 shards) completed and checkpointed in the *first* run,
before the fix required killing and restarting the whole `masc` worker. On restart, the
checkpoint correctly made it skip re-processing those 9 already-committed files — but
the fresh process's in-memory row counters start at zero, so its self-reported summary
undercounted `masc/lev`/`masc/non_lev` by exactly the amount the first (now-exited)
process had already safely written. The parquet files on disk were correct the whole
time; only `reports/worker_summaries/masc.json`'s self-reported counts were wrong.
Fixed by [.logs/finalize_ground_truth_summary.py](/root/Palestinian-ASR/.logs/finalize_ground_truth_summary.py),
which builds the final `reports/summary.json` by scanning row counts directly from the
output parquet files rather than trusting any worker's in-process counters — the only
approach that stays correct regardless of how many resume cycles a leaf went through.

**Verified final result** (every row from the 1,887,366-row combined input accounted for
exactly once, train+val+test summing to each leaf's known-correct total):

| leaf | train | val | test | total |
|---|---|---|---|---|
| `masc/lev` | 6,383 | 1,596 | 235 | 8,214 |
| `masc/non_lev` | 285,502 | 71,375 | 8,388 | 365,265 |
| `qasr/lev` | 21,824 | 4,676 | 4,678 | 31,178 |
| `qasr/non_lev` | 1,034,147 | 221,602 | 221,604 | 1,477,353 |
| `omni` | 917 | 196 | 198 | 1,311 |
| `layla` | 715 | 153 | 154 | 1,022 |
| `casa/pal` | 531 | 133 | 664 | 1,328 |
| `casa/jor` | 678 | 169 | 848 | 1,695 |
| **total** | | | | **1,887,366** |

Note `masc`'s and `qasr`'s `lev`/`non_lev` totals are unchanged from the scoped runs
above (8,214/365,265 and 31,178/1,477,353) — only *which* rows landed in `test` changed
(native shard membership instead of random sampling), not the underlying dialect
classification. Output: `/workspace/asr/Palestinian-ASR/data_curated_levant_binary_v1/`,
`reports/summary.json`. The two superseded scoped outputs
(`data_curated_levant_binary_qasr_only_v1/`, 191GB; `data_curated_levant_binary_masc_c_only_v1/`,
78GB) were deleted after this was verified, freeing 269GB against this project's
~800GB quota (see the quota gotcha earlier in this file) at a moment it was needed —
usage had reached 782GB/~800GB with this build still writing.

## Note: Same Script Reused Across Multiple Steps

Several pipeline steps are not distinct implementations — they reuse an already-documented script
against a different dataset or a different input, sometimes via a generator script and sometimes
via direct subprocess orchestration:

- **Fast text-cleaning notebook, run 3x.** The base notebook
  [preprocess/fast_asr_data_cleaning_text_only_arrow_parquet.ipynb](/home/MohammadNabulsi/whisper/preprocess/fast_asr_data_cleaning_text_only_arrow_parquet.ipynb)
  (broad pass, effectively `masc_c_only`), the QASR+Casablanca+Omnilingual notebook
  [preprocess/fast_asr_data_cleaning_text_only_arrow_parquet_qasr_casablanca_omni.ipynb](/home/MohammadNabulsi/whisper/preprocess/fast_asr_data_cleaning_text_only_arrow_parquet_qasr_casablanca_omni.ipynb),
  and the Layla notebook
  [preprocess/fast_asr_data_cleaning_text_only_arrow_parquet_layla.ipynb](/home/MohammadNabulsi/whisper/preprocess/fast_asr_data_cleaning_text_only_arrow_parquet_layla.ipynb)
  all share the same cleaning logic (drop English letters, drop numbers, drop `<0.5s` duration,
  create `manual_normalized_transcript`). Only `INPUT_ROOT`/`OUTPUT_ROOT`/discovery globs differ.
  The QASR+Casablanca+Omnilingual notebook is mechanically generated from the base notebook by
  [preprocess/build_qasr_casablanca_omnilingual_cleaning_notebook.py](/home/MohammadNabulsi/whisper/preprocess/build_qasr_casablanca_omnilingual_cleaning_notebook.py);
  the Layla notebook is a hand-adapted copy of the same base with the same diff shape (config cells
  changed, cleaning logic untouched).

- **QASR repair-and-rebuild orchestrator has no processing logic of its own.**
  [scripts/repair_qasr_audio_and_rebuild_levant_binary.py](/home/MohammadNabulsi/whisper/scripts/repair_qasr_audio_and_rebuild_levant_binary.py)
  is a subprocess wrapper: it reruns
  [dialect_identifiaction/arabic_dialect_scan_badrex_mms300m.py](/home/MohammadNabulsi/whisper/dialect_identifiaction/arabic_dialect_scan_badrex_mms300m.py)
  (the same script used for the original audio dialect ID pass) with the PCM-aware loader, then
  reruns [scripts/create_levant_non_levant_splits.py](/home/MohammadNabulsi/whisper/scripts/create_levant_non_levant_splits.py)
  (the same script used for the first binary split) against the repaired audio output. Neither call
  is a new implementation.

- **The audio dialect scan script (`arabic_dialect_scan_badrex_mms300m.py`) has been invoked at
  least three times** against different candidate subsets over the pipeline's history: once against
  combined `masc_c`+`qasr` candidates (`Runs/dialect_scan_badrex_mms300m_lev08_text_candidates_masc_c_qasr/`),
  once against a `qasr`-only candidate set (`Runs/dialect_scan_badrex_mms300m_lev08_text_candidates_qasr_only_qasrfix/`),
  and again via the repair-and-rebuild orchestrator above.

## Related Utility Scripts

- [scripts/compute_data_duration_stats.py](/home/MohammadNabulsi/whisper/scripts/compute_data_duration_stats.py): computes duration stats for staged datasets
- [outputs/compute_data_duration_stats.py](/home/MohammadNabulsi/whisper/outputs/compute_data_duration_stats.py): copied next to `outputs/data_duration_stats.json`
- [preprocess/unify.py](/home/MohammadNabulsi/whisper/preprocess/unify.py): older/alternate raw unification pipeline, different output layout from current `data/`

## Final Dataset Statistics — 2026-07-31

`scripts/compute_data_duration_stats.py` (above) predates the current parquet-shard
layout and doesn't run against it as-is (it expects raw file layouts, e.g. reading
Layla's `.wav` files directly off disk). The actual stats for
`data_curated_levant_binary_v1/` were generated by
[scripts/compute_final_dataset_stats.py](/root/Palestinian-ASR/scripts/compute_final_dataset_stats.py),
written specifically for this build. It reads only the `duration` column from every
shard (no audio decode) and reports bottom-up, from the smallest directory on disk up
to the grand total:

1. leaf within split (e.g. `train/masc/lev`) — the smallest unit, one row per actual
   output directory
2. leaf across all splits (e.g. `masc/lev` = train+val+test)
3. source across all splits, sub-labels merged (e.g. `masc` = lev+non_lev combined,
   `casa` = pal+jor combined)
4. split across all sources (e.g. `train` = every leaf combined)
5. grand total

Run: `python scripts/compute_final_dataset_stats.py`. Output:
`data_curated_levant_binary_v1/reports/duration_stats.json` (full breakdown of all 24
leaf-within-split cells plus every aggregation level above).

**Level 5 — grand total:** 1,887,366 rows, **2,209.91 hours**, 224 shards.

**Level 4 — per split:**

| split | rows | hours |
|---|---|---|
| train | 1,350,697 | 1,578.05 |
| val | 299,900 | 350.27 |
| test | 236,769 | 281.60 |

**Level 3 — per source (lev/non_lev and pal/jor merged):**

| source | rows | hours |
|---|---|---|
| qasr | 1,508,531 | 1,778.97 |
| masc | 373,479 | 412.39 |
| casa | 3,023 | 3.95 |
| omni | 1,311 | 7.91 |
| layla | 1,022 | 6.70 |

**Level 2 — per leaf, across all splits:**

| leaf | rows | hours |
|---|---|---|
| `qasr/non_lev` | 1,477,353 | 1,741.14 |
| `masc/non_lev` | 365,265 | 401.88 |
| `qasr/lev` | 31,178 | 37.83 |
| `masc/lev` | 8,214 | 10.51 |
| `omni` | 1,311 | 7.91 |
| `layla` | 1,022 | 6.70 |
| `casa/jor` | 1,695 | 1.98 |
| `casa/pal` | 1,328 | 1.97 |

Sanity check: Layla's 6.70h here matches the "Long-Audio Segmentation" section's
`hours kept` figure for Layla exactly (218 recordings → 1,022 segments, no duration
lost). Omni's 7.91h is slightly higher than that section's "7.70 hours kept" because
this leaf also includes the rows that were always ≤30s and never went through
segmentation at all, not just the 375 that were flagged and re-cut.

