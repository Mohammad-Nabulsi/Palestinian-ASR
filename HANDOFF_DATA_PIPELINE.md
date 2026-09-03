# HANDOFF — Data prep pipeline run (this session)

_Last updated: 2026-09-03. Read this first if picking this up cold._

## UPDATE 2026-09-03 — the split is now speaker-disjoint, new stage `speaker_select`

Step 14's train/val/test assignment was a per-row hash, which put ~100% of val's and ~99%
of test's QASR recordings in train as well (measured with
`scripts/r2_speaker_group_disjointness.py`). A new `speaker_select` stage now runs **after
both dialect passes**, groups rows per speaker (QASR `recording_id` / MASC-C `video_id`),
scores each speaker on both models' agreement, and assigns whole speakers to val / test /
train; `split` follows that assignment instead of hashing. Disjointness is verified in the
stage and is a hard failure. Full method, numbers and caveats:
[SPEAKER_DISJOINT_SELECTION.md](SPEAKER_DISJOINT_SELECTION.md).

The short-lived `speaker_aggregate` stage (which sat *between* the two dialect passes) is
removed — it only saw text scores, so it could not judge a speaker on both models.

## UPDATE 2026-07-31 — long-audio (>30s) segmentation done, new step 16

Added a new step **16** (table below): flagged and segmented every `data/` row longer
than Whisper's 30s limit (608 rows: 218/218 `layla`, 375/416 `omnilingual_apc`, 15
`masc_c_only`; QASR and Casablanca needed none) via Silero VAD + WhisperX forced
alignment, then spliced the resulting 2,322 sub-30s segments into `data/` in place of
the originals. Full detail, commands, and per-group numbers: `DATA_CURATION.md`,
"Long-Audio Segmentation (VAD + Forced Alignment)". Originals backed up to
`segmented/v1/backup_originals/` before any overwrite.

**Ordering note for step 14 (not yet started):** dialect ID (steps 12-13) ran on
`masc_c_only` *before* this segmentation step, keyed by `video_id`. The 15
`masc_c_only` rows this step replaced are gone under their old `video_id`s; their 30 new
`{video_id}__segNN` sub-rows were never dialect-scored. Negligible volume
(15/373,464 ≈ 0.004%) but real — see the "Known ordering gap" note in `DATA_CURATION.md`
before trusting step 14's coverage to be exhaustive. `layla`/`omnilingual_apc` were never
in the dialect-ID scan, so unaffected.

## UPDATE 2026-07-30 — QASR is now COMPLETE in the merge; steps 6-8 done

Supersedes the 07-26 and 07-16 notes below wherever they conflict.

- **The QASR segmentation gap is closed.** The 2,584 recordings that had audio but were never
  segmented are now segmented, cleaned, and merged. `data_cleaned_text_merged_v1/` holds
  **840 clean shards / 1,885,061 rows**, of which QASR is **414 shards / 1,508,531 rows** — up from
  110 shards / 399,633 rows, a 3.8x increase. Verified 0 corrupt files across all 2,221 shards
  (clean + dropped), with clean+dropped reconciling exactly against the segmenter's own count
  (1,177,332 = 1,177,332). Full detail: `DATA_CURATION.md`, "QASR Part 2" section.
- **Steps 6, 7, 8 are done.** Step 6 (merge) was already done before this session; steps 7 and 8
  (Omnilingual v2 reclean, v3 token-span recovery) completed 2026-07-30 ~14:05 by a parallel session.
- **Step 9 ran 2026-07-30 ~16:30 UTC** (after the QASR merge, as required) — see the pipeline
  table below. It now snapshots via **symlinks, not hardlinks** (MooseFS quota charges hardlinks
  as full copies). `intermediate/` was deleted by the script as designed.
- **Steps 12-13 (dialect ID) are done for MASC** (2026-07-30, GPU box) — see table. QASR remains.
  Note for step 14: the scan JSONLs record `source_file` under `data_cleaned_text_merged_v1/clean/`,
  and `data/clean/` symlinks resolve there too, but `create_levant_non_levant_splits.py` matches by
  resolved absolute path — verify the match (or patch to basename matching) before trusting counts.
- **Storage:** the 4 QASR split archives (170.6 GB) were deleted after verifying the extraction.
  `QASR/wav_extracted/` (125 GB) + `QASR/alt/` (59 GB) are verified byte-identical subsets of
  `wav_all/` and are still deletable for another 184 GB.
- **New hazard to know about — a ~800 GB quota on the project dir that `df` does not show.** It
  killed a Step B run mid-write and corrupted 57 clean + 392 dropped shards scattered across the
  run, not just the file being written. Test headroom with a `dd` write, and validate parquet
  footers on *every* output file before trusting a run — sampling gives false all-clears.
- **Multiple Claude Code sessions were operating on this repo concurrently**, including one that
  re-ran the QASR pipeline script mid-flight against partial output. Add a lockfile to any
  re-runnable driver script here.

## UPDATE 2026-07-26 — read this before trusting anything below dated 07-16

Significant progress happened between the 07-16 snapshot below and today, both in earlier sessions
(confirmed by inspecting network storage timestamps/content) and in this session. Corrections to the
07-16 table:

- **Step 6 (merge) is done.** `data_cleaned_text_merged_v1/` exists on network storage (not present
  as of 07-16). Repo symlinks for `Layla`, `QASR`, `omnilingual_selected`, and
  `data_cleaned_text_merged_v1` were missing from the local clone (only `Runs`/`intermediate` existed)
  and have been added, matching `scripts/stage_raw_datasets.py`'s symlink pattern.
- **Steps 12-13 (dialect ID) were attempted** (2026-06-27) but left no usable
  `row_probabilities.jsonl` artifacts on disk — only `summary.json`/log files survived on network
  storage. The audio-stage run (`Runs/dialect_scan_badrex_mms300m_lev08_text_candidates_masc_c_qasr/`)
  also used the pre-fix QASR loader and had a 68% error rate (91,351/133,053 rows). A second,
  fixed-loader attempt (`..._qasr_only_qasrfix/`) only got to 15.3% before stopping. **Both need a
  full rerun from scratch** — treat steps 12-13 as still `⏳ not started` in practice, not `done`.
- **Gap 1 (Layla) is partially resolved.** Raw Layla audio is back (874 files under `Layla/`,
  confirmed present as of 07-18/07-26) — the "total loss" described below no longer applies. However
  the 4 merged/normalized transcript JSONs (`normalized_output_appended.json` etc.) that steps 10-11
  need are still missing, and no `layla__*.parquet` shards exist yet. Steps 10-11 remain blocked, but
  now on "re-run normalization+sharding," not "re-source the raw dataset from scratch."
- **Gap 2 (QASR) is now fully resolved (this session, 2026-07-26).** The archive was missing 2 of its
  4 split parts (`part_ac`, `part_ad` — confirmed via the container's blob listing, not a corruption
  issue). Downloaded both (do **not** use `wget -c` against
  `arabicspeechdata.blob.core.windows.net` — measured ~30x slower than plain `wget` for a fresh
  download, see gotchas) and extracted all 4 parts in one pass into a new `QASR/wav_all/` (220GB):
  **3,545/3,545 wav files now match all 3,545 xml transcripts — 100%, up from 57%.** `pbzip2 -p4`
  did not parallelize (this archive is plain single-threaded `bzip2`, not multi-stream `pbzip2`
  output — decompression is inherently single-core regardless of flags). Old `QASR/wav_extracted/`
  (125GB) and `QASR/alt/` (59GB) are now redundant subsets of `wav_all/`, not yet deleted.
  Full details/commands: `data.md`'s QASR section and `DATA_CURATION.md`'s "QASR Full Extraction"
  section, both updated today. Script used: `.logs/qasr_full_extract.sh`.
- Net effect: **the QASR audio gap that blocked getting the full ~1.19M-row QASR volume is gone.**
  Steps 12-15 can now be re-run against the complete QASR set instead of the 27%/57% partial one.



Goal: run the raw→cleaned data pipeline described in `DATA_CURATION.md` on this box
(`/root/Palestinian-ASR`, symlinked to network storage at `/workspace/asr/Palestinian-ASR/`),
since almost every intermediate/output directory there turned out to be **empty** — only the
raw source datasets (QASR, Casablanca, MASC-Arabic2, Omnilingual, partially) had survived
whatever migration put this box's `/workspace` in place. See prior conversation turns for the
full audit; this file tracks execution state going forward.

## Is it on network storage? Yes.

Every output below was written directly to `/workspace/asr/Palestinian-ASR/...` (either via the
repo's symlinks, which point there, or in the notebook-derived scripts by pointing `OUTPUT_ROOT`
straight at the `/workspace/...` path to dodge a symlink/`shutil.rmtree` crash — see Gotchas).
Verified with `du -sh` against the real `/workspace` paths, not just the local symlink view.

## What's actively running right now

- **One background check**: `tar -tjf QASR/qasr_wav_v1.0.tar.bz2.part_aa | wc -l` (task id
  `bzrej4828`, output at `/tmp/part_aa_listing.txt` / `/tmp/part_aa_err.txt`), started ~18:34 UTC.
  Purpose: check whether the original split `.tar.bz2` archives contain **more** QASR audio than
  the already-extracted `QASR/alt/` folder (which only has 961 wav files — see Gap 2 below). This
  is a single-threaded bzip2 decompression of a 48GB file, so it's slow (~2 lines/min observed) —
  expect it to take up to an hour. Check `wc -l /tmp/part_aa_listing.txt` — if the process
  (`ps -ef | grep "tar -tjf"`) is gone, it's done; compare final entry count against 961.
- Nothing else. All prior background data-processing jobs (steps 1, 2, 4, 5 below) have exited.
  The `Runs/`-watching Monitor tasks from earlier were stopped once their jobs finished.

## Pipeline status

| # | Step | Status | Result |
|---|---|---|---|
| 1 | QASR segment→Arrow (`preprocess/qasr_segment_to_arrow.py`) | ✅ done | 961/961 wav files → 424,086 segments, 110 shards → `processed_qasr_segments/` |
| 2 | MASC filter `type=='c'` (`scripts/filter_masc_c_only.py`) | ✅ done | 417/417 shards, kept 373,624/909,020 rows → `data/masc_c_only/` |
| 3 | Stage raw datasets (`scripts/stage_raw_datasets.py`) | ✅ done | `data/` populated (see below); Layla staged 0 files (Gap 1) |
| 4 | Fast clean, broad pass (`preprocess/fast_asr_data_cleaning_text_only_arrow_parquet.ipynb`) | ✅ done | 373,624 total, 373,464 kept, 160 dropped (too short) → `data_cleaned_text_v1/` (417 clean shards) |
| 5 | Fast clean, targeted pass (QASR+Casablanca+Omni notebook) | ✅ done, **QASR partial** | 427,633 total, 402,699 kept → `data_cleaned_text_qasr_casablanca_omni_v1/` (119 clean shards) — see per-dataset breakdown below |
| 5b | QASR part 2: segment + clean the 2,584 uncovered recordings | ✅ done 07-30 | 1,177,332 segments → 304 clean shards / 1,108,898 rows; see `DATA_CURATION.md` "QASR Part 2" |
| 6 | Merge cleaned outputs (`scripts/merge_cleaned_outputs_and_report.py`) | ✅ done | `data_cleaned_text_merged_v1/` = 840 clean shards / 1,885,061 rows, QASR 414 shards / 1,508,531 rows, 0 corrupt |
| 7 | Omnilingual reclean v2 (`scripts/reclean_omnilingual_v2.py`) | ✅ done 07-30 | `data_cleaned_text_omnilingual_v2/` — 517 total, 406 kept, 111 dropped English |
| 8 | Omnilingual recovery v3 (`scripts/recover_omnilingual_token_span_rows_v3.py`) | ✅ done 07-30 | `data_cleaned_text_omnilingual_v3_recovered_from_v2/` — 10 recovered, 101 still English |
| 9 | Rebuild `data/` with final Omnilingual rows (`scripts/create_data_with_final_omnilingual.py`) | ✅ done 07-30 | `data/clean/` = 837 symlinked shards + 6 real Omnilingual shards (416 rows). **Script patched to symlink instead of hardlink** — MooseFS charges hardlinks against the ~800GB quota like full copies (first attempt died on `Errno 122`; partial tree was removed). Repo symlink `data` → `/workspace/asr/Palestinian-ASR/data` added. |
| 10-11 | Layla normalize/shard + flatten into `data/` | ✅ done 07-31 | 218/218 rows, 218/218 using the completed 6-batch normalization (`normalized_all.json`), 0 dropped by the `clean` stage. 4 shards hardlinked to top-level `data/layla__*.parquet`. See `DATA_CURATION.md` "Layla Prompt Merge and Sharding". |
| 12 | Text dialect ID, MarBERTv2 | 🟡 MASC done 07-30 | MASC-only run (GPU box): 318,078/373,464 rows → `Runs/text_dialect_scan_marbertv2_written_clean_masc_c/row_probabilities.jsonl`; 42,092 rows with LEV≥0.80. ~55k rows skipped by `make_uid` video_id collisions (same as original run). **QASR (1.5M rows) still to do.** |
| 13 | Audio dialect ID, badrex mms300m | 🟡 MASC done 07-30 | On the 42,092 text candidates: 39,162 classified (2,930 <2s, 2 errors) → `Runs/dialect_scan_badrex_mms300m_lev08_text_candidates_masc_c/row_probabilities.jsonl`; **6,949 rows pass text≥0.80 AND audio Levantine≥0.80**. Crashed 2x at ~58% from 31GB cgroup page-cache pressure; `--resume` recovered. **QASR still to do.** |
| 14 | Build Levant/non-Levant binary split | ⏳ not started | → `data_curated_levant_binary_v1/`. **Now runs through `speaker_select` first** (2026-09-03) so splits are speaker-disjoint; `split` reads `speaker_assignments.json` rather than hashing rows. |
| 15 | QASR audio-decode repair + rebuild | ⏳ not started | → `data_curated_levant_binary_v2_qasr_audio_fix/` |
| 16 | Long-audio (>30s) segmentation: flag → VAD+forced-align segment → splice into `data/` (`scripts/segmentation/segment_whisperx.py` + `scripts/segmentation/replace_long_audio_in_data.py`) | ✅ done 07-31 | 608 rows flagged (218 `layla`, 375 `omnilingual_apc`, 15 `masc_c_only`) → 2,322 segments, spliced into `data/` in place; see `DATA_CURATION.md`. **Ran after step 11 (Layla), before steps 14-15 — see ordering note above re: stale `masc_c_only` dialect-ID rows.** |

### Step 5 per-dataset breakdown (from `data_cleaned_text_qasr_casablanca_omni_v1/reports/cleaning_report.json`)

| dataset | total | kept | historical (DATA_CURATION.md) |
|---|---|---|---|
| qasr | 424,086 | 399,633 | 1,194,234 — **short, see Gap 2** |
| casablanca_jordanian | 1,696 | 1,695 | 1,696 — matches exactly |
| casablanca_palestinian | 1,334 | 1,328 | 1,334 — matches exactly |
| apc_Arab (omnilingual) | 517 | 43 | 517 — matches exactly (kept count differs by design, v1 fast-pass vs v2/v3 recovery logic) |

## Known gaps (data genuinely missing/short on this network storage copy)

1. **Layla — resolved 07-31.** Was total loss as of the note below; raw audio/transcripts came
   back (874 files) and a full 6-batch hand-normalization pass (218/218 sources) landed in
   `Layla/normalized_json/`. Shards are now built, cleaned, and hardlinked into `data/` — see
   `DATA_CURATION.md` "Layla Prompt Merge and Sharding" and pipeline status row 10-11 above. Original
   note, kept for history: `Layla/` was completely empty (no files at all, verified via `find`).
   `.intermediate_data/Layla/` (where `DATA_CURATION.md` says the raw source was archived after
   sharding) also doesn't exist anywhere under `/workspace`. The four merged/normalized JSON files
   (`normalized_output_appended.json` etc.) that fed the Layla sharding step were also nowhere on
   this box, and the dataset looked like it needed re-sourcing from scratch.

2. **QASR — partial, possibly fixable.** Only 961 of 3,545 xml transcript files have a matching
   wav in `QASR/alt/arabic-speech-web/mgb2.1/wav/` (that's the *only* wav directory found anywhere
   under `QASR/`). Historical `qasr` row count was ~1.19M vs. our 424K — a ~3x shortfall. The
   original split archives (`QASR/qasr_wav_v1.0.tar.bz2.part_aa` + `part_ab`, 48GB each) are still
   present un-deleted. **The in-progress background check (see above) will tell us whether those
   archives contain more audio than what's already extracted** — if yes, re-extracting fully
   should close this gap; if the archive only has the same 961 files, the shortfall is inherent to
   what was ever downloaded to this box and would need re-downloading from the QASR source (see
   `data.md` / `HANDOFF.md` for the original `wget` URLs, though those had a 2030 SAS-token expiry
   so may need fresh URLs).

## Gotchas discovered (read before re-running any step 6+ script/notebook)

- **Most `scripts/*.py` and `preprocess/*.ipynb` past step 5 hardcode the old VM's absolute path**
  `/home/MohammadNabulsi/whisper/...`, which doesn't exist on this box. Check for it
  (`grep -l "/home/MohammadNabulsi/whisper" scripts/*.py preprocess/*.ipynb`) and patch to either
  `/root/Palestinian-ASR/...` (via the repo symlinks) or directly to
  `/workspace/asr/Palestinian-ASR/...` (bypassing the symlinks) before running.
- **Never point a script's output root at a path that is itself a symlink if the script does
  `shutil.rmtree(output_root)`** (several of the cleaning notebooks do this when
  `OVERWRITE_OUTPUT=True`) — Python's `shutil.rmtree` raises `OSError: Cannot call rmtree on a
  symbolic link` on the top-level path. Fix: point `OUTPUT_ROOT`/`INPUT_ROOT` straight at the real
  `/workspace/asr/Palestinian-ASR/...` path instead of the local repo symlink. This is what steps 4
  and 5 do now (see `.logs/clean_broad_v1.py` / `.logs/clean_targeted_qasr_casablanca_omni_v1.py`).
- **`find <symlinked-top-level-dir> ...` silently under-traverses** in this environment unless you
  pass `-L` (or use `ls`/`du` instead). Cost us a false "empty directory" scare twice. Always use
  `find -L` or `ls -la` when checking these symlinked dataset roots.
- **stdout is fully buffered (not line-buffered) when redirected to a file**, so `nohup python
  script.py > log.txt &` can make a script look stalled when it isn't. Always run with `python -u`
  for any long background job so progress prints show up live.
- **`load average` reported inside this container reflects the shared physical host**, not just
  this container's own usage — a load of 250+ on a "4 CPU" box is normal here and means the shared
  RunPod node is busy, not that something local is broken. Don't chase it as a bug.
- **`filter_masc_c_only.py` and the notebook-derived cleaning scripts have no `--skip-existing` /
  resume support** — they reprocess everything from scratch. Don't kill and restart them for
  parallelism once partway through; you'll redo completed work.
- Local Python env: `/root/Palestinian-ASR/.venv_data` (python3.11, built from
  `requirements-data.txt` + `tqdm`, `ipykernel`, `jupyter_client`). System `/usr/bin/python3` is
  3.8 and unrelated; don't use it. The pre-existing `/workspace/asr/Palestinian-ASR/.venv` has a
  broken numpy install — don't use it either.
- Notebooks were executed by extracting code cells to plain `.py` scripts (via
  `.logs/nb_to_script.py`, which also swaps Jupyter `display(...)` for `print(...)`) rather than
  through a Jupyter kernel — simpler, and gives real-time flushed progress logs. The extracted
  scripts for the two already-run notebooks are at `.logs/clean_broad_v1.py` and
  `.logs/clean_targeted_qasr_casablanca_omni_v1.py`; re-derive similarly for any other notebook
  (`preprocess/fast_asr_data_cleaning_text_only_arrow_parquet_layla.ipynb` is the only one left,
  and it's blocked on Gap 1 anyway).

## How to carry on

1. Check on the QASR archive investigation first (see "actively running" above) — it determines
   whether Gap 2 is fixable before you invest time in later steps that depend on QASR volume.
2. **Step 6** — patch `scripts/merge_cleaned_outputs_and_report.py`'s hardcoded paths (same fix
   pattern as steps 4/5), then run it. Straightforward, no model inference, should be fast.
3. **Steps 7-9** — Omnilingual v2/v3 reclean. Same path-patching needed. Note
   `scripts/reclean_omnilingual_v2.py`'s `INPUT_ROOT` default
   (`.intermediate_data/omnilingual_selected/apc_north_levantine_all_splits`) doesn't exist here —
   repoint it at the top-level `omnilingual_selected/apc_north_levantine_all_splits/` symlink
   instead, which has the same data.
4. **Steps 12-13 (dialect ID)** are the heaviest remaining stage — MarBERTv2 + badrex-mms300m
   model inference over ~1.4M and ~130K rows respectively, no GPU on this box (4 CPUs only). Budget
   real wall-clock time and consider batching/multiprocessing across the 4 cores (this is one place
   where parallelizing is actually justified — CPU-bound model inference, not I/O-contended, unlike
   the earlier steps). Needed inputs (`Runs/.../row_probabilities.jsonl`) are currently missing
   entirely on network storage (only summary/log files survived) — must be regenerated from
   scratch. **Update 2026-07-26: QASR audio is now 100% complete (`QASR/wav_all/`, 3,545/3,545,
   see update note at top of this file) — re-running QASR segmentation
   (`preprocess/qasr_segment_to_arrow.py`) against `wav_all/` before this step will produce
   substantially more QASR rows than the 424K/961-wav or ~1.19M historical count this file cites
   elsewhere; expect the dialect-ID compute budget to grow accordingly.**
5. **Steps 14-15** depend on 12-13's output and the QASR gap resolution — do these last.
6. Keep patching hardcoded `/home/MohammadNabulsi/whisper/` paths as you hit them; no script has
   been "fully" audited past step 6, only fixed reactively per-step so far.
