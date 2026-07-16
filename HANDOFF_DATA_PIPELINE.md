# HANDOFF — Data prep pipeline run (this session)

_Last updated: 2026-07-16 18:40 UTC. Read this first if picking this up cold._

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
| 6 | Merge cleaned outputs (`scripts/merge_cleaned_outputs_and_report.py`) | ⏳ not started | → `data_cleaned_text_merged_v1/` |
| 7 | Omnilingual reclean v2 (`scripts/reclean_omnilingual_v2.py`) | ⏳ not started | → `data_cleaned_text_omnilingual_v2/` |
| 8 | Omnilingual recovery v3 (`scripts/recover_omnilingual_token_span_rows_v3.py`) | ⏳ not started | → `data_cleaned_text_omnilingual_v3_recovered_from_v2/` |
| 9 | Rebuild `data/` with final Omnilingual rows (`scripts/create_data_with_final_omnilingual.py`) | ⏳ not started | updates `data/` |
| 10-11 | Layla normalize/shard + flatten `data/clean/` | ❌ blocked | Gap 1 — raw Layla source is gone |
| 12 | Text dialect ID, MarBERTv2 | ⏳ not started | → `Runs/text_dialect_scan_.../row_probabilities.jsonl` (heavy, CPU-only box) |
| 13 | Audio dialect ID, badrex mms300m | ⏳ not started | → `Runs/dialect_scan_badrex.../row_probabilities.jsonl` (heavy) |
| 14 | Build Levant/non-Levant binary split | ⏳ not started | → `data_curated_levant_binary_v1/` |
| 15 | QASR audio-decode repair + rebuild | ⏳ not started | → `data_curated_levant_binary_v2_qasr_audio_fix/` |

### Step 5 per-dataset breakdown (from `data_cleaned_text_qasr_casablanca_omni_v1/reports/cleaning_report.json`)

| dataset | total | kept | historical (DATA_CURATION.md) |
|---|---|---|---|
| qasr | 424,086 | 399,633 | 1,194,234 — **short, see Gap 2** |
| casablanca_jordanian | 1,696 | 1,695 | 1,696 — matches exactly |
| casablanca_palestinian | 1,334 | 1,328 | 1,334 — matches exactly |
| apc_Arab (omnilingual) | 517 | 43 | 517 — matches exactly (kept count differs by design, v1 fast-pass vs v2/v3 recovery logic) |

## Known gaps (data genuinely missing/short on this network storage copy)

1. **Layla — total loss.** `Layla/` is completely empty (no files at all, verified via `find`).
   `.intermediate_data/Layla/` (where `DATA_CURATION.md` says the raw source was archived after
   sharding) also doesn't exist anywhere under `/workspace`. The four merged/normalized JSON files
   (`normalized_output_appended.json` etc.) that fed the Layla sharding step are also nowhere on
   this box. **This dataset needs to be re-sourced from wherever it was originally downloaded** (a
   local machine, per `data.md`) — it cannot be reconstructed from anything present here.

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
   scratch.
5. **Steps 14-15** depend on 12-13's output and the QASR gap resolution — do these last.
6. Keep patching hardcoded `/home/MohammadNabulsi/whisper/` paths as you hit them; no script has
   been "fully" audited past step 6, only fixed reactively per-step so far.
