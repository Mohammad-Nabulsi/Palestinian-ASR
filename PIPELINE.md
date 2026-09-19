# Unified Data Pipeline

One config-driven runner replacing the per-dataset scripts and notebooks catalogued in
[DATA_CURATION.md](DATA_CURATION.md). Same rules, same outputs — but the logic exists
once instead of six times.

```bash
./tools/smoke.sh                                          # end-to-end on tiny fixtures
python -m pipeline run --config configs/full.yaml          # production
python -m pipeline run --config configs/full.yaml --only clean --dry-run
python -m pipeline list --config configs/full.yaml
```

## Why

The audit that motivated this found the same code copied with only paths changed:

| Duplicated logic | Copies | Where |
|---|---|---|
| Fast text cleaning (~680 lines) | 6 | 3 × `.logs/clean_*.py`, 3 × `preprocess/fast_asr_*.ipynb`, plus a notebook *generator* |
| `normalize_arabic_transcript` | 6 | verbatim in each cleaner + omni v2/v3 + recovery |
| Omnilingual reclean | 2 | v2 vs v3 differ only in strip strategy and a column suffix |
| Levant binary split | 3 | 1 implementation + 2 subprocess wrappers |
| Assemble a data root | 4 | merge, replace-omni, create-data-final, finalize-layla |

A `diff` of the three cleaning scripts returns only `INPUT_ROOT`, `OUTPUT_ROOT` and the
discovery globs. Those are now config values.

> **Where the superseded scripts went.** The `.logs/clean_*.py` cleaners and the
> `preprocess/fast_asr_*.ipynb` notebooks named in this table and in "Replaces" below
> were removed from the working tree in the 2026-09-06 cleanup, since this pipeline is
> the maintained implementation of all of them. They are still in git history — e.g.
> `git show HEAD~1:.logs/clean_broad_v1.py` — and are listed here as a record of what
> this design absorbed, not as files you should expect to find.

## Stages

Six stage types, composed in any order by a config. A config entry has an `id` (unique,
used for `--only`/`--skip` and report paths) and a `stage` (the type below).

| Stage | Does | Replaces |
|---|---|---|
| `ingest` | raw sources → shard trees, via per-source adapters | `qasr_segment_to_arrow.py`, `filter_masc_c_only.py`, `stage_raw_datasets.py`, the sharding half of `finalize_data_with_layla.py` |
| `clean` | drop rules + Arabic normalization, per-dataset globs | `.logs/clean_broad_v1.py`, `.logs/clean_targeted_*.py`, `.logs/clean_qasr_part2.py`, the 3 notebooks, `build_qasr_casablanca_omnilingual_cleaning_notebook.py`, `reclean_omnilingual_v2.py`, `reclean_omnilingual_v3.py`, `recover_omnilingual_token_span_rows_v3.py` |
| `assemble` | many cleaned trees → one curated root, with replacement | `merge_cleaned_outputs_and_report.py`, `replace_merged_omnilingual_with_recovered.py`, `create_data_with_final_omnilingual.py`, the manual "flatten `clean/`" step |
| `dialect` | per-row dialect ID → `row_probabilities.jsonl` | `arabic_text_dialect_scan_marbertv2_written.py`, `arabic_dialect_scan_badrex_mms300m.py` |
| `speaker_select` | group both dialect passes per speaker, rank, cut val/test/train blocks → `speaker_assignments.json` | (new; supersedes the short-lived `speaker_aggregate`) |
| `split` | binary Levantine routing + speaker-disjoint train/val/test | `create_levant_non_levant_splits.py`, `rebuild_qasr_only_levant_binary.py`, `repair_qasr_audio_and_rebuild_levant_binary.py` |

Shared code lives in [pipeline/textnorm.py](pipeline/textnorm.py) (normalization,
detectors, pre-check strategies) and [pipeline/shards.py](pipeline/shards.py)
(parquet/arrow/jsonl readers, shard writers, `stable_file_id`).

### How the per-dataset differences survive

Real differences became options rather than forks:

- **`ingest` keeps adapters.** This is the one stage where sources genuinely differ
  (XML+WAV segmentation vs. a parquet column filter vs. audio/transcript pairing), so it
  is an adapter registry. `qasr_xml_wav` calls `preprocess/qasr_segment_to_arrow.py`
  in-process rather than reimplementing its 745 lines of XML/PCM handling.
- **`clean` takes a `precheck` list.** `[]` is the historical fast pass;
  `[placeholders]` is the Omnilingual v2 reclean; `[spans, placeholders]` is v3. The
  drop rules never change — only what the text looks like when they run.
- **`clean` takes `mode: recover`.** Same rules re-applied to rows a previous pass
  dropped, writing `recovered_clean/` and `still_dropped/` so the two generations of
  shards stay distinguishable.
- **`column_suffix`** reproduces the historical `_v2`/`_v3` column names.
- **`dialect` takes a `backend`.** Text and audio were never the same model, but they
  were the same loop.

## Data flow

```
raw sources ──ingest──▶ {work}/ingested ──clean──▶ {work}/cleaned/{clean,dropped}
                                                          │
                            ┌─────────────────────────────┘
                            ▼
             clean(mode: recover) ──▶ {work}/recovered/{recovered_clean,still_dropped}
                            │
                            ▼
                       assemble ──▶ {work}/curated   (flat shard root)
                            │
                 ┌──────────┴──────────┐
                 ▼                     ▼
          dialect(text)  ────▶  dialect(audio)      candidates_from: text scores
                 │                     │
                 └──────────┬──────────┘
                            ▼
                    speaker_select ──▶ speaker_assignments.json
                            │           (needs BOTH verdicts, so it runs
                            ▼            after the audio pass, not between)
                          split ──▶ {split}/{leaf}/{lev|non_lev}/
```

## Configs

`vars` are `str.format`-substituted into every string in the document, and `repo_root`
is always available.

- **[configs/sample.yaml](configs/sample.yaml)** — tiny end-to-end smoke run. Uses the
  offline `hash_stub` dialect backend, so it needs no GPU, no network and no model cache.
- **[configs/full.yaml](configs/full.yaml)** — production paths on this box, real models.

### Raw-source availability (production config)

| Source | Raw on disk | Handling |
|---|---|---|
| QASR | yes — `wav_all/`, the complete 3,545-recording set | ingested |
| Omnilingual APC | yes | ingested |
| Layla | yes (874 files; 218 `.txt` + 218 `.docx`) | ingested |
| MASC | **deleted post-merge** (~173GB) | re-enters at `assemble` from `data_cleaned_text_merged_v1/clean/` |
| Casablanca | **deleted post-merge** (~8.6GB) | re-enters at `assemble` from `data_cleaned_text_merged_v1/clean/` |

The two deleted sources keep disabled `ingest`/`clean` entries, so restoring raw data is
a one-word change (`enabled: false` → `true`).

Layla no longer blocks on the four missing `normalized_*.json` files (HANDOFF Gap 1) —
the `audio_text_pairs` adapter reads transcripts from the sibling `.txt`/`.docx`, and
layers the normalized JSONs on top when they are present.

## Testing

```bash
./tools/smoke.sh                        # fixtures + full run + 15 assertions
python tests/test_textnorm.py           # normalization pinned to the original scripts
python tests/verify_run.py --run-root .sample_run
```

`tools/make_sample_data.py` carves genuinely small slices of the *real* sources where
they still exist (2 QASR recordings with XML trimmed to a 60s window, 8 Omnilingual rows,
3 Layla files truncated to 3s) and synthesizes fixtures for the deleted ones. Fixture
rows deliberately include English, digits and sub-0.5s durations so every drop rule
fires, plus two crafted Omnilingual rows that exercise the v2-vs-v3 recovery difference.

Last verified run (all 7 stage instances `ok`, 15/15 assertions passed):

| | rows |
|---|---|
| clean input | 120 |
| kept | 40 |
| dropped (english / too-short / number) | 44 / 20 / 16 |
| recovered by v3 span-strip | 1 of 2 |
| curated | 41 |
| split (train / val / test) | 27 / 7 / 7 |

## Gotchas this design closes

Each of these bit the earlier ad-hoc scripts this pipeline replaced:

- **Hardcoded `/home/MohammadNabulsi/whisper/` paths.** Gone — all paths come from config `vars`.
- **`shutil.rmtree` on a symlinked output root.** Every stage that overwrites now raises a
  clear error naming the symlink instead of crashing mid-run.
- **No resume support.** `dialect` has `resume: true` (keyed on `source_file` + `row_idx`);
  `--only`/`--skip` restart at any stage boundary.
- **Buffered stdout hiding progress.** The runner flushes every log line and mirrors it to
  `{run_root}/reports/run.log`.
- **Speaker leakage across splits.** Split assignment used to be a hash of
  `(seed, leaf, source_file, row_idx)`: stable and reproducible per row, but blind to who
  was speaking, so ~99–100% of val's and test's qasr recordings also had rows in train.
  `speaker_select` now assigns whole speakers and `split` follows that assignment; the
  hash path survives only for leaves with no speaker key (omni, layla, casa/*). See
  [SPEAKER_DISJOINT_SELECTION.md](SPEAKER_DISJOINT_SELECTION.md).

## A bug this consolidation surfaced

Copying the normalizer by hand corrupted `DIACRITICS_RE`: bidirectional text rendering
reordered the codepoints in the character class, turning the ranges into
`[U+0610-U+064B …]`, which covers the entire Arabic letter block. Every transcript
normalized to the empty string — silently, no exception, and downstream stages happily
scored empty text.

`tests/test_textnorm.py` now pins all four Arabic regexes codepoint-for-codepoint against
`scripts/reclean_omnilingual_v2.py`, asserts no Arabic letter can match the diacritics
class, and `tests/verify_run.py` fails the run if any normalized transcript comes out
blank. **Do not retype the Arabic literals in `pipeline/textnorm.py` by hand.**
