# HANDOFF — start here

_Last updated: 2026-09-06. This is the only handoff document; the older per-session ones
(`HANDOFF_DATA_PIPELINE.md`, `HANDOVER_LORA_EXPERIMENTS.md`, `FULL_ACOUSTIC_PIPELINE.md`,
`DISCOVERY.md`) were folded in here and deleted. Method detail is in the docs in §9._

## 1. State of play

We are **re-running the LoRA chunk-order experiment from scratch** on a new dataset
(`full_acoustic_split_v2`) selected by acoustic dialect ID over the entire corpus.

The v1 experiments (`forward`, `reverse2`) are finished, their findings are kept (§5),
and their data is **not reproducible** (§8). `reverse_all` was cancelled for that reason.

| Phase | Script | Status |
|---|---|---|
| scan | `scripts/full_acoustic_scan.py` | **done** — 1,347,856 rows, published to R2 |
| score | `scripts/build_full_acoustic_speaker_split.py` | **done** — corrected formula (§3) |
| extract | `scripts/extract_full_acoustic_speaker_split.py` | in progress at time of writing |
| train | `scripts/train_whisper_medium_lora_sequence.py` | queued, starts automatically |

Supervisor: `/workspace/asr_env/full_acoustic_v2/master_orchestrator_v2.sh` (a copy of
`scripts/run_full_acoustic_pipeline.sh` with credentials). Progress in `master.log`; each
phase writes `<phase>.done`, or `<phase>.stuck` if it exhausted its 8 retries and gave up.
A second detached process, `publish_to_r2.sh`, uploads each artifact as its phase lands and
logs to `publish.log`. Both run under `setsid` with `PPID 1` and keep everything on
`/workspace`, so they survive the session that started them.

**To check where things are:**

```bash
ls  /workspace/asr_env/full_acoustic_v2/*.done /workspace/asr_env/full_acoustic_v2/*.stuck
tail -20 /workspace/asr_env/full_acoustic_v2/master.log
tail -20 /workspace/asr_env/full_acoustic_v2/publish.log
```

## 2. The corpus, fully classified

Every QASR and MASC-C row in `curated_corpus/train` was run through
`badrex/mms-300m-arabic-dialect-identifier`. Report: `transfer/acoustic_scan_report.{md,json}`
on R2; raw per-row output in `transfer/dialect_id_scans_full_v2.tar.zst`.

| | |
|---|---|
| Rows scanned | 1,347,856 |
| Classified | 1,325,248 (98.3%) |
| Audio | 1,561.3 h |
| Speakers | 8,565 — QASR 3,545, MASC-C 5,020 |
| Not classified | 22,605 clips under 1s, 3 corrupt, **0 decode errors** |

Dialect distribution: MSA 47.4%, **Levantine 20.3%**, Egyptian 14.3%, Gulf 12.8%,
Maghrebi 5.1%.

QASR's 3,545 speakers match the documented complete 3,545-recording set — independent
confirmation that coverage is total.

### The `lev` / `non_lev` folders are not a speaker-level judgment

This is the most consequential finding, and it is why scanning everything mattered:

| Dominant leaf | Speakers | Max score | Median |
|---|---|---|---|
| `masc_lev` | 46 | 0.9297 | 0.7667 |
| `masc_non_lev` | 4,974 | 0.8312 | 0.1246 |
| `qasr_lev` | **1** | 0.7972 | 0.7972 |
| `qasr_non_lev` | 3,544 | 0.7324 | 0.1411 |

The `lev` folders hold **47 of 8,565 speakers**. The old routing was per *row*, by a text
classifier, so one recording's utterances were scattered across both folders. The scan
finds 269,563 acoustically Levantine clips, the overwhelming majority inside `non_lev`.
**194.4 of the 200 selected training hours come from `non_lev`.** Restricting to the `lev`
folders would have capped the whole study at ~35 hours.

Every lev-dominant speaker was selected (46/46 and 1/1, all above the train cutoff), so
nothing genuinely Levantine was dropped. The `lev`-folder hours left behind (18% of
`masc_lev`, 42% of `qasr_lev`) are orphan fragments of speakers who are overwhelmingly
`non_lev` and score poorly overall.

## 3. The v2 selection rule

Each speaker (QASR `recording_id`, MASC-C `video_id`) is scored

```
score = 0.3 * mean(text LEV)  +  0.7 * mean(acoustic Levantine)
```

a **weighted arithmetic average** — not the product v1 used. All four leaves are pooled
into one ranking; a speaker's rows are aggregated across leaves before scoring. Speakers
sort by score and are cut into consecutive, speaker-disjoint blocks: highest **8h test**,
next **8h val**, next **200h train**. Selection is per speaker — once chosen, *all* of that
speaker's samples are extracted, including individually weak ones. A speaker with no usable
acoustic result is never selected.

| Block | Hours | Speakers | Score range | Mean text | Mean acoustic |
|---|---|---|---|---|---|
| test | 8.37 | 170 | 0.9297 → 0.6304 | 0.5816 | 0.7815 |
| val | 8.01 | 64 | 0.6279 → 0.5769 | 0.3946 | 0.6889 |
| train | 200.00 | 918 | 0.5767 → 0.3368 | 0.2744 | 0.5054 |

Verified: 0 speakers appear in more than one split, and the blocks are strictly consecutive
(`min(test) ≥ max(val) ≥ max(train)`).

`train_full_rank.json` gives each train speaker's cumulative hours, which cuts the train
block into the four ~50h confidence-ranked chunks (chunk 1 = most confident … chunk 4 = least).

### ⚠ The bug this replaced — read before trusting any earlier v2 numbers

The first v2 split was built with the text term silently **zeroed**, making it a pure
acoustic ranking mislabelled as a blend. Two independent faults:

1. The text scan stores probabilities as `predictions: [{label, score}]`; the script read
   `label_scores` (a dict), which that file does not have → every text score read `0.0`.
2. The text scan's `uid` is the long form (`…parquet:uid=qasr:GUID:…`), but the key matcher
   only handled the short form → QASR text rows dropped entirely, MASC rows keyed by filename.

Text/audio speaker join was **0 of 8,565** and nothing errored — the ranking still looked
plausible. Both are fixed (join is now 8,561/8,565) and the script **aborts** if fewer than
half the audio speakers join to a text score. Any split with `text_mean == 0`, a max score
near 0.697, or 132/87/927 speakers is the broken one; the superseded copy is kept at
`/workspace/asr_env/full_acoustic_v2/split_acoustic_only_superseded/`.

### A design consequence to be aware of

Because test takes the highest band, **test and val are systematically more confidently
Levantine than train** (test 0.63–0.93 vs train 0.34–0.58, no overlap). That is deliberate —
evaluate on the cleanest Levantine available — but train and test are not drawn from the
same distribution, so WER here is not comparable to a random-split baseline, and a reviewer
will ask. Alternatives if that becomes a problem: carve test/val from the middle of the
ranking, or stratify them across score bands.

## 4. Which training script to use

**Use `scripts/train_whisper_medium_lora_sequence.py` for every run.**

`train_whisper_medium_lora_progressive.py` is *not* dead code — it holds the shared core
(dataset, chunking, train loop, eval, checkpointing) the sequence script imports — but it
hardcodes the forward order and cannot inherit a checkpoint. The sequence script does
everything it does:

```bash
python3 scripts/train_whisper_medium_lora_sequence.py \
  --sequence 1,2,3,4,1,2,3,4 --run-name forward_v2 \
  --train-parquet .../data/train.parquet --rank-meta .../split/train_full_rank.json \
  --val-parquet .../data/val.parquet --test-parquet .../data/test.parquet \
  --out-dir .../whisper_medium_lora_forward_v2 --lr 1e-4
```

LoRA r=32, α=32, dropout 0.05, targets q/k/v/o/fc1/fc2; batch 8; one continuous
`OneCycleLR` at `max_lr=1e-4`, warmup 0.1, spanning **all** stages.

### Continuing a run at the LR it left off at

`OneCycleLR` spans the whole run, so the LR at stage 5 depends on how many stages precede
it. Starting a continuation fresh at `--lr 1e-4` restarts the cycle and re-warms up,
discarding the decay the earlier stages earned.

`--init-from <stage checkpoint dir>` loads that checkpoint's adapter **and optimizer state**;
`--init-stage-offset N` declares the checkpoint already covers the first `N` stages of
`--sequence`, and the scheduler is fast-forwarded by exactly those stages' step count, so
stage `N+1` begins at the LR this run's own cycle prescribes:

```bash
python3 scripts/train_whisper_medium_lora_sequence.py \
  --sequence 4,3,2,1 --run-name reverse_all \
  --init-from outputs/whisper_medium_lora_reverse2/checkpoints/s2_c3 \
  --init-stage-offset 2 ...
```

Pass the **full** intended sequence, not just the remaining stages — the schedule is computed
over the whole thing. A plain re-run of an interrupted command needs neither flag: the
resume state restores weights, optimizer, scheduler and RNG, so a crash-restart never resets
the LR either.

## 5. v1 findings, and why v2 exists

**forward** — chunks 1,2,3,4,1,2,3,4 (`outputs/whisper_medium_lora_progressive/all_results.json`):

| stage | val WER | test WER |
|---|---|---|
| ep1_h50 | 49.80% | 42.90% |
| ep1_h100 | 49.13% | 41.59% |
| ep1_h200 | 49.68% | 42.48% |
| **ep2_h50** | **44.67%** | **38.52%** |
| ep2_h100 | 45.52% | 38.60% |
| ep2_h200 | 45.72% | 39.11% |

**reverse2** — chunks 4,3,4,3, the two *worst* chunks only
(`outputs/whisper_medium_lora_reverse2/all_results.json`): best stage `s4_c3` reaches
49.26% val / 41.76% test.

1. **More data made it worse.** The best checkpoint of the whole v1 study is the *first* 50h
   chunk of epoch 2; adding chunks 3–4 costs ~1 point of WER in both epochs.
2. **It is quality, not quantity.** Training only on the worst two chunks is ~4.6 points
   worse than best-first, so the ranking tracks something real.

Hence v2: rank every speaker acoustically over the whole corpus rather than by text over a
pre-filtered candidate set, and see whether a better ranking raises the ceiling.
**The bar to beat is 44.67% val / 38.52% test.**

## 6. Where everything lives

**R2** — bucket `backup`, prefix `transfer/` (~308 GiB). Full layout in `MANIFEST.md` there
and in `R2_BUCKET_USAGE.md` here.

| Path | What |
|---|---|
| `curated_corpus/` | the corpus, source of truth (222 GB) |
| `data/full_acoustic_split_v2/` | **the v2 dataset**: `speaker_assignments.json`, `train_full_rank.json`, `{train,val,test}.parquet` |
| `acoustic_scan_report.{md,json}` | the classification report |
| `dialect_id_scans_full_v2.tar.zst` | raw per-row acoustic scan (128 MB) |
| `dialect_id_scans.tar.zst` | the text scans (MarBERTv2) + earlier audio bands |
| `data/data_lev_custom_split_v1/` | v1 derived splits — closest surviving copy of v1 training data |
| `data/speaker_disjoint_split_v1/` | v1 split definition |
| `adapters/lora_speaker_disjoint_2026-09-04/` | v1 forward + reverse2 adapters and benchmarks |
| `adapters/lora_full_acoustic_v2/` | v2 adapters (published when training finishes) |
| `adapters/whisper/`, `adapters/FINAL_200h/` | ⚠ weights pruned 2026-09-06; all metrics/configs/READMEs kept in place |
| `HANDOFF.md` | a copy of this file |

**GitHub** — `Mohammad-Nabulsi/Palestinian-ASR`, branch
`omni_tune-speaker-disjoint-analysis`, PR #1 into `main`. Code, docs and all retained
metrics (`outputs/`, `generalization_results/`). No credentials are committed.

## 7. Gotchas that have already cost real time

- **QASR audio is raw headerless PCM16**, real rate in a sibling `sampling_rate` column.
  `soundfile.read()` fails on ~100% of QASR rows, or mis-decodes to a length that OOMs the
  GPU. MASC's audio *is* self-describing and must keep going through `sf.read()`. Both the
  scan and the extractor branch on this; the extractor re-encodes QASR to real WAV so the
  training loader sees one uniform format. See `scripts/zero_shot_eval/audio_io.py`.
- **Never stage pipeline state in a session scratchpad.** `/tmp/claude-*/**/scratchpad` is
  deleted with its session — that is how the v1 dataset was lost (§8). Durable state goes on
  `/workspace` (MooseFS network volume) or R2.
- **A silent join failure looks like a working pipeline.** §3's bug produced a plausible
  ranking with no error. Assert on join cardinality, not just on schema.
- **The scan ledger commits whole shards; rows append individually.** A mid-shard kill leaves
  rows the ledger calls pending, which would be re-appended on resume and double-count
  speaker hours. The scan drops uncommitted-shard rows at startup — but only rows that *name*
  an uncommitted shard; rows with no `source_file` are kept, because treating
  "unattributable" as "delete" once cost a full masc re-scan.
- **A GPU OOM in the scan is usually bad audio, not fragmentation.** A single mis-decoded
  multi-minute "sample" can exceed capacity alone, so lowering batch size never fixes it.
  There is a duration cross-check that drops those rows.

## 8. What was lost, and why it cannot be recovered

The v1 speaker-disjoint dataset was staged only in a session scratchpad, which was deleted
with its session. Recovery from R2 got most of it back, but roughly **20–30% of the masc_c
training audio is in no R2 object and is gone for good**. So the v1 split *definition* is
intact and reproduces the documented 200.01h exactly; the v1 *audio* is not fully
reproducible, so v1 numbers can be cited but not re-run. That is why `reverse_all` was
cancelled rather than deferred.

## 9. Reference docs

| Doc | Covers |
|---|---|
| `README.md` | repo entry point and map |
| `PIPELINE.md` | the corpus-building pipeline (`pipeline/`, stages, CLI) |
| `DATA_CURATION.md` | per-source provenance, cleaning, normalization decisions |
| `SPEAKER_DISJOINT_SELECTION.md` | why splits are speaker-disjoint; how v1 was built |
| `R2_BUCKET_USAGE.md` | bucket layout and how to connect |
| `PAPER_DRAFT.md`, `PRETRAINING_STUDY_REPORT.md`, `scratch_full_model_report.md` | the earlier studies |
| `outputs/`, `generalization_results/` | retained metrics for every run, including ones whose weights were pruned |
