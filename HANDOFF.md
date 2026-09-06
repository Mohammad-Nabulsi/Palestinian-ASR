# HANDOFF — start here

_Last updated: 2026-09-06. This is the only handoff document in the repo; the older
per-session ones (`HANDOFF_DATA_PIPELINE.md`, `HANDOVER_LORA_EXPERIMENTS.md`,
`FULL_ACOUSTIC_PIPELINE.md`, `DISCOVERY.md`) were folded in here and deleted. Method
detail still lives in the reference docs listed in §7._

## 1. Where the project is

We are **re-running the LoRA chunk-order experiment from scratch** on a new dataset.

The v1 experiments (`forward`, `reverse2`) are finished and their findings are kept
(§3), but their training data can no longer be rebuilt exactly (§6), and the speaker
selection behind it only ever saw **text** dialect scores. The replacement selects
speakers using **acoustic** dialect ID over the *entire* corpus, weighted with text.

In flight right now: `/workspace/asr_env/full_acoustic_v2/master_orchestrator_v2.sh`,
a four-phase resumable chain. Progress is in `master.log`; each phase drops a
`<phase>.done` marker, or `<phase>.stuck` if it exhausted its retries.

| # | Phase | Script | Output |
|---|---|---|---|
| 1 | `scan` | `scripts/full_acoustic_scan.py` | `acoustic_full_scan/{masc,qasr}_{lev,non_lev}/row_probabilities.jsonl` |
| 2 | `score` | `scripts/build_full_acoustic_speaker_split.py` | `split/speaker_assignments.json`, `split/train_full_rank.json` |
| 3 | `extract` | `scripts/extract_full_acoustic_speaker_split.py` | `data/{train,val,test}.parquet` |
| 4 | `train_v2` | `scripts/train_whisper_medium_lora_sequence.py` | `whisper_medium_lora_forward_v2/` |

## 2. The v2 selection rule

Every QASR and MASC row in `curated_corpus` is scored by the Badrex acoustic dialect
model. Each speaker (QASR `recording_id`, MASC-C `video_id`) gets

```
score = 0.3 * mean(text LEV) + 0.7 * mean(acoustic Levantine)
```

Speakers are sorted by that score and cut into **speaker-disjoint** blocks: the
highest 8h → **test**, the next 8h → **val**, the next 200h → **train**. Selection is
per *speaker*, not per row — once a speaker is chosen, **all** of its samples are
extracted, including ones with a weak individual score. A speaker with no usable
acoustic result is never selected.

`train_full_rank.json` records each train speaker's cumulative hours, which is what
splits the train block into the four ~50h confidence-ranked chunks (chunk 1 = most
Levantine-confident … chunk 4 = least).

## 3. v1 findings worth keeping

Whisper-medium + LoRA (r=32, α=32, dropout 0.05, q/k/v/o/fc1/fc2), batch 8, one
continuous `OneCycleLR` at `max_lr=1e-4`, warmup 0.1.

**forward** — chunks 1,2,3,4,1,2,3,4 (`outputs/whisper_medium_lora_progressive/all_results.json`):

| stage | val WER | test WER |
|---|---|---|
| ep1_h50 | 49.80% | 42.90% |
| ep1_h100 | 49.13% | 41.59% |
| ep1_h200 | 49.68% | 42.48% |
| **ep2_h50** | **44.67%** | **38.52%** |
| ep2_h100 | 45.52% | 38.60% |
| ep2_h200 | 45.72% | 39.11% |

**reverse2** — chunks 4,3,4,3, i.e. the two *worst* chunks only
(`outputs/whisper_medium_lora_reverse2/all_results.json`): best stage `s4_c3` reaches
49.26% val / 41.76% test.

Two things follow, and they are the reason for the v2 rebuild:

1. **More data made it worse.** The best checkpoint in the whole v1 study is the
   *first* 50h chunk of epoch 2. Adding chunks 3–4 costs ~1 point of WER rather than
   helping, in both epochs.
2. **It is quality, not quantity.** Training on only the worst two chunks
   (reverse2, 100h) is ~4.6 points worse than the best-first schedule. So the
   ranking is tracking something real, and a better ranking should raise the ceiling —
   hence scoring every speaker acoustically instead of by text alone.

`reverse_all` (chunks 4,3,2,1) was designed as the third arm and **cancelled**: it
would have had to train on the v1 dataset, which is no longer reconstructable (§6).

## 4. Which training script to use

**Use `scripts/train_whisper_medium_lora_sequence.py` for every new run.**

`train_whisper_medium_lora_progressive.py` is *not* dead code — it holds the shared
core (dataset, chunking, train loop, eval, checkpointing) that the sequence script
imports — but it hardcodes the forward order and has no way to inherit a checkpoint.
Everything it can do, the sequence script does:

```bash
python3 scripts/train_whisper_medium_lora_sequence.py \
  --sequence 1,2,3,4,1,2,3,4 --run-name forward_v2 \
  --train-parquet .../data/train.parquet --rank-meta .../split/train_full_rank.json \
  --val-parquet .../data/val.parquet --test-parquet .../data/test.parquet \
  --out-dir .../whisper_medium_lora_forward_v2 --lr 1e-4
```

### Continuing a run at the LR it left off at

This is the part that is easy to get wrong. `OneCycleLR` spans the **whole** run, so
the LR at stage 5 depends on how many stages precede it. Starting a continuation
fresh at `--lr 1e-4` would restart the cycle and re-warm-up, wiping out the decay the
earlier stages already earned.

`--init-from <stage checkpoint dir>` loads that checkpoint's adapter **and optimizer
state**, and `--init-stage-offset N` declares that the checkpoint already covers the
first `N` stages of `--sequence`. The script then fast-forwards the scheduler by
exactly those stages' step count, so stage `N+1` starts at the LR this run's own
cycle prescribes for that position — not at the initial `1e-4`:

```bash
# continue for another 50h from a run that already covers 2 stages
python3 scripts/train_whisper_medium_lora_sequence.py \
  --sequence 4,3,2,1 --run-name reverse_all \
  --init-from outputs/whisper_medium_lora_reverse2/checkpoints/s2_c3 \
  --init-stage-offset 2 ...
```

Pass the **full** intended sequence, not just the remaining stages — the schedule is
computed over the whole thing. A plain re-run of an interrupted command needs neither
flag: resume state (`resume_state/`) restores weights, optimizer, scheduler and RNG,
so a crash-restart never resets the LR either.

## 5. Gotchas that have already cost real time

- **QASR audio is raw headerless PCM16.** Its `audio` struct is *not* self-describing;
  the true rate is in a sibling `sampling_rate` column. Calling `soundfile.read()` on
  it fails on ~100% of rows ("Format not recognised") or mis-decodes to an absurd
  length that OOMs the GPU. MASC's audio *is* self-describing and must keep going
  through `sf.read()`. Both `full_acoustic_scan.py` and
  `extract_full_acoustic_speaker_split.py` branch on this; the extractor re-encodes
  QASR to real WAV so the training loader sees one uniform format.
  See also `scripts/zero_shot_eval/audio_io.py`.
- **Never put pipeline state in the session scratchpad.** `/tmp/claude-0/**/scratchpad`
  is deleted when its session ends. That is what destroyed the v1 dataset (§6).
  Durable state belongs in `/workspace` (MooseFS network volume) or R2.
- **The scan's ledger commits whole shards, the JSONL appends per row.** A kill
  part-way through a shard therefore leaves rows the ledger calls pending, which
  would be re-appended (double-counting speaker hours) on resume. The scan drops
  uncommitted-shard rows at startup — but only rows that *name* an uncommitted shard;
  rows with no `source_file` are left alone, because treating "unattributable" as
  "delete" once cost a full masc re-scan.
- **A GPU OOM in the scan is usually bad audio, not fragmentation.** Both happen, but
  a single mis-decoded multi-minute "sample" can exceed capacity on its own, so
  lowering the batch size never fixes it. There is a duration cross-check that drops
  those rows.

## 6. What was lost, and why it cannot be recovered

The v1 speaker-disjoint dataset was staged only in a session scratchpad, which was
deleted with its session. Recovery from R2 got most of it back
(`data_lev_custom_split_v1` + `curated_corpus`), but roughly **20–30% of the masc_c
training audio is gone for good** — it is in no R2 object. So:

- the v1 split *definition* is intact (`speaker_disjoint_split_v1/speaker_assignments.json`,
  and it reproduces the documented 200.01h exactly);
- the v1 *audio* is not fully reproducible, so v1 numbers can be cited but not re-run;
- this is why `reverse_all` was cancelled rather than deferred.

## 7. Reference docs

| Doc | Covers |
|---|---|
| `PIPELINE.md` | the corpus-building pipeline (`pipeline/`, stages, CLI) |
| `DATA_CURATION.md` | per-source provenance, cleaning, normalization decisions |
| `SPEAKER_DISJOINT_SELECTION.md` | why the split is speaker-disjoint and how v1 was built |
| `R2_BUCKET_USAGE.md` | bucket layout and how to connect (no credentials in repo) |
| `PAPER_DRAFT.md`, `PRETRAINING_STUDY_REPORT.md`, `scratch_full_model_report.md` | the earlier multi-model / pretraining studies |
| `generalization_results/`, `outputs/` | the retained metrics for every run, including ones whose weights were pruned from R2 |
