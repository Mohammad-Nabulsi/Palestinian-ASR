# Palestinian / Levantine Arabic ASR

Fine-tuning `openai/whisper-medium` with LoRA on a curated Levantine Arabic corpus, and
studying how the **dialect quality of the training data** — not just its quantity —
moves WER.

**→ Start at [HANDOFF.md](HANDOFF.md).** It is the single start point: where the project
is right now, what is running, the results so far, and the mistakes that have already
cost real time. Everything below is just a map.

## The question

Training data is ranked by how confidently Levantine each *speaker* is, then cut into
four ~50h chunks (chunk 1 = most confident … chunk 4 = least). Feeding those chunks in
different orders separates "more data" from "better data" as causes of a WER change.

The v1 answer was that **more data made it worse**: the best checkpoint of the whole run
was the *first* 50h chunk of epoch 2 (44.67% val / 38.52% test WER), and adding chunks
3–4 cost about a point. Training on only the worst two chunks was ~4.6 points worse
again — so the ranking tracks something real. That is why the current work re-ranks every
speaker using **acoustic** dialect ID over the entire corpus rather than text alone, and
re-runs the experiment from scratch. See [HANDOFF.md](HANDOFF.md) §2–§3.

## Layout

| Path | What |
|---|---|
| [HANDOFF.md](HANDOFF.md) | **Read first.** Current state, results, gotchas, what was lost |
| [pipeline/](pipeline/) | The corpus-building pipeline — one config-driven runner. Stages: `ingest → clean → assemble → dialect → speaker_select → split` |
| [PIPELINE.md](PIPELINE.md) | How that pipeline works and which gotchas its design closes |
| [DATA_CURATION.md](DATA_CURATION.md) | Per-source provenance, cleaning and normalization decisions |
| [dialect_identifiaction/](dialect_identifiaction/) | Dialect-ID scans: MarBERTv2 (text) and Badrex MMS-300m (acoustic) |
| [SPEAKER_DISJOINT_SELECTION.md](SPEAKER_DISJOINT_SELECTION.md) | Why splits are speaker-disjoint and how the v1 split was built |
| [scripts/](scripts/) | Corpus build, the full-acoustic v2 pipeline, training, zero-shot eval |
| [asr_milestone7/](asr_milestone7/) | Model-agnostic training/eval harness used by the earlier multi-model study |
| [outputs/](outputs/), [generalization_results/](generalization_results/) | Retained metrics for every run — including ones whose weights are gone |
| [R2_BUCKET_USAGE.md](R2_BUCKET_USAGE.md) | Bucket layout and how to connect (no credentials in this repo) |
| [PAPER_DRAFT.md](PAPER_DRAFT.md), [PRETRAINING_STUDY_REPORT.md](PRETRAINING_STUDY_REPORT.md) | Write-ups of the earlier studies |

## Running things

```bash
pip install -r requirements.txt          # training / eval
pip install -r requirements-data.txt     # data pipeline

# corpus build
./tools/smoke.sh                                     # end-to-end on tiny fixtures
python -m pipeline run --config configs/full.yaml

# the current experiment: acoustic scan -> speaker split -> extract -> train
bash scripts/run_full_acoustic_pipeline.sh
```

Training always goes through
[scripts/train_whisper_medium_lora_sequence.py](scripts/train_whisper_medium_lora_sequence.py).
It takes an arbitrary chunk order, and it is the only entry point that can inherit a
checkpoint and resume the LR schedule where that checkpoint left off instead of
restarting the cycle — see [HANDOFF.md](HANDOFF.md) §4 before writing a continuation
command. (`train_whisper_medium_lora_progressive.py` is not run directly; it holds the
shared training core the sequence script imports.)

## Data

The corpus and every derived split live in Cloudflare R2 (bucket `backup`, prefix
`transfer/`), not in git — see [R2_BUCKET_USAGE.md](R2_BUCKET_USAGE.md) and the bucket's
own `MANIFEST.md`. Credentials are never committed; export them as environment
variables.

Two things will bite you if you read the corpus directly:

- **QASR audio is raw headerless PCM16**, with the real rate in a sibling
  `sampling_rate` column. `soundfile.read()` fails on essentially every QASR row. MASC's
  audio *is* self-describing. Use `scripts/zero_shot_eval/audio_io.py`.
- **Never stage pipeline state in a session scratchpad** (`/tmp/claude-*/**/scratchpad`).
  It is deleted with its session — that is how the v1 dataset was lost. Durable state
  goes in `/workspace` or R2.
