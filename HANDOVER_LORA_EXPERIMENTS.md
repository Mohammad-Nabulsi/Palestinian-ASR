# HANDOVER — LoRA speaker-disjoint chunk-order experiments

_Written 2026-09-04, ~21:00 UTC. Start point for the next session._

## What this is

Three LoRA fine-tunes of `openai/whisper-medium` on the speaker-disjoint 200h train
block (see `SPEAKER_DISJOINT_SELECTION.md`), each walking the four confidence-ranked
50h chunks (chunk 1 = most Levantine-confident, chunk 4 = least) in a different
order, to separate "more training data" from "training-data quality" as causes of
the WER degradation seen when chunks 3-4 are included:

| run | chunk order | purpose | status |
|---|---|---|---|
| **forward** | 1,2,3,4,1,2,3,4 (2 epochs) | baseline: best-first | **done**, 8/8 checkpoints |
| **reverse2** | 4,3,4,3 (2 epochs, worst 2 chunks only) | isolate quality from quantity | **done**, 4/4 checkpoints |
| **reverse_all** | 4,3,2,1 (full reverse) | does order matter across the whole set | **not started** |

`reverse_all` inherits `reverse2`'s first 100h (stages `s1_c4`, `s2_c3`) via
`--init-from outputs/whisper_medium_lora_reverse2/checkpoints/s2_c3
--init-stage-offset 2` rather than retraining that shared prefix — see
`scripts/train_whisper_medium_lora_sequence.py`'s module docstring for exactly how
that works (adopts LoRA weights + optimizer moments, fast-forwards the new run's own
OneCycleLR schedule past the inherited steps).

All three share: base model `openai/whisper-medium`, seed 42, LoRA r=32/alpha=32/
dropout=0.05 on `[q_proj,fc1,v_proj,k_proj,fc2,out_proj]`, batch size 8, lr 1e-4,
warmup 0.1. Full config machine-readable at
`outputs/experiment_pipeline` runs (see below) and mirrored in
`scripts/push_results_to_r2.py`'s `CONFIG` dict.

## Everything is paused right now

I stopped the orchestrator/supervisor pair **after `reverse2`'s last stage
(`s4_c3`) finished cleanly** — no process was killed mid-training or mid-eval.
Nothing is running. GPU is idle. This was a deliberate pause per instruction, not a
crash — do not treat this as something to "recover" from.

**To resume the queue exactly where it left off:**
```bash
cd /root/Palestinian-ASR
setsid bash /root/asr_pipeline/pipeline_supervisor.sh > /dev/null 2>&1 < /dev/null &
disown
```
The supervisor re-derives "what's left" from `outputs/experiment_pipeline/*.done`
markers and each run's `checkpoints/<tag>/summary.json` — it will skip phases 0,
0b, 1, and 2 (already `.done`) and start phase 3 (`reverse_all`) directly. Both
scripts live at `/root/asr_pipeline/` (moved there mid-session from a
session-scoped scratchpad, which would not have survived past that session).

Remaining phases if resumed: phase 3 (`reverse_all`, 2 new stages, chunks 2 then 1,
~2.5h) → phase 4 (benchmark the 6 new adapters — reverse2's 4 + reverse_all's 2 —
against Casablanca-PAL/JOR, Layla, Omni, ~1h). Total ETA from resume: ~3.5h.

## Results so far

Full detail: `outputs/experiment_pipeline/RESULTS.md` (regenerates from
`scripts/report_all_results.py`, safe to rerun any time).

### forward run — in-domain (speaker-disjoint val/test)

| checkpoint | val WER | test WER | val loss | test loss |
|---|---|---|---|---|
| ep1_h50 | 49.80 | 42.90 | 0.7427 | 0.6015 |
| ep1_h100 | 49.13 | 41.59 | 0.7259 | 0.5698 |
| ep1_h150 | 49.80 | 42.34 | 0.7106 | 0.5608 |
| ep1_h200 | 49.68 | 42.48 | 0.6979 | 0.5518 |
| **ep2_h50** | **44.67** | **38.52** | **0.5516** | **0.4519** |
| ep2_h100 | 45.52 | 38.60 | 0.5960 | 0.4725 |
| ep2_h150 | 45.58 | 38.93 | 0.5894 | 0.4697 |
| ep2_h200 | 45.72 | 39.11 | 0.5933 | 0.4735 |

`ep2_h50` is the best checkpoint on every axis of the whole forward run.

### reverse2 — in-domain

| checkpoint | hours | val WER | test WER | val loss | test loss |
|---|---|---|---|---|---|
| s1_c4 | 50 (chunk 4 only) | 53.23 | 45.79 | 0.8029 | 0.6496 |
| s2_c3 | 100 (chunk 4+3) | 50.82 | 43.59 | 0.7340 | 0.5853 |
| s3_c4 | 150 (chunk 4+3+4) | 49.64 | 42.16 | 0.7067 | 0.5621 |
| s4_c3 | 200 (chunk 4+3+4+3) | 49.26 | 41.76 | 0.7029 | 0.5582 |

**reverse2 is complete: all 4 stages done, all 4 checkpoints benchmarked in-domain.**

**Finding**: worst-first (reverse2) starts markedly worse than best-first (forward)
at every matched hour-count (50h: 45.79 vs 42.90 test WER) but the gap **fully
closes and inverts** as training repeats the same two hard chunks: by 200h,
reverse2's `s4_c3` (41.76 test WER) actually **beats** forward's `ep1_h200`
(42.48) — 2 epochs over 100h of hard data ends up better than 1 epoch over 200h
that includes the same hard data once. Repetition matters as much as, or more
than, which chunks are novel. This is a genuinely interesting result worth
flagging for the paper: it argues against "more diverse data is better" and
toward "repeated exposure to lower-confidence data still helps, just needs more
passes" — very different framings for how to use chunks 3-4 going forward
(revisit them more, rather than discard them).

### Held-out loss vs WER — a real divergence, not yet explained

Within the forward run, chunks 3-4 **raise train loss** (chunk-2→4: 0.26→0.33 in
epoch 2) while **held-out loss keeps falling** through epoch 1 (0.57→0.55) — i.e.
teacher-forced likelihood improves even as free-running WER gets worse after the
h100 peak. This is consistent with exposure bias (teacher forcing never sees error
accumulation) rather than overfitting, but is not proven — `reverse_all` finishing
would help confirm whether this pattern is chunk-identity-specific or a general
property of "later stages in a long OneCycle."

### Out-of-domain (Casablanca-PAL/JOR, Layla, Omni) — forward run only so far

Every forward checkpoint beats zero-shot Whisper-medium's WER on every OOD set —
fine-tuning on the custom corpus generalizes rather than overfitting to it. CER is
a different story: it's often *worse* than the zero-shot baseline even where WER
improves (e.g. Layla: base 14.03 CER vs `ep2_h100`'s 22.96) — plausibly more
character-level noise (hallucinated diacritics/function words) despite closer
word-level structure. Full table in `RESULTS.md`. `reverse2`/`reverse_all`
adapters have NOT been OOD-benchmarked yet (that's phase 4).

## What's on R2 now

Pushed by `scripts/push_results_to_r2.py` (run it again any time — it always
targets today's date-stamped directory unless you pass an explicit path, and
`pyarrow.fs.copy_files` overwrites cleanly on rerun):

```
backup/transfer/adapters/lora_speaker_disjoint_<DATE>/
├── forward/checkpoints/<tag>/{adapter/, summary.json}   x8   + all_results.json + train.log
├── reverse2/checkpoints/<tag>/{adapter/, summary.json}  x4   + all_results.json + train.log
├── benchmarks/                                          out-of-domain WER/CER (forward only)
├── pipeline_results/{RESULTS.md,RESULTS.json,val_loss.json,train_loss_curves.json}
├── config.json                                          hyperparameters (see above)
└── README.md
```

`resume_state/` (in-flight optimizer/RNG state) is deliberately **not** pushed —
it's only meaningful for resuming on this exact box, not as a portable artifact.
When `reverse_all` finishes, rerun `push_results_to_r2.py` to add it and refresh
`benchmarks/`.

**No new source data was created this session.** The speaker-disjoint split
definition (`speaker_assignments.json`, `selection.csv`, etc.) was already fully
uploaded to R2 under `backup/transfer/data/speaker_disjoint_split_v1/` before this
session started (2026-09-03). The training parquets this session extracted
(`train_full_200h.parquet`, `val.parquet`) are derived, reproducible copies of that
plus `data_lev_custom_split_v1` — both already in R2 — so nothing new belonged
under the `data/` prefix.

## What's in this GitHub push

New/changed files, all committed:
- `scripts/train_whisper_medium_lora_progressive.py` — the forward run (now also
  returns its training-loss curve; see `train_one_chunk`'s docstring)
- `scripts/train_whisper_medium_lora_sequence.py` — generalizes it to an arbitrary
  chunk sequence + `--init-from` (used by reverse2/reverse_all)
- `scripts/eval_val_loss.py` — teacher-forced val/test cross-entropy; backfilled it
  for the forward run's 8 checkpoints (it never computed one originally) and it's
  now wired natively into the sequence trainer
- `scripts/extract_train_loss_curves.py` — recovers per-step training loss from log
  text (the forward run never wrote it to a summary)
- `scripts/eval_adapter_benchmarks.py` — out-of-domain benchmark harness (merges a
  LoRA adapter into the base model, reuses the zero-shot eval pipeline unmodified)
- `scripts/report_all_results.py` — regenerates `outputs/experiment_pipeline/RESULTS.md`
  from whatever's on disk; safe at any point, doubles as a progress view
- `scripts/push_results_to_r2.py` — this session's R2 push (see above)
- `outputs/whisper_medium_lora_progressive/all_results.json`,
  `outputs/whisper_medium_lora_reverse2/all_results.json`,
  `outputs/adapter_benchmarks/**`, `outputs/experiment_pipeline/**` — small JSON/MD
  result files (checkpoints themselves — the actual adapter weights — are
  `.gitignore`d as always; find them in R2 instead, see above)
- `.gitignore` — added `outputs/experiment_pipeline/*.{pid,done}` and
  `**/resume_state/` (both are runtime-only state, meaningless as repo history)

The orchestrator/supervisor bash scripts themselves
(`/root/asr_pipeline/pipeline_{orchestrator,supervisor}.sh`) are **not** in this
push — they live outside the repo at `/root/asr_pipeline/` by design (see "survives
disconnects" reasoning in that session's chat). If you want them version-controlled
too, say so explicitly next session; they weren't copied in here to keep this push
focused on what was asked.

## Resuming / next steps

1. To finish the queue: relaunch the supervisor (command above) — phases 3-4 will
   run unattended, same self-healing guarantees as before (verified this session:
   supervisor and orchestrator each independently restore the other within ~15s of
   being killed).
2. Once `reverse_all` finishes, rerun `scripts/eval_adapter_benchmarks.py --discover`
   (or just resume the queue — phase 4 does this automatically) to OOD-benchmark
   its 2 new checkpoints plus reverse2's 4, then rerun `push_results_to_r2.py`.
3. Open question worth chasing: does the train/held-out-loss divergence in the
   forward run reproduce in `reverse_all`'s chunk-2/chunk-1 stages, or is it
   specific to chunks 3-4's data composition? `reverse_all` finishing settles this.
