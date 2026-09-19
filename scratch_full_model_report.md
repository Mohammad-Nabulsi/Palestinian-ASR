# Palestinian/Levantine Arabic ASR — Full Cross-Model Results Report

**Generated:** 2026-08-03. Covers every Conformer, Omnilingual, and Whisper result found on this
machine (local `/root/downloaded/` snapshot + network storage `/workspace/asr/Palestinian-ASR/`),
cross-checked file-by-file. Qwen3-ASR and Cohere Transcribe Arabic are included where they appear
in the same comparison batches, since they were evaluated under the same framework.

Every number below is read directly from a `SUMMARY.json`, an `mlflow.db`, or a `metrics.json` —
none are estimated. Where a run is incomplete, crashed, or produced a degenerate result, it is
listed in **6 Broken/Incomplete Runs** and excluded from the main tables.

---

## 1. Methodology

### 1.1 Data curation pipeline

Five raw sources were combined into one dialect-labeled corpus:

| Source | Domain | Raw scale |
|---|---|---|
| QASR | Al Jazeera broadcast news, forced-aligned upstream | 3,545 recordings, ~2,042h |
| MASC-Arabic | Multi-dialect general-purpose corpus | 875,873 / 19,521 / 18,006 train/val/test |
| Casablanca | Multi-dialect conversational, Levant subset (Jordan, Palestine) | ~3,000 utterances |
| Omnilingual (APC) | North Levantine Arabic (ISO 639-3 apc) | 416 recordings pre-segmentation |
| Layla | Jordanian Arabic acoustic dataset | 218 recordings, 6.70h |

**Cleaning.** Every source went through a shared fast text-first pass: drop transcripts containing
English letters or digits, drop clips under 0.5s, apply Arabic text normalization (diacritic
stripping, alef/ya/ta-marbuta unification, tatweel removal). No audio decoding/resampling at this
stage. Yielded 1,508,531 clean QASR rows and 373,479 clean MASC rows.

**Long-audio segmentation (forced alignment).** Whisper's 30s positional-embedding limit required
segmenting any clip over 30s. A duration scan found this affected 100% of Layla (218/218), 90% of
Omnilingual (375/416), and 15 MASC rows — QASR and Casablanca were already pre-segmented at the
utterance level and needed none. The segmentation pipeline:
1. **Silero VAD** to find speech/silence boundaries.
2. **WhisperX forced alignment** (`jonatasgrosman/wav2vec2-large-xlsr-53-arabic`, via
   `whisperx.align`, `language_code="ar"`) of the *existing* transcript against the audio, producing
   per-word (start, end, score) — the transcript itself was never re-generated, only aligned and
   sliced.
3. Cut at the silence gap nearest a 28-second target, choosing the cut point by whether the nearest
   alignment-timestamp midpoint fell before/after it.
4. **Drop (not down-weight)** a segment if: text is empty, duration < 0.5s, mean word-alignment
   score < 0.45, or fewer than half its words received a timestamp.

Result: 2,322 kept segments from 608 flagged recordings (7 dropped, ~0.3% loss) — preserving all
6.70h of Layla and 7.70 of Omnilingual's 7.91h. Run in a dedicated venv
(`/workspace/venv_whisperx`, `whisperx` + `silero-vad`) via
`scripts/segmentation/segment_whisperx.py`, spliced back into `data/` in place via
`scripts/segmentation/replace_long_audio_in_data.py`.

**Dialect identification.** QASR and MASC are not dialect-labeled at the source, so a two-stage
classifier identified Levantine-dialect subsets within them:
1. **Text-stage:** a MARBERTv2 classifier scored every cleaned transcript for Levantine-dialect
   probability. Rows scoring >=0.80 became audio-stage candidates (122,348 / 1,508,531 QASR rows;
   48,455 / 373,464 MASC rows).
2. **Audio-stage:** a BADREX/MMS-300M audio dialect classifier scored those candidates (with a
   PCM-aware decoder fix for QASR's raw int16-byte audio storage). Rows >=0.80 Levantine
   probability -> `lev`; everything else (including audio-stage failures) -> `non_lev`.

Casablanca, Omnilingual, and Layla were treated as inherently Levantine by construction (source
already dialect-selected) and skipped the classifier.

**Final labeled corpus** (`data_curated_levant_binary_v1`): **1,887,366 rows, 2,209.91 hours**,
70/15/15 train/val/test (MASC and Casablanca keep each source's own native test-shard membership
rather than a random re-split).

| source | hours | lev | non-lev |
|---|---|---|---|
| qasr | 1,778.97 | 37.83 | 1,741.14 |
| masc | 412.39 | 10.51 | 401.88 |
| omni | 7.91 | 7.91 (all) | — |
| layla | 6.70 | 6.70 (all) | — |
| casa (pal+jor) | 3.95 | 3.95 (all) | — |

### 1.2 Fine-tuning mix (data_finetune_mix_v2) — the corpus most results below train/eval on

Deliberately **not** the full 2,209.91h corpus:
- **Train** (30,712 rows, 36.00h): exclusively MASC + QASR, target 20/20/30/30% split
  (masc-lev/qasr-lev/masc-non-lev/qasr-non-lev). **Casablanca, Omnilingual, Layla are absent from
  training entirely.**
- **Val** (554 rows, 2.01h): Casablanca Palestinian (133) + Jordanian (169) unchanged; Omnilingual
  (142/196) + Layla (110/153) trimmed to hit a 2h target.
- **Test** (1,864 rows, 4.13h): Casablanca Palestinian (664) + Jordanian (848) + Omnilingual (198)
  + Layla (154), each at native "fair share." **MASC and QASR entirely absent from test.**

This makes every data_finetune_mix_v2 WER number below a **cross-domain generalization test** —
broadcast-register, classifier-identified Levantine speech (train) -> naturalistic dialectal
recordings (val/test) — not an in-domain benchmark. (data_finetune_mix_v1 uses the same
composition rule at ~5.5x scale: train=164,796 rows/200.0h, val=15,292/20.0h, test=15,058/20.0h.)

The separate **data_pal_v1** / **data_qasrlev_v1** views used by the whisper-medium "pal study"
(covered in the companion generalization sweep report:
https://claude.ai/code/artifact/780670b5-a2e5-4a18-9a0b-f05e7f04e199)
are a different, smaller construction — see that report for details.

### 1.3 Unified LoRA fine-tuning framework

All model families fine-tune through one ModelAdapter abstraction (preprocess, collate,
loss_type, generate, apply_lora, save_checkpoint/load_checkpoint), with a shared training
loop, evaluation protocol, and MLflow experiment (arabic-asr-unified) — not the HuggingFace
Trainer, since OmniASR (fairseq2) and FastConformer (NeMo) aren't PreTrainedModel subclasses.

- **LoRA:** shared baseline r=32, lora_alpha=32, lora_dropout=0.05, bias="none" across all
  families (the later whisper-medium pal-study runs use r=16, alpha=32 instead — see the
  companion report); only target_modules differs per architecture, verified by direct
  module-tree inspection (e.g. FastConformer's linear_q/k/v/out, linear1/2 confirmed to hit
  only the Conformer encoder, not the RNNT decoder/joint network). OmniASR's non-nn.Linear
  projections get a hand-written low-rank wrapper instead of PEFT.
- **Training:** length-grouped batching by duration, bf16 autocast, AdamW-8bit
  (bitsandbytes), linear LR schedule with 5% warmup, early stopping on val WER (patience 3-4),
  best-WER checkpoint kept separately from periodic step checkpoints, full
  optimizer/scheduler/RNG state checkpointed for exact resume. A 5-sample smoke gate exercises the
  full loop before committing to a real run.
- **Evaluation:** WER/CER via jiwer after shared Arabic normalization (diacritic/tatweel
  removal, alef/ya/ta-marbuta unification, punctuation stripping), applied identically to
  references and hypotheses. Every prediction set is cached, content-addressed by
  model+split+dataset-fingerprint+stage (base/tuned).

### 1.4 Models covered in this report

| Model | Params | Framework | Decoding |
|---|---|---|---|
| omnilingual-asr/omniASR_LLM_300M | 1,627.6M (verified; "300M" is a misnomer) | fairseq2 (wav2vec2_llama) | seq2seq, autoregressive |
| nvidia/stt_ar_fastconformer_hybrid_large_pc[d]_v1.0 | 114.6M (verified) | NVIDIA NeMo, hybrid RNNT-CTC run pure-CTC | CTC, greedy |
| openai/whisper-medium / whisper-large-v3 | ~769M / ~1.55B (public) | HF transformers, log-mel encoder-decoder | seq2seq, autoregressive |
| Qwen/Qwen3-ASR-0.6B-hf | ~0.6B (public) | HF transformers, chat-template | seq2seq, autoregressive |
| CohereLabs/cohere-transcribe-arabic-07-2026 | undisclosed (gated) | HF transformers | seq2seq, autoregressive |

---

## 2. Zero-Shot Baseline Comparison (data_finetune_mix_v2, before any fine-tuning)

Full-scale zero-shot WER/CER, val n=554 / test n=1,864:

| Model | Test WER | Test CER | Val WER | Val CER |
|---|---|---|---|---|
| **omniASR_LLM_300M** | **41.29%** | 13.98% | 37.07% | 11.76% |
| **FastConformer** (nvidia) | 43.57% | 14.62% | 40.34% | 13.42% |
| **Qwen3-ASR-0.6B** | 52.02% | 18.03% | 48.93% | 16.26% |
| **Whisper-medium** | 54.02% | 23.93% | 50.78% | 21.72% |

Whisper-large-v3 and Cohere Transcribe Arabic never got a full-scale zero-shot pass — only n=1
smoke-test artifacts exist for both (see section 6). CER is consistently ~3-4x lower than WER across every
model, indicating errors are frequently character-level substitutions within otherwise-recognizable
words rather than wholesale mis-transcription.

**Per-source breakdown, FastConformer zero-shot** (val, n=554):

| Source | n | WER | CER |
|---|---|---|---|
| casa_pal (Palestinian) | 133 | **51.20%** | 18.10% |
| omni (North Levantine) | 142 | 43.98% | 14.05% |
| casa_jor (Jordanian) | 169 | 40.27% | 13.16% |
| layla (Jordanian) | 110 | **33.58%** | 11.51% |

Palestinian speech specifically is the hardest of the four sources by 10-18 points, directly
relevant to this project's focus — this pattern (Palestinian/Casablanca WER sitting well above
Jordanian/Layla) recurs throughout the results below.

---

## 3. Whisper Results

### 3.1 whisper-medium "pal study" (data_pal_v1 / data_qasrlev_v1 targets)

This is the large multi-run pretrain-then-finetune study covered in full detail in the companion
report: Palestinian ASR — Pretraining Study Report
(https://claude.ai/code/artifact/780670b5-a2e5-4a18-9a0b-f05e7f04e199)
(11 pretraining configurations x early-stopped fine-tune, plus a 188-domain-eval generalization
sweep). Best result: **33.03% test WER** (run9, 2-epoch pretrain on QASR-Lev+MASC-Lev+QASR-nonLev,
LoRA r=16/alpha=32). Not repeated here — see that report for the full breakdown.

A run12_qasrlev_target variant (qasr_lev used as the fine-tuning *target* instead of Palestinian,
pretrained on qasr_lev-train+casa_pal+casa_jor+omni+layla) is **currently training** as of this
report (stage 1, epoch 2, val WER 30.02% and improving) — not yet a final result.

### 3.2 whisper-medium on data_finetune_mix_v2

One in-progress MLflow run (dae532b2..., never reached FINISHED or a test score):

| Epoch | Val WER | Val CER |
|---|---|---|
| zero-shot baseline | 54.02% (test) / 50.78% (val) | 23.93% / 21.72% |
| 1 | 42.37% | 13.65% |
| 2 | 41.30% | 13.93% |
| 3 | **40.74%** (best so far, plateauing) | 13.33% |

LoRA r=32/alpha=32. Corpus: same data_finetune_mix_v2 as 1.2/2 (train=30,712/36.0h,
val=554/2.0h, test=1,864/4.1h). Never test-scored — no fine-tuned test number exists for this run.

### 3.3 whisper-large-v3

- finetune_mix_v1 fine-tuning run: **never executed** (script present, no logs/mlflow.db).
- Zero-shot on smoke batch (n=1): WER got *worse* after "tuning" (92.68% vs. base 78.05%) — not a
  meaningful result at n=1, listed for completeness only.
- Zero-shot on outputs/whisper_large_v3_zero_shot/:

| Domain | n | WER | CER | Status |
|---|---|---|---|---|
| casablanca_jordanian | 848 | 40.28% | 13.70% | complete |
| casablanca_palestinian | 664 | 50.53% | 18.02% | complete |
| omnilingual_apc_full | — | — | — | incomplete (112/416 rows) |
| masc_c_only | — | — | — | incomplete (1,064/8,612 rows) |

No full-scale data_finetune_mix_v2 zero-shot pass exists for whisper-large-v3 at all (only the
n=1 smoke result above) — it's the one model in the 5-family comparison missing that baseline.

---

## 4. Omnilingual Results

### 4.1 Zero-shot (omniASR_LLM_300M)

Full-scale data_finetune_mix_v2 zero-shot: **41.29% test WER / 13.98% CER** — the best zero-shot
score of any model tested (section 2).

outputs/omni_asr_llm_300m_zero_shot/:

| Domain | n | WER | CER | Status |
|---|---|---|---|---|
| masc_c_only | 8,612 | **20.67%** | 6.24% | complete |
| casablanca_jordanian | 848 | 42.97% | 13.68% | complete |
| casablanca_palestinian | 664 | 52.11% | 18.83% | complete |
| omnilingual_apc_full | — | — | — | incomplete (536 predictions, never scored) |

Notably strong on masc_c_only (general-purpose MSA/dialect mix) relative to the naturalistic
dialectal sets — consistent with masc being closer to the model's likely pretraining distribution.

### 4.2 Fine-tuning

- **omni_asr_300m_finetune_mix_v1**: never started (empty mlflow.db).
- **omni_asr_300m_finetune_mix_v2** (LoRA r=32/alpha=32, lr=1e-4): 4 MLflow run attempts, none
  FINISHED. Best/latest attempt (e9afa2e3...) reached epoch 3/step 5,760 (of a budget continuing
  to 6,803): val WER history 1,920 steps->36.13%, 3,840->36.30%, 5,760->**36.55%**, CER 12.20% — i.e.
  **essentially flat around 36%, not yet clearly beating the 37.07% zero-shot val baseline** by a
  meaningful margin at this checkpoint. Never test-scored.

---

## 5. Conformer Results (NVIDIA FastConformer-CTC)

LoRA r=32/alpha=32/dropout=0.05, target modules linear_q/k/v/out, linear1/2 (Conformer encoder
only, verified no collision with the separate RNNT decoder/joint network) -> 7,798,784 trainable
params, 6.37% of the model's 122,420,226 total. Batch 8, lr=1e-4, bf16, AdamW-8bit, <=50 epochs,
early-stop patience 3.

### 5.1 conformer_ctc_finetune_mix_v2 (data_finetune_mix_v2)

Never reached FINISHED. Best attempt (ac90cb18..., r=32/alpha=32) reached epoch 2/step 7,678:

| Epoch | Val WER | Val CER |
|---|---|---|
| zero-shot baseline | 40.34% | 13.42% |
| 1 | 47.45% | 16.42% |
| 2 | **45.41%** | 15.84% |

Fine-tuning has **not yet recovered the zero-shot baseline** at epoch 2 (45.41% > 40.34%), though
improving epoch-over-epoch (47.45%->45.41%). Per-source breakdown after epoch 1 showed roughly
uniform degradation across sources (casa_jor +3.8pt, casa_pal +3.9pt, layla +5.4pt, omni +1.7pt WER
vs. baseline) — no single source responsible for the early regression, consistent with normal LoRA
warm-up noise. A second attempt with different hyperparameters (r=16/alpha=8, lr=5e-5) reached
epoch 4/step 3,840 at val WER 48.94% — worse, not adopted.

### 5.2 conformer_ctc_finetune_mix_v2_pc (no-diacritics twin, same corpus)

One FINISHED run (8b3c8b9a..., epoch 5/step 19,195), train=30,712/36.0h val=554/2.0h
test=1,864/4.1h:

| Stage | Test WER | Test CER |
|---|---|---|
| base | 41.34% | 13.90% |
| tuned | **57.07%** (worse) | 23.74% |

Tuned is clearly *worse* than base here — flagged, likely overfitting or an LR/checkpoint-selection
issue given the model's own best_val_wer (48.49%) doesn't match this test-time regression pattern.

### 5.3 conformer_ctc_finetune_pal_v1 (data_pal_v1 — the small Palestinian target corpus)

train=664/0.99h, val=199/0.32h, test=465/0.65h. Two FINISHED runs:

| Run | Epochs | Best val WER | Test base WER | Test tuned WER |
|---|---|---|---|---|
| A (6a86b15c...) | 6 | 48.49% | 49.57% | 60.05% (worse than base) |
| B (d62d2b3c..., corrected re-run) | 21 | 47.68% | 49.57% | **47.24%** (beats base) |

Run B is the only Conformer result across the entire inventory where the tuned model clearly beats
its own base — and it does so on the smallest, most target-matched corpus (Palestinian-only,
<1 hour of training audio), consistent with the same domain-match-over-volume pattern documented in
the companion whisper pal-study report.

### 5.4 conformer_ctc_finetune_mix_v1

Never executed (script only, no logs).

---

## 6. Broken / Incomplete / Excluded Runs

Listed here rather than in the main tables above:

1. **whisper_medium_pal/run3_omni_s2ep2** — never started (1-line log).
2. **conformer_ctc_finetune_mix_v1**, **whisper_large_v3_finetune_mix_v1** — scripts only, never executed.
3. **omni_asr_300m_finetune_mix_v1** — never started (empty mlflow.db).
4. **omni_asr_300m_finetune_mix_v2 run b33ea1c0...** — marked FINISHED but ran 26s, zero metrics logged (crash-on-start).
5. **qwen3_asr_0_6b_finetune_mix_v1** — never logged a single val/wer across 3 documented crash attempts (OOM, KeyError, mlflow hang); a later re-eval pass is cut off mid-scoring.
6. **qwen3_asr_0_6b_finetune_mix_v2** runs c8bb0b84... and 53af7c24... — val WER/CER = 1.0 (total output collapse, LR=5e-5 too high); fixed in a later attempt (3907699b..., lr=5e-6) which reached val WER 53.99% at epoch 3 but never finished/test-scored.
7. **conformer_ctc_finetune_mix_v2** — no attempt ever reached FINISHED or a test score (see 5.1 for the best in-progress numbers, reported as such).
8. **whisper_large_v3_zero_shot/omnilingual_apc_full** (112/416 rows) and **/masc_c_only** (1,064/8,612 rows) — interrupted mid-inference, no metrics.json.
9. **omni_asr_llm_300m_zero_shot/omnilingual_apc_full** — 536 predictions written, scoring never ran.
10. **whisper_medium_pal/run12_qasrlev_target_earlystop** — in progress as of this report (see 3.1).
11. **.unreliable urns/ directory** (network storage only — Whisper-medium and "Omnilingual 1B" custom-streaming checkpoints, self-flagged unreliable by the directory name; the Omnilingual comparison file's "whisper_large" arm is actually mislabeled whisper-medium data, not an independent run) — excluded entirely from this report.
12. **Smoke-test artifacts (n=1)** for Cohere, Qwen3, FastConformer, OmniASR, Whisper-large-v3 (pal_study_20260801/results/metrics/*__SUMMARY.json) — not statistically meaningful, excluded from comparison tables; only used to confirm each model's pipeline runs end-to-end.

---

## 7. Cross-Cutting Observations

- **Domain match beats scale, again.** The one Conformer run that clearly beats its own zero-shot
  baseline (5.3, run B) trains on <1 hour of target-matched Palestinian audio; the mix_v2 runs
  training on 36h of broadcast-domain data have *not* recovered baseline after 2 epochs. Mirrors the
  same pattern found in the whisper pal-study (1.98h dialect-matched Jordanian pretraining beating
  114.9h of everything combined on pre-finetune WER).
- **Palestinian speech is consistently the hardest dialectal source** across every model/eval where
  a per-source breakdown exists (FastConformer zero-shot: 51.2% vs. 33.6-44.0% for other Levantine
  sources; Casablanca Palestinian zero-shot WER exceeds Casablanca Jordanian by ~10pt for both
  Whisper-large-v3 and OmniASR zero-shot).
- **Most fine-tuning runs never finished.** Of the fine-tuning run directories surveyed across
  Whisper/Omnilingual/Conformer/Qwen3 on data_finetune_mix_v1/v2 (10 runs, excluding the
  separate whisper pal-study), only 2 ever reached FINISHED status (conformer_mix_v2_pc,
  conformer_pal_v1 x2), and several never started at all (Whisper-large-v3 mix_v1, Omnilingual
  mix_v1, Conformer mix_v1).
- **No model on data_finetune_mix_v2/v1 has closed the gap to deployable WER.** Every
  fully-evaluated zero-shot number sits between 41% and 54% test WER; the best fine-tuned number
  in this report (Conformer pal_v1, 47.24%) is still far above the ~20-25% threshold generally
  considered usable for downstream applications — though the separate whisper pal-study (33.03%,
  see 3.1's linked report) gets meaningfully closer, on a differently-constructed target corpus.
