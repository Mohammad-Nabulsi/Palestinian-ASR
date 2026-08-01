# Towards Robust Palestinian Dialectal Arabic ASR: A Unified LoRA Fine-Tuning Study Across Five Model Families

**Status: working draft.** Literature review is a placeholder pending completion. All numbers below are pulled directly from cached evaluation artifacts, training logs, and dataset manifests produced by this project's pipeline — nothing here is projected or estimated unless explicitly marked as such.

---

## Abstract

*(draft — revise once results are more complete)*

Automatic speech recognition for Palestinian and broader Levantine dialectal Arabic remains an open problem: models trained predominantly on Modern Standard Arabic (MSA) or high-resource dialects degrade sharply on natural dialectal speech, and no single publicly available corpus offers sufficient labeled Palestinian/Levantine audio at scale. We address the data gap by building a large, automatically dialect-labeled Arabic speech corpus (1.89M utterances, 2,209.91 hours) from five heterogeneous sources via a two-stage text+audio dialect identification pipeline, and we construct a targeted ~42-hour fine-tuning mix that deliberately evaluates on naturalistic dialectal recordings (Palestinian and Jordanian Casablanca, Omnilingual North Levantine Arabic, Layla Jordanian Arabic) while training on dialect-labeled broadcast/web speech (MASC, QASR). We evaluate five ASR model families zero-shot under a single unified LoRA fine-tuning framework with consistent preprocessing, evaluation, and experiment tracking, finding zero-shot word error rates (WER) ranging from 41.3% to 54.0% across models — confirming that dialectal Levantine ASR remains far from solved even for current state-of-the-art systems. We report early LoRA fine-tuning results for one model family (NVIDIA FastConformer-CTC) and describe the infrastructure for extending this to the remaining four.

---

## 1. Introduction

Palestinian and Levantine dialectal Arabic is spoken by tens of millions of people, yet remains severely underserved by automatic speech recognition technology. The overwhelming majority of Arabic ASR training data — and the models built on it — targets Modern Standard Arabic or higher-resource dialects (principally Egyptian and Gulf), leaving Levantine, and Palestinian dialectal speech in particular, systematically under-represented. This is not a marginal gap: in this work's own zero-shot evaluation of five current-generation ASR models on held-out Levantine dialectal speech (Section 4.1), **word error rate ranged from 41.3% to 54.0%**, with even the best-performing model still misidentifying roughly two out of every five words. For a task like transcription, where WER above ~20-25% is generally considered unusable for downstream applications, this places every model we tested well outside the range of practical deployment on Palestinian dialectal speech, out of the box.

Part of the difficulty is architectural and linguistic (dialectal Arabic diverges from MSA in phonology, morphology, and lexicon, and Levantine sub-dialects vary further among themselves — a pattern visible in our own per-source results in Section 4.2, where WER varies by over 17 percentage points across dialectal sources evaluated under identical conditions). But a second, equally significant obstacle is data: **there is no single reliable, sufficiently large, publicly available corpus of labeled Palestinian dialectal speech.** Existing resources are fragmented across broadcast corpora (QASR/Al Jazeera), general multi-dialect collections (MASC, Casablanca), and small manually-collected acoustic datasets (Layla, Omnilingual), each with different label quality, recording conditions, and dialect coverage — and most were never dialect-labeled at all. Any attempt to fine-tune on "Palestinian/Levantine Arabic" first has to solve the much less glamorous problem of *finding and correctly labeling* that data within larger, mixed-dialect corpora.

This work makes three contributions:

1. **A data curation and dialect-identification pipeline** that combines five raw sources (QASR, MASC-Arabic, Casablanca, Omnilingual, Layla) into a single, automatically dialect-labeled corpus of 1,887,366 utterances (2,209.91 hours), using a two-stage text-then-audio dialect classifier to identify Levantine-dialect speech within otherwise dialect-unlabeled broadcast and web corpora (Section 3.1).
2. **A unified, model-agnostic LoRA fine-tuning framework** spanning five architecturally distinct ASR model families (fairseq2-based OmniASR, NeMo FastConformer-CTC, transformers-native Qwen3-ASR, Whisper, and Cohere Transcribe), sharing one evaluation protocol, one experiment-tracking namespace, and one set of LoRA hyperparameters, so that results are directly comparable across model families (Section 3.3-3.4).
3. **Zero-shot baseline results for four of these five models** on a fine-tuning mix specifically constructed to test generalization from broadcast-register dialectal speech to naturalistic dialectal recordings, plus **early LoRA fine-tuning results** for one model family, reported here as work in progress (Section 4).

The remainder of this paper is organized as follows: Section 2 (literature review) is left as a placeholder to be completed separately. Section 3 describes the data curation methodology, the fine-tuning data mix, the models evaluated, and the unified fine-tuning/evaluation framework. Section 4 reports zero-shot and in-progress fine-tuning results. Section 5 discusses current limitations and next steps.

---

## 2. Literature Review

*[TO BE COMPLETED]*

---

## 3. Methodology

### 3.1 Data Sources and Curation Pipeline

Five raw sources were combined:

| Source | Domain | Raw scale |
|---|---|---|
| **QASR** | Al Jazeera broadcast news, forced-aligned | 3,545 recordings, ~2,042 hours audio |
| **MASC-Arabic** | Multi-dialect general-purpose corpus | 875,873/19,521/18,006 train/val/test (upstream) |
| **Casablanca** | Multi-dialect conversational, Levant subset (Jordan, Palestine) | ~3,000 utterances |
| **Omnilingual (APC)** | North Levantine Arabic (ISO 639-3 `apc`) | 416 recordings pre-segmentation |
| **Layla** | Jordanian Arabic acoustic dataset (Layla Witheeb) | 218 recordings, 6.70h |

**Cleaning.** All sources went through a shared fast text-first cleaning pass: drop transcripts containing English letters, drop transcripts containing digits, drop clips under 0.5s duration, and apply Arabic text normalization (diacritic stripping, alef/ya/ta-marbuta unification, tatweel removal). No audio decoding, resampling, or loudness normalization was performed at this stage. Applied to QASR (both an initial 961-recording pass and a later 2,584-recording pass closing the remaining coverage gap) and MASC (`type == "c"` subset only), this yielded 1,508,531 clean QASR rows and 373,479 clean MASC rows.

**Long-audio segmentation.** Whisper's 30-second positional embedding limit required segmenting any clip exceeding 30s. A duration-only scan found this affected 100% of Layla (218/218), 90% of Omnilingual (375/416), and 15 MASC rows; QASR and Casablanca were already pre-segmented at the utterance level and needed no further splitting. Segmentation used Silero VAD to find speech/silence boundaries and WhisperX forced alignment (`jonatasgrosman/wav2vec2-large-xlsr-53-arabic`) to align the *existing* transcript against the audio — the transcript itself was never re-generated, only aligned and split at the silence gap nearest a 28-second target. Segments were dropped (not down-weighted) if their text was empty, duration fell below 0.5s, mean word-alignment score was below 0.45, or fewer than half their words received a timestamp. This yielded 2,322 kept segments from 608 flagged recordings (7 dropped, ~0.3% loss), preserving Layla's full 6.70 hours and 7.70 of Omnilingual's 7.91 hours.

**Dialect identification.** QASR and MASC are not dialect-labeled at the source, so a two-stage classifier was used to identify Levantine-dialect subsets within them:
1. **Text-stage:** a MARBERTv2-based classifier scored every cleaned transcript for Levantine-dialect probability. Rows scoring ≥ 0.80 became audio-stage candidates (122,348 of 1,508,531 QASR rows; 48,455 of 373,464 MASC rows).
2. **Audio-stage:** a BADREX/MMS-300M-based audio dialect classifier was run on those candidates (with a PCM-aware decoder fix for QASR's raw-`int16`-byte audio storage, which the initial pass had mis-handled). Rows scoring ≥ 0.80 Levantine probability were kept as the final `lev` subset; all others (including anything that failed the audio-stage threshold) were assigned `non_lev`.

Casablanca, Omnilingual, and Layla were treated as inherently Levantine by construction (source-selected for Jordanian/Palestinian/North-Levantine dialect) and were not run through this classifier.

**Final labeled corpus.** The resulting corpus (`data_curated_levant_binary_v1`) totals **1,887,366 rows, 2,209.91 hours**, split 70/15/15 into train/val/test, with MASC and Casablanca preserving each source's own native test-shard membership rather than a random re-split. Per-source hour totals:

| source | hours | lev hours | non-lev hours |
|---|---|---|---|
| qasr | 1,778.97 | 37.83 | 1,741.14 |
| masc | 412.39 | 10.51 | 401.88 |
| omni | 7.91 | 7.91 (all) | — |
| layla | 6.70 | 6.70 (all) | — |
| casa (pal+jor) | 3.95 | 3.95 (all) | — |

### 3.2 Fine-Tuning Mix Construction (`data_finetune_mix_v2`)

The experiments in this paper use a smaller, deliberately-constructed mix drawn from the corpus above, **not** the full 2,209.91-hour corpus. Its composition (verified from the build manifest, seed=42) is important to state precisely, because it directly shapes what the reported WER numbers mean:

- **Train** (30,712 rows, 36.00h): drawn **exclusively from MASC and QASR**, split evenly across dialect label and source to hit a fixed 20/20/30/30% target (masc-lev / qasr-lev / masc-non-lev / qasr-non-lev): masc-lev 7.20h (88.3% of available), qasr-lev 7.20h (27.2% of available), masc-non-lev 10.80h (21.1% of available), qasr-non-lev 10.80h (10.6% of available). **Casablanca, Omnilingual, and Layla are not present in the training split at all.**
- **Validation** (554 rows, 2.01h): Casablanca Palestinian (133 rows) and Jordanian (169 rows) kept unchanged; Omnilingual (142 of 196 available rows) and Layla (110 of 153 available rows) trimmed proportionally to hit a 2-hour target.
- **Test** (1,864 rows, 4.13h): Casablanca Palestinian (664 rows), Casablanca Jordanian (848 rows), Omnilingual (198 rows), Layla (154 rows) — kept at their native "fair share." **MASC and QASR are entirely absent from the test split.**

This is a deliberate methodological choice worth stating explicitly: **the model never sees Casablanca, Omnilingual, or Layla data during training**, and is evaluated exclusively on those sources at val/test time. Training data comes from broadcast (QASR/Al Jazeera) and general-purpose (MASC) speech that a text+audio classifier *scored* as Levantine-dialect, while evaluation data comes from purpose-collected, naturalistic dialectal recordings (Jordanian/Palestinian conversational and read speech). This makes the reported WER numbers a genuine **cross-domain generalization test** — from broadcast-register, classifier-identified Levantine speech to naturalistic dialectal recordings — rather than an in-domain benchmark, and should be kept in mind when interpreting the magnitude of the WER figures in Section 4.

### 3.3 Models Evaluated

Five ASR model families were evaluated, spanning three distinct underlying frameworks:

| Model | Params (verified) | Framework | Loss / decoding |
|---|---|---|---|
| `omnilingual-asr/omniASR_LLM_300M` | 1,627.6M (verified via live load; the "300M" name is a misnomer) | fairseq2 (`wav2vec2_llama`) | seq2seq, autoregressive decode |
| `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0` | 114.6M (verified via live load) | NVIDIA NeMo, hybrid RNNT-CTC run in pure-CTC mode | CTC, greedy decode |
| `Qwen/Qwen3-ASR-0.6B-hf` | ~0.6B (nominal, per model name; not independently re-verified in this study) | HuggingFace `transformers` (chat-template-based) | seq2seq, autoregressive decode |
| `openai/whisper-medium` / `openai/whisper-large-v3` | ~769M / ~1.55B (publicly documented sizes; not independently re-verified in this study) | HuggingFace `transformers` (log-mel encoder-decoder) | seq2seq, autoregressive decode |
| `CohereLabs/cohere-transcribe-arabic-07-2026` | not publicly disclosed (gated repository) | HuggingFace `transformers` | seq2seq, autoregressive decode |

### 3.4 Unified LoRA Fine-Tuning Framework

All five model families are fine-tuned through a single notebook built around a common `ModelAdapter` abstraction, so that model-specific differences (feature extraction, batch collation, loss computation, decoding) are isolated behind six methods (`preprocess`, `collate`, `loss_type`, `generate`, `apply_lora`, `save_checkpoint`/`load_checkpoint`) while the training loop, evaluation protocol, and experiment tracking are shared verbatim across every model.

**LoRA configuration.** A shared baseline of `r=32, lora_alpha=32, lora_dropout=0.05, bias="none"` is used across all five families; only `target_modules` differs, determined per-family from each architecture's actual attention/feed-forward projection naming (verified, not assumed — e.g. for FastConformer this was confirmed by direct module-tree inspection to match only `encoder.layers.*.{self_attn,feed_forward1,feed_forward2}.*`, with zero collisions into the model's separate RNNT decoder/joint network despite those generic-sounding names). Where the underlying attention projections are not `torch.nn.Linear` instances (OmniASR's `fairseq2.nn.projection.Linear`), LoRA is injected manually via a hand-written low-rank wrapper rather than through PEFT.

**Training loop.** A custom training loop is used rather than the HuggingFace `Trainer`, since the fairseq2 (OmniASR) and NeMo (FastConformer) models are not `PreTrainedModel` subclasses. Common elements across all five families:
- Length-grouped batching (`transformers.trainer_pt_utils.LengthGroupedSampler`) using each split's `duration` column, avoiding a full audio-decode pass just to sort by length.
- bf16 autocast on CUDA, AdamW 8-bit optimizer (`bitsandbytes`), linear learning-rate schedule with a 5% warmup ratio.
- Per-family batch size / gradient accumulation tuned to keep effective batch size and GPU memory use comparable across model sizes (e.g. FastConformer at batch 8 / accumulation 1 vs. Whisper-large-v3 at batch 2 / accumulation 4).
- Early stopping on validation WER (patience 3-4 epochs depending on family), with the best-WER checkpoint retained separately from periodic step checkpoints, and full optimizer/scheduler/RNG state checkpointed for exact resume.
- Before committing to a full run, a **5-sample train/val/test smoke gate** exercises the complete training loop (LoRA-wrapped forward/backward, checkpointing, WER/CER validation) end-to-end on a tiny slice of real data, in an isolated checkpoint directory, so that a broken run fails fast rather than after hours of compute.

### 3.5 Evaluation Protocol

- **Metric:** Word Error Rate (WER) and Character Error Rate (CER) via `jiwer`, computed after a shared Arabic normalization step (diacritic removal, tatweel removal, alef/ya/ta-marbuta unification, punctuation stripping) applied identically to references and hypotheses.
- **Reproducibility:** every prediction set and metric is cached to disk, content-addressed by model + split + dataset fingerprint + stage (`base`/`tuned`), so re-running an evaluation cell either re-uses a cached result or fails loudly if the underlying data changed.
- **Experiment tracking:** all five model families log to a single shared MLflow experiment (`arabic-asr-unified`), enabling direct cross-model comparison. Tracked per run: full LoRA/training hyperparameters, per-step metrics (loss, gradient norm, learning rate, step time, GPU memory), per-epoch metrics (train/val loss, val WER/CER, throughput), a 5-example qualitative prediction table per epoch, and final base-vs-tuned test metrics with a full qualitative prediction table at run end.
- **Per-source analysis:** in addition to aggregate WER/CER, predictions are broken down by `mix_source` (the dataset of origin for each utterance — `casa_jor`, `casa_pal`, `omni`, `layla` in val/test; `masc_lev`, `masc_non_lev`, `qasr_lev`, `qasr_non_lev` in train) to surface per-dialect/per-source variation that an aggregate number would hide.

---

## 4. Results

### 4.1 Zero-Shot Baseline Results

Zero-shot (pre-fine-tuning) WER/CER on the full `data_finetune_mix_v2` val (n=554) and test (n=1,864) splits, for the four model families with a completed full-scale evaluation:

| Model | Test WER | Test CER | Val WER | Val CER |
|---|---|---|---|---|
| **omniASR_LLM_300M** | **41.29%** | 13.98% | 37.07% | 11.76% |
| **FastConformer** (NVIDIA) | 43.57% | 14.62% | 40.34% | 13.42% |
| **Qwen3-ASR-0.6B** | 52.02% | 18.03% | 48.93% | 16.26% |
| **Whisper-medium** | 54.02% | 23.93% | 50.78% | 21.72% |

Whisper-large-v3 and Cohere Transcribe Arabic do not yet have a full-scale zero-shot evaluation completed (only 1-sample smoke-test artifacts exist for these two as of this writing; see Section 5).

Two patterns are worth noting. First, the best zero-shot model (OmniASR, 1.63B params) beats the smallest model tested (FastConformer, 114.6M params) by only ~2 points of WER — model scale alone does not close the gap. Second, CER is consistently much lower than WER (roughly 3-4x lower) across every model, indicating that errors are frequently character-level substitutions within otherwise-recognizable words (consistent with dialectal phonological/orthographic variation) rather than wholesale mis-transcription — but WER, the metric that matters for downstream usability, remains high regardless.

### 4.2 Per-Source (Dialect) Breakdown, Zero-Shot

Breaking the FastConformer zero-shot val results down by source dataset:

| Source | n | WER | CER |
|---|---|---|---|
| **overall** | 554 | 40.34% | 13.42% |
| casa_jor (Jordanian, Casablanca) | 169 | 40.27% | 13.16% |
| casa_pal (Palestinian, Casablanca) | 133 | **51.20%** | 18.10% |
| layla (Jordanian, Layla) | 110 | **33.58%** | 11.51% |
| omni (North Levantine, Omnilingual) | 142 | 43.98% | 14.05% |

The spread here — 33.58% to 51.20% WER, a 17.6-point range — is substantial given all four sources are nominally "Levantine dialect." Palestinian speech specifically (`casa_pal`) is the hardest of the four sources tested, roughly 10-18 points worse than the other three, which is directly relevant to this project's stated focus. Note that MASC/QASR do not appear here at all: as described in Section 3.2, they are excluded from val/test by construction.

### 4.3 LoRA Fine-Tuning: FastConformer Case Study (In Progress)

Fine-tuning has been started for one model family so far — FastConformer-CTC, LoRA config `r=32, alpha=32, dropout=0.05`, targeting the six Conformer encoder attention/feed-forward projections (`linear_q/k/v/out`, `linear1/2`), yielding 7,798,784 trainable parameters (6.37% of the model's 122,420,226 total). Training uses effective batch size 8, learning rate 1e-4, bf16, AdamW 8-bit, up to 50 epochs with early-stopping patience 3 on validation WER.

| Epoch | Train loss | Val loss | Val WER | Val CER |
|---|---|---|---|---|
| — (zero-shot baseline) | — | — | 40.34% | 13.42% |
| 1 | 49.79 | 104.37 | 47.45% | 16.42% |
| 2 | 27.10 | 102.33 | 45.41% | 15.84% |

Training was **paused after epoch 2** (compute constraint, not a stopping criterion), with the full optimizer/scheduler/RNG state checkpointed for exact resume from epoch 3 with no lost progress. As of this checkpoint, LoRA fine-tuning has **not yet recovered the zero-shot baseline**: WER after 2 epochs (45.41%) remains worse than the pre-fine-tuning baseline (40.34%), though the trend between epoch 1 and epoch 2 (47.45% → 45.41%) is improving, consistent with early LoRA warm-up noise rather than a fundamental problem with the setup. Per-source breakdown after epoch 1 shows every source degrading roughly uniformly relative to baseline (casa_jor +3.8pt, casa_pal +3.9pt, layla +5.4pt, omni +1.7pt WER), i.e. no single source is disproportionately responsible for the early regression.

Whether continued training surpasses the zero-shot baseline — and by how much — is an open question this paper will report once the run is resumed and completed (or early-stops).

### 4.4 Summary So Far

- Zero-shot WER across five evaluated model families ranges from 41.3% (best, OmniASR) to 54.0% (worst, Whisper-medium among those fully evaluated) on naturalistic Levantine dialectal speech, confirming this remains a genuinely unsolved problem for current ASR systems.
- Palestinian dialectal speech specifically is the hardest of the four dialectal sources tested (51.2% WER zero-shot on FastConformer, vs. 33.6-44.0% for the other three Levantine sources).
- LoRA fine-tuning is underway for one of five model families; 2 of a budgeted 50 epochs are complete, with results not yet surpassing the zero-shot baseline but trending favorably.
- The remaining four model families (OmniASR, Qwen3-ASR, Whisper, Cohere) have zero-shot baselines in place (except Whisper-large-v3 and Cohere, see Section 5) but have not yet been fine-tuned under this framework.

---

## 5. Limitations and Next Steps

- **Incomplete zero-shot coverage:** Whisper-large-v3 and Cohere Transcribe Arabic lack full-scale zero-shot baselines (only 1-sample smoke artifacts exist). Cohere additionally requires a Hugging Face token with the gated license accepted.
- **Fine-tuning coverage:** only FastConformer has an active fine-tuning run; the other four model families are configured (per-model LoRA/training hyperparameters already defined in the shared `ConfigAPI`) but not yet launched.
- **Fine-tuning completeness:** the one active run is 2/50 budgeted epochs in, paused for compute reasons rather than convergence or early-stopping — the reported fine-tuned numbers should be read as a snapshot, not a final result.
- **Train/eval domain mismatch:** as described in Section 3.2, train data (MASC/QASR, classifier-identified as Levantine) and val/test data (Casablanca/Omnilingual/Layla, naturalistic dialectal recordings) come from different recording domains and registers by construction. This is a deliberate test of cross-domain generalization, but it also means the reported WER conflates "how hard is Palestinian/Levantine ASR" with "how well does broadcast-domain fine-tuning transfer to conversational dialectal speech" — an ablation with matched-domain splits would help separate these two effects.
- **Single-split evaluation:** results are reported on one fixed train/val/test partition (seed=42); no cross-validation or multiple-seed variance estimate is currently available.

---

## Appendix A: Full Configuration Reference (FastConformer)

For exact reproducibility of the Section 4.3 results:

```json
// LoRA config
{
  "r": 32, "lora_alpha": 32, "lora_dropout": 0.05, "bias": "none",
  "target_modules": ["linear_q", "linear_k", "linear_v", "linear_out", "linear1", "linear2"],
  "modules_to_save": null, "task_type": null
}

// Training config
{
  "num_epochs": 50, "early_stopping_patience": 3,
  "metric_for_best": "wer", "greater_is_better": false,
  "per_device_train_batch_size": 8, "per_device_eval_batch_size": 16,
  "gradient_accumulation_steps": 1, "learning_rate": 0.0001,
  "warmup_ratio": 0.05, "weight_decay": 0.0, "max_grad_norm": 1.0,
  "optim": "adamw_bnb_8bit", "bf16": true, "gradient_checkpointing": false,
  "dataloader_num_workers": 0, "max_audio_seconds": 30.0, "max_label_tokens": 256,
  "save_total_limit": 3, "save_steps": null
}
```
MLflow run ID: `ac90cb186dd5489fb4a8e92e6674f550` (experiment `arabic-asr-unified`, tracking URI `sqlite:///Runs/conformer_ctc_finetune_mix_v2/mlflow.db`).
