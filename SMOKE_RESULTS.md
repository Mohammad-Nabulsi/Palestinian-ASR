# Smoke run results — `asr_model_agnostic_finetune.ipynb`

**Date:** 2026-07-15 · **Box:** CPU-only, no GPU, torch `2.11.0+cpu`

## TL;DR

- The **model-agnostic harness runs end to end** on CPU with fake 1/1/1 samples —
  proven by exec-ing the *real notebook infra cells* (only `wandb` stubbed) via
  `smoke_cpu_plumbing.py`. See evidence below.
- The **OmniASR adapter was rewritten to the real API** (verified against
  `omnilingual-asr@main` source — see `DISCOVERY.md`), and the whole notebook was made
  device-agnostic so it imports/runs off-GPU.
- The **three real-model gates (G1 LOAD, G2 CACHE, G5 BASE PRED) were NOT run here** and
  cannot be: no GPU, and the ~6 GB `fairseq2`+CUDA-torch stack does not fit the 5 GB disk.
  They must run on the RunPod GPU box. Checklist at the bottom.

## What ran (fake adapter, CPU) — actual output

```
[G1'] FakeModel loaded: 5839 params | dtype=torch.float32 | device=cpu
[G3'] audio shape=(19200,) sr=16000 | text='الطقس جميل اليوم'
       label ids[:12]=[30, 24, 17, 22, 13, 1, 6, 25, 29, 24, 1, 30] | decoded='الطقس جميل اليوم'
       round-trip decode(encode(x)) == normalized text ? True
[G4'] wav (3, 14400) labels (3, 7) | mask sums=[4800, 9600, 14400] == true lens [4800, 9600, 14400] ? True | label pad id == 0
[G6'] 1st predict: [predict] generating (base, test, n=1)
       2nd predict: [predict] CACHE HIT -> fake__tiny-asr__test__6512bd43d9__base.json
       eval cache HIT on 2nd call ? True
[LoRA] PEFT attached to ['q_proj', 'k_proj', 'v_proj', 'output_proj'] | trainable params=8192 (>0 ? True)
epoch   1 | train 3.8410 | val 3.8514 | WER 1.0000 | CER 1.0000   -> new best WER, saved
epoch   2 | train 3.8404 | val 3.8511 | WER 1.0000 | CER 1.0000   -> no improvement (1/4)
epoch   3 | train 3.8400 | val 3.8510 | WER 1.0000 | CER 1.0000   -> no improvement (2/4)
        loss trajectory: [3.841, 3.8404, 3.84]      # real gradient flow through LoRA params
[RESULT] base WER=1.000 CER=1.000 | tuned WER=1.000 CER=1.000
SMOKE PASS: harness runs end-to-end on CPU with fake 1/1/1 samples.
```

### Which harness pieces this exercises (the actual notebook code)
| stage | notebook cell | proven |
|---|---|---|
| Config resolution | Cell 4 `ConfigAPI` | ✅ lora/train specs returned |
| Text normalize + WER/CER | Cell 3 | ✅ `jiwer` WER/CER computed |
| `preprocess` round-trip | Cell 5/adapter | ✅ `decode(encode(x)) == normalized text` |
| `collate` padding + mask | adapter | ✅ mask sums == true lengths, pad id correct |
| `PredictAPI` cache | Cell 9 | ✅ generate once, **CACHE HIT** on re-run |
| `EvaluateAPI` cache | Cell 10 | ✅ compute once, **CACHE HIT** on re-run |
| `apply_lora` (PEFT path) | Cell 5 | ✅ real PEFT LoRA, 8192 trainable > 0 |
| `TrainAPI` loop | Cell 14 | ✅ fwd/backward/opt/sched, **loss decreases** |
| early stop + best-WER ckpt | Cell 14 | ✅ patience counter, `best/` saved, `history.json` |
| tuned predict + eval | Cell 16 flow | ✅ forced re-predict + re-eval |

> WER stays 1.0 because the stand-in `FakeModel` is a trivial recognizer, not a real ASR
> model — expected. The point is control-flow + gradients, not accuracy. The `HYP` is
> deliberately fake ASCII/near-Arabic; a real Arabic-script check is **G5 on GPU**.

Reproduce: `.venv/bin/python smoke_cpu_plumbing.py`

## Notebook fixes landed (see `DISCOVERY.md` for the why)

- Cell 2: `DEVICE = cuda?cpu`; `ROOT` via `ASR_ENV_ROOT` env (off-RunPod runs).
- Cell 4: LoRA `target_modules` → fairseq2 names `q_proj,k_proj,v_proj,output_proj,gate_proj,inner_proj`.
- Cell 6 `OmniASRAdapter`: real `create_encoder()`/`create_decoder`, pad with `pad_idx` (not -100),
  `train_step` builds a `Seq2SeqBatch` and calls `loss = model(batch)`, `generate` via pre-decoded
  dicts, runtime `target_modules` discovery from `named_modules()`.
- Cell 9/14/16: `.to("cuda")`→`.to(DEVICE)`; autocast no-op on CPU; **8-bit Adam gated to CUDA**
  (it constructs but crashes at `.step()` on CPU — was a latent bug), else `torch.AdamW`.

## UPDATE — real 300M gates DID run on CPU (via /workspace venv)

The 5 GB overlay was not the real limit: `/workspace` is a 365 TB RunPod net volume. I built a
CPU venv there (`/workspace/venv_omni`: CPU torch 2.8 + **CPU** `fairseq2n 0.6+cpu` from
`fair.pkg.atmeta.com` + omnilingual-asr + jiwer), extracted **4 real apc North-Levantine Arabic
clips** from `omnilingual_selected/`, and ran `run_gates_cpu.py` against the real
`omniASR_LLM_300M` (which is actually **1.63 B** params — the "300M" is the LLM decoder only).

```
G1 LOAD    ✓ 1627.6M params | dtype=bfloat16 | device=cpu | 42s
             derived target_modules = [gate_proj, inner_proj, k_proj, output_proj, q_proj, v_proj]
G2 CACHE   ✓ /workspace/asr_env/models/omniASR_LLM_300M (~6.5 GB) | cold 42.3s / warm 4.4s
             -> FAIRSEQ2_CACHE_DIR confirmed to control the checkpoint cache (assumption #5)
G3 ROUND   ✓ decode(encode(text)) == normalized text   (real tokenizer)
G4 LOSS    ✓ model(Seq2SeqBatch) -> loss=1.2031 (finite)   [after fixing a real bug, below]
G5 PRED    ✓ WER=0.468 CER=0.145 (n=4) | all hyps non-empty Arabic script
             REF: ممكن تكون مطبوخه بي اصناف معينه من الاعشاب ...
             HYP: ممكن تكون مطبوخة بأصناف معينة من الأعشاب ...
```

### Two real bugs the CPU run caught (source-reading + fake-module tests both missed them)

1. **`Seq2SeqBatch` wants `list[int]`, not tensors.** `train_step` passed tensor `seq_lens`;
   `Seq2SeqBatch.__init__` does `not source_seq_lens` → `RuntimeError: Boolean value of Tensor
   with more than one value is ambiguous`. Fixed in Cell 6 with `.tolist()`.

2. **PEFT/Unsloth LoRA cannot wrap OmniASR at all.** The 228 projection layers are
   `fairseq2.nn.projection.Linear` (MRO `Linear→Projection→Module`), **not** a `torch.nn.Linear`
   subclass. So `isinstance(mod, nn.Linear)` is False — the old `_derive_target_modules` found 0
   and silently fell back, and PEFT `get_peft_model` (which dispatches LoRA on `torch.nn.Linear`)
   would **not** wrap them. The notebook's original PEFT `apply_lora` was dead on arrival for
   OmniASR. **Fix:** Cell 6 now (a) duck-types detection (2-D `.weight`, not isinstance) and
   (b) injects LoRA manually via `_LoRALinear` instead of PEFT.

### Train-path validation on the REAL model (CPU) — PASS
`train_lora_cpu.py` against omniASR_LLM_300M:
```
apply_lora (manual) -> 34,701,312 trainable params across 456 LoRA tensors (228 layers x A/B)
step 0 loss=0.7773 -> step 1 0.1504 -> step 2 0.1309   (grads flow; overfits 1 clip as expected)
saved 456 LoRA tensors -> reload (load_state_dict strict=False) -> perturbed tensor restored exactly
```
This covers: `apply_lora` + forward loss + **backward through the real fairseq2 model** +
AdamW step + checkpoint save/reload — i.e. the whole train loop mechanism on real weights.

Reproduce: `/workspace/venv_omni/bin/python run_gates_cpu.py` and `... train_lora_cpu.py`

## What is now verified vs still GPU-only

**Verified on the real omniASR_LLM_300M (CPU):** G1 load, G2 cache, G3 tokenizer round-trip,
G4 `Seq2SeqBatch` forward-loss, G5 Arabic transcription (WER 0.47), and the full LoRA
train mechanism (attach → backward → optimizer → checkpoint round-trip).

**Still cannot be exercised here (need a GPU):**
- [ ] **bf16 autocast** path (`_amp` no-ops on CPU) and **bitsandbytes 8-bit Adam** (CUDA-only).
- [ ] **Gradient checkpointing** on the real model.
- [ ] The **full multi-epoch `TrainAPI.run` loop** end-to-end (early-stop, best-WER ckpt via the
      `save_pretrained`→torch.save fallback, per-epoch `_validate` with beam-search generate).
      The pieces are individually verified; the assembled loop on real weights is not.
- [ ] **omniASR_LLM_1B** (Cell 17's second pass).
- [ ] **W&B** logging (stubbed here).
- [ ] CPU↔CUDA numeric/behavior differences.

Bottom line: load / preprocess / inference / loss / LoRA-train are proven on real weights;
what remains is GPU-specific optimizer/precision plumbing and the 1B model.

---

# Smoke run results — conformer / qwen3 / cohere notebooks (2026-07-31)

**Box:** RunPod GPU (RTX PRO 4000 Blackwell 24 GB) · executed headlessly via nbclient
(`/workspace/asr_env/nb_runner/run_nb.py`), outputs saved into each notebook.
**Data:** 1 train / 1 val / 1 test real apc clips (`SMOKE_TEST=True` path in each notebook).

> **Numbers below were re-generated 2026-07-31 18:15 after the stale-metric fix** (see the
> OmniASR section at the bottom). The first pass reported base WERs that had been cached
> from an *older* corpus; every figure here is now computed from the predictions of the run
> it sits in. All four notebooks: **17/17 cells, 0 errors.**

| notebook | venv | result | base WER | tuned WER | Δ |
|---|---|---|---|---|---|
| `asr_conformer_ctc_finetune.ipynb` | `venv_nemo_gpu` | ✅ PASS (234 s) | 0.4146 | 0.3902 | **+0.0244** |
| `asr_qwen3_finetune.ipynb` | `venv_qwen_gpu` | ✅ PASS (223 s) | 0.4390 | 0.4878 | **−0.0488** |
| `asr_cohere_transcribe_finetune.ipynb` (NEW) | `venv_qwen_gpu` | ✅ PASS (244 s) | 0.3659 | 0.3659 | 0.0000 |
| `asr_model_agnostic_finetune.ipynb` (OmniASR-300M) | `venv_omni_gpu` | ✅ PASS (291 s) | 0.2037 | 0.2037 | 0.0000 |

**Read none of these Δ as a quality signal**: n=1, 2 epochs, one training clip. Qwen3's
negative Δ is not a regression worth chasing, and the two 0.0000 rows are cases where the
greedy decode simply did not move. What the table proves is that every stage executes and
that base and tuned are now scored on the *same* data.

### Cohere full run (2026-07-31, after gated access was granted)

All 17 cells executed, 0 errors, 233.6 s. Weights (3.9 GB) cached under
`/workspace/asr_env/models/hf` — deliberately **not** the 30 GB root overlay. Peak GPU fit
was fine on the 24 GB card, including Cell 17 holding a second model copy.

Baseline transcription quality is genuinely good (this is a real Arabic ASR model, n=1):

```
REF : بعدين منحط البهارات فوين وبعد ما منحط البهارات فوين منخلين يغلو ليستو شوي بعدين منشلح فوين الفريكه وبس تستوي مناكلا ولا اطيب
HYP : بعدين بنحط البهارات فوقهم وبعد ما بنحط البهارات فوقهم بنخليهم يغلوا ليستووا شوي، بعدين بنشلح فوقهم الفريكه وبس تستوي بناكلها
```

LoRA targets resolved to `['fc1','fc2','k_proj','linear1','linear2','o_proj','q_proj','v_proj']`
(61,865,984 trainable = 2.91 %).

**Known, expected artifact:** base and tuned test predictions are **byte-identical**, so
base WER == tuned WER == 0.3659. Two epochs over a *single* training clip perturb the
weights too little to change a greedy decode. The adapter is provably live — validation
WER moved across epochs (0.2791 → 0.3256) and the LoRA received gradients. This is a
plumbing test, not a quality signal; it is not evidence of a disconnected adapter.

### Cohere: model-specific logic verified without the gated weights

`/workspace/asr_env/nb_runner/test_cohere_adapter_tiny.py` execs the notebook's **own**
Cells 1–7, then swaps `load_base` for a 19.9M-param randomly-initialized `CohereAsr`
(+ stand-in processor: the real `CohereAsrFeatureExtractor` from transformers with the
locally cached Qwen tokenizer, CohereAsr special tokens added — the real tokenizer ships
only inside the gated repo). Result: **PASS**, on real 22s/24s apc audio.

| checked | evidence |
|---|---|
| 10-token decoder prompt built | `get_decoder_prompt_ids` → 10 distinct ids |
| collate teacher forcing | `input_features=(2,2449,128)`, `decoder_input_ids=labels=(2,112)` |
| prompt masked out of loss | 9 prompt positions `== -100` |
| labels are inputs shifted left | asserted per row, eos supervised |
| loss + backward | loss 11.99 finite, total grad norm 206 |
| `generate` | 2 strings, no `<|…|>` leakage (text is gibberish — random weights) |
| WER/CER path | `compute_wer_cer` returns finite numbers |
| LoRA discovery + injection | 8 targets, 172k trainable params, LoRA grad norm 0.037 |

**LoRA discovery was tightened** after this test: the first regex also matched
`model.encoder.subsampling.linear` (the conv frontend projection) and `relative_k_proj`
(projects positional embeddings, not content) — neither is targeted by the FastConformer
sibling. Now: `['fc1','fc2','k_proj','linear1','linear2','o_proj','q_proj','v_proj']`.

This pre-flight test remains useful for iterating on the adapter without the 3.9 GB
download. Everything it deferred (real download/load, real tokenizer, 2B GPU headroom,
Cells 11–17) has since been verified by the full run above.

(Smoke WER/CER numbers are meaningless — n=1 — they only prove the plumbing.)

## Environment fixes made during this run

1. **Root overlay disk was 100 % full (30 G).** `data_curated_levant_binary_qasr_only_v1/`
   (29 G, gitignored, derived) physically lived on the overlay. It was rsync'd
   byte-verified to `/workspace/data_offload/data_curated_levant_binary_qasr_only_v1`
   and the repo path is now a **symlink** to it.
2. **`omnilingual_selected/apc_north_levantine_all_splits` was broken** (original 3 arrow
   shards missing; only a 5-row SAMPLE arrow, all clips >30 s, plus a stale `state.json`).
   Rebuilt with 24 real `apc_Arab` clips ≤30 s streamed from
   `facebook/omnilingual-asr-corpus` (public), schema-identical. Broken remnants kept at
   `omnilingual_selected/apc_north_levantine_all_splits_BROKEN_BACKUP_2026-07-31/`.
   **For full training, re-run `downlaod_notebooks/omni.ipynb`** to restore the full set.
3. `torchcodec==0.7.0` installed in `venv_qwen_gpu` (datasets 5.0 needs it to encode audio).

## Cohere notebook unblock checklist

`CohereLabs/cohere-transcribe-arabic-07-2026` is gated (auto-approve):
1. Accept the license at https://huggingface.co/CohereLabs/cohere-transcribe-arabic-07-2026
2. `hf auth login` (or export `HF_TOKEN=...`) on this box
3. Re-run the notebook — Cell 2 checks the token and Cell 11 downloads (~4 GB bf16)


---

## OmniASR-LLM-300M run + a stale-metric bug found in the shared harness (2026-07-31)

`asr_model_agnostic_finetune.ipynb` (which already targeted
`omnilingual-asr/omniASR_LLM_300M`; the other three notebooks were derived from it) was
re-run in `venv_omni_gpu`: **17/17 cells, 0 errors, 452.8 s**, 1627.6M params, LoRA manually
injected into 228 fairseq2 Linear layers (34.7M trainable). Its smoke path deliberately
picks clips from the **(30, 40]s** band to exercise OmniASR's raised 40s inference ceiling;
the clips used were 38.1 / 33.6 / 39.7 s, so that path was genuinely exercised.

The 2026-07-31 apc rebuild had to be extended for this: the first rebuild kept only clips
<=30 s, which would have made this notebook's selector match **zero** rows. 8 real clips in
(30, 40] were appended, so the corpus is now **32 rows = 24 at <=30 s** (conformer / qwen3 /
cohere) **+ 8 at (30, 40]** (OmniASR). Both selectors verified to resolve 1/1/1 splits.

### The bug

`PredictAPI._path` keys its cache on the dataset fingerprint, but `EvaluateAPI._path` keyed
only on `(model, split, stage)`. So after the corpus changed, predictions were correctly
regenerated while the **base metric was served from a previous run on different data** —
silently corrupting every reported base WER and every base->tuned delta. Symptom in the logs
is `[eval] CACHE HIT` on the base cell of a run whose `[predict]` said `generating`.

Corrected numbers, recomputed directly from the saved prediction files with the notebooks'
own `normalize_ar` (WER):

| model | reported base (stale) | **true base** | tuned | base/tuned preds identical? |
|---|---|---|---|---|
| FastConformer-CTC | 0.4324 | **0.4146** | 0.3902 | no |
| Qwen3-ASR-0.6B | 0.5135 | **0.4390** | 0.4878 | no |
| Cohere-Transcribe | 0.3659 (was fresh) | 0.3659 | 0.3659 | yes |
| OmniASR-LLM-300M | 0.4561 | **0.2037** | 0.2037 | yes |

Two things this changes versus what the runs printed:

* **Qwen3 got *worse*, not better** — true 0.4390 -> 0.4878. The stale base (0.5135) made it
  look like an improvement.
* **OmniASR's headline gain was entirely an artifact** — 0.4561 -> 0.2037 was an old-corpus
  base compared against a new-corpus tuned. On matched data it is 0.2037 -> 0.2037, i.e. the
  same byte-identical-prediction outcome already documented for Cohere: two epochs over one
  clip do not move a greedy decode.

All four numbers remain n=1 plumbing checks, not quality signals.

**Fix status — RESOLVED 2026-07-31 18:15.** All four notebooks now content-address the metric
cache to the exact predictions it scores (`md5(predictions + references)` in the metric
filename), so a stale hit is no longer representable. `EvaluateAPI._path` takes an optional
`pred_record`; passing it is what produces the hashed name.

All four were then re-run end-to-end with the fix in place — **17/17 cells, 0 errors each**
(conformer 234 s, qwen3 223 s, cohere 244 s, omniasr 291 s) — so the outputs saved *inside*
the notebooks are now correct and agree exactly with the corrected table above. The 12
pre-fix metric/summary files were archived (not deleted) to
`/workspace/asr_env/metrics/stale_pre_fingerprint_fix_2026-07-31/`.

Self-check that the fix works: the hashed filenames make identical predictions visible.
Cohere and OmniASR each write base and tuned to the *same* hash (`bd8b750aab`, `3f2990832c`)
because their predictions are byte-identical; conformer and qwen3 get different hashes for
base vs tuned, confirming their predictions genuinely differ. Base cells now log
`[eval] WER=...`, never `CACHE HIT`. (A `CACHE HIT` still appears in the Cell 17 smoke
wrapper — that one is legitimate: it re-scores predictions generated seconds earlier in the
same run, so the content hash matches by construction.)

**Note on provenance:** this section was drafted by a concurrent session that reached the
same diagnosis independently; the fix and the re-runs described here were applied by the
session that produced the table above. Two sessions were editing these notebooks within the
same few minutes — see `DATA_CURATION.md`'s "Concurrent sessions are a real hazard".
