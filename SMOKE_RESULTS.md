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
