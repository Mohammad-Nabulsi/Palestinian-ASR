# HANDOFF — OmniASR LoRA fine-tuning notebook

_Last updated: 2026-07-16. Read this first next session._

Target: validate/repair `asr_model_agnostic_finetune.ipynb` (model-agnostic ASR LoRA harness)
for `omnilingual-asr/omniASR_LLM_300M` → then `_1B`, language `arb_Arab`.

## 2026-07-16 UPDATE #2 — 30s→40s audio limit + Qwen3-ASR-0.6B notebook

**Audio length limit raised 30s→40s (verified empirically on GPU).** Question: can OmniASR
train on >30s segments? Answer: **training YES, but inference caps at 40s.** Evidence
(`scratchpad/test_long_audio.py`): a 41.5s clip trained (fwd+bwd) at 11.5GB peak, an 87.7s clip
trained at 20.8GB peak — no hard training limit (LLaMA decoder ctx is 8192). BUT
`pipeline.transcribe` (used by `generate()` for BOTH validation and test) runs `assert_max_length`
and **raises for anything >40s** (`MAX_ALLOWED_AUDIO_SEC=40` in
`omnilingual_asr/models/inference/pipeline.py`) — the 87.7s clip crashed there. So the old
`max_audio_seconds=30` was an arbitrary under-limit; **raised it to 40** (the model's true
inference ceiling) rather than removing it. Removing entirely would crash validation/test on
>40s clips. For >40s data you'd need chunked/streaming decode at eval time. The smoke test now
deliberately picks clips from the (30,40]s band and re-ran green end-to-end (base WER 0.456 →
tuned 0.421, `[prep] kept 1, dropped 0` — the >30s clip was NOT dropped). Cells changed:
`TrainConfigSpec.max_audio_seconds` (Cell 4) and `load_splits` smoke filter (Cell 8).

**New sibling notebook: `asr_qwen3_finetune.ipynb` for `Qwen/Qwen3-ASR-0.6B-hf`.** Same pipeline
structure (ConfigAPI / ModelAdapter / PredictAPI / EvaluateAPI / TrainAPI), different model. Key
differences from OmniASR, all real (not guessed): Qwen3-ASR is **transformers-native**
(`Qwen3ASRForConditionalGeneration`, needs transformers ≥5.13 — env has fairseq2-pinned 4.57.6,
so Qwen lives in a **separate venv** `/workspace/venv_qwen_gpu` with transformers 5.14.1).
It is an **instruction/chat model** — training MUST go through
`processor.apply_chat_template(chat, tokenize=True, return_dict=True, output_labels=True)` (audio +
transcript in one user turn), then `loss = model(**inputs).loss`. Inference via
`processor.apply_transcription_request(audio=..., language="ar")` → `model.generate` →
`processor.decode(..., return_format="transcription_only")`. Its Qwen3 decoder projections are
real `torch.nn.Linear`, so **standard PEFT LoRA works** (`q/k/v/o_proj, gate/up/down_proj`) and
`save_pretrained`/`load_adapter` round-trip natively — no manual `_LoRALinear` injection or
`torch.save` fallback needed (the big OmniASR headache does not apply here). See that notebook +
`scratchpad/qwen_api_check.py` for the verified API.

## 2026-07-16 UPDATE — full notebook verified end-to-end on a real GPU ✅

This box got a GPU (RTX PRO 4500, Blackwell / sm_120, 32GB) since the last session. Ran the
**entire notebook top-to-bottom via `jupyter nbconvert --execute`** with 1 fake train / 1 fake
val / 1 fake test sample (real audio+text from the apc North-Levantine corpus, not Common
Voice) on the real, cached `omniASR_LLM_300M`. Full pass — every stage the notebook is
supposed to do now has real evidence:

```
[load] 1627.6M params | dtype=bfloat16 | device=cuda:0                      (cold ~80-100s over the net volume)
[predict] base test WER=0.4595 CER=0.2609 (n=1) — coherent real Arabic hyp
[lora]  manual injection into 228 fairseq2 Linear layers | trainable=34,701,312
epoch   1 | train 0.3372 | val 0.3087 | WER 0.1930 | CER 0.0990  -> new best, ckpt saved
epoch   2 | train 0.3341 | val 0.3088 | WER 0.1930 | CER 0.0990  -> no improvement (1/4)
[ckpt] loaded best state_dict <- checkpoints/.../best   (adapter.pt, 456 LoRA tensors)
[predict] tuned test WER=0.4595 CER=0.2609 (n=1)
SUMMARY.json written; bottom Cell 17 smoke() loop also reran the same flow -> status=PASS
```

**Everything requested is now proven on GPU, not just CPU-simulated:** load → real-data prep
→ per-epoch train-loss/val-loss/WER/CER printing → best-WER checkpoint selection → manual
LoRA (Unsloth confirmed unusable for this fairseq2 architecture, PEFT also unusable — see
DISCOVERY.md #4; this is by design, not a shortcut) → reload best checkpoint → predict on
test → evaluate saved preds. All artifacts verified on disk: `preds/*base*.json`,
`preds/*tuned*.json`, `metrics/*base*.json`, `metrics/*tuned*.json`, `metrics/*SUMMARY.json`,
`checkpoints/.../best/adapter.pt` (456 tensors, 34.7M params, matches the training printout
exactly), `checkpoints/.../history.json`.

### What changed in the notebook to make this run
- **Cell 2**: `MODEL_NAME` pinned to `omniASR_LLM_300M` (already cached, already gate-verified;
  `_1B` is not cached on this box — would need a fresh multi-GB download, left as a TODO, swap
  back by uncommenting). `WANDB_MODE=disabled` (no W&B account here; the per-epoch print
  statements already satisfy the "show train/val loss + WER/CER" requirement without it).
- **Cell 8 (`load_splits`)**: swapped the Common Voice placeholder for the real North-Levantine
  (Palestinian) Arabic corpus at `omnilingual_selected/apc_north_levantine_all_splits`
  (517 rows total). For `SMOKE_TEST`, filters to clips ≤30s first (median clip in this corpus
  is ~70s, well over `TrainConfigSpec.max_audio_seconds=30` — an unfiltered pick would get
  silently dropped by `TrainAPI._prep`, producing a misleadingly "successful" empty epoch),
  then takes 3 disjoint real rows as the 1 fake train/1 fake val/1 fake test sample.
  **Audio decode gotcha (new, GPU-box-specific):** the `datasets` version here (5.0.0) hard-
  requires `torchcodec` for its built-in `Audio(decode=True)` — for BOTH decode and re-encode,
  so even `.map()` on an `Audio(decode=False)` column fails. Installing torchcodec risked
  clobbering the carefully-pinned cu128 torch build, so audio is decoded manually via
  `soundfile` and materialized into a fresh `Dataset.from_generator(...)`, which gets a plain
  inferred struct type for "audio" instead of the `Audio` feature type — sidesteps torchcodec
  entirely. (Also: the real audio is 48kHz, not 16kHz — already handled by the adapter's
  existing `librosa.resample` fallback.)
- **Cell 17 (bottom duplicate smoke loop)**: restricted to `omniASR_LLM_300M` only (was
  `[300M, 1B]`) for the same not-cached-yet reason as above.

### Environment note — this GPU needs cu128, not cu126
The GPU here is Blackwell (`sm_120`). `torch==2.8.0+cu126` installs fine and reports
`cuda.is_available()==True`, but its kernels **do not include `sm_120`** — it's a silent trap,
not an import error. Reinstalling with `--index-url https://download.pytorch.org/whl/cu128`
(same torch version, cu128 build) fixes it — `torch.cuda.get_arch_list()` then includes
`sm_120` and a real `torch.randn(...).cuda() @ ...` matmul works. `fairseq2n` has a matching
`0.6+cu128` wheel at `fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cu128/`. New venv built at
`/workspace/venv_omni_gpu` (the old `/workspace/venv_omni` is CPU-only, keep for CPU work).
Kernel registered as `omni_gpu` (`python -m ipykernel install --user --name=omni_gpu`).

### Still open (unchanged from before, now lower priority)
- `omniASR_LLM_1B` — not cached, never run (swap `MODEL_NAME` back and let it download).
- W&B logging — intentionally disabled this run, no account configured on this box.
- Full-size (non-smoke) training run on the real apc corpus (currently only 1/1/1 fake
  samples were exercised, per this task's smoke-test scope).

---

## STATUS: what is FINISHED ✅

Everything below is **done and verified on the real omniASR_LLM_300M** (on CPU — see constraints):

| Gate / step | Result |
|---|---|
| G1 LOAD | ✓ real model loads: **1.63 B params**, bf16 (the "300M" card = LLM decoder only) |
| G2 CACHE | ✓ cache at `/workspace/asr_env/models/omniASR_LLM_300M` (6.5 GB); cold 42s / warm 4s; **`FAIRSEQ2_CACHE_DIR` controls it** |
| G3 tokenizer round-trip | ✓ `decode(encode(text)) == normalized text` |
| G4 forward loss | ✓ `loss = model(Seq2SeqBatch)` finite (1.2031) |
| G5 base predict | ✓ real Levantine Arabic, **WER 0.468 / CER 0.145** on 4 real clips |
| LoRA train path | ✓ manual-LoRA attach (34.7M params / 456 tensors) → backward → AdamW → loss 0.78→0.13 → **ckpt save+reload** |
| Notebook health | ✓ 17/17 code cells compile |

**Adapter (Cell 6) is rewritten and correct.** DISCOVERY.md + SMOKE_RESULTS.md hold full evidence.

### Bugs found & fixed (all in the notebook now)
1. **`Seq2SeqBatch` needs `seq_lens` as `list[int]`, not tensors** → `train_step` uses `.tolist()`.
2. **PEFT/Unsloth LoRA cannot wrap OmniASR.** Its 228 projections are
   `fairseq2.nn.projection.Linear` (NOT a `torch.nn.Linear` subclass), so PEFT's dispatch misses
   them and the old `isinstance` derivation found 0. → Cell 6 now does **manual LoRA injection**
   (`_LoRALinear`) + duck-typed detection (2-D `.weight`). **This was the biggest fix.**
3. **8-bit Adam gated to CUDA** (it constructs but crashes at `.step()` on CPU) → falls back to `torch.AdamW`.
4. Device-agnostic: `DEVICE = cuda?cpu`; `ROOT` via `ASR_ENV_ROOT` env; `.to(DEVICE)` everywhere.
5. `create_encoder()` takes **no** lang; labels pad with `pad_idx` (not −100); target names are
   fairseq2 (`q/k/v/output_proj, gate/inner_proj`).

### Assumption ledger (from the task) — all resolved
1 model attr ✓ (`pipeline.model`) · 2 tokenizer ✓ (`pipeline.tokenizer`) · 3 loss **YES** via
`model(Seq2SeqBatch)` (no manual CE) · 4 **fixed** (names + manual LoRA) · 5 **`FAIRSEQ2_CACHE_DIR`
confirmed** · 6 `create_encoder()` no lang.

---

## STATUS: what REMAINS

As of 2026-07-16, items 1, 2, 3 and 6 below are **done** — see the 2026-07-16 UPDATE section at
the top of this file for the real-GPU evidence. Kept here (struck through) for history; only
item 4 (1B) and item 5 (W&B) are still genuinely open.

1. [x] ~~Run the notebook top-to-bottom on GPU for **300M** with a tiny split (`SMOKE_TEST=True`).~~
       Done. bf16 autocast + `bitsandbytes.AdamW8bit` both worked with no fallback triggered
       (no `[opt] AdamW8bit unavailable` message printed) — confirmed on a Blackwell (sm_120) GPU.
2. [x] ~~Full **`TrainAPI.run`** multi-epoch loop end-to-end~~ (early-stop counter, best-WER
       ckpt via the `torch.save` fallback path — `save_pretrained` isn't defined on the raw
       fairseq2 module so it always takes the `except` branch, per-epoch `_validate` generate).
       All ran for real on `omniASR_LLM_300M`.
3. [x] ~~Confirm **manual-LoRA ckpt round-trips through Cell 16**~~ Done — `load_adapter` raised
       (as expected, no PEFT wrapper), fell to `torch.load(adapter.pt)` +
       `load_state_dict(strict=False)`, reload succeeded, tuned predict ran off the reloaded ckpt.
4. [ ] Repeat for **omniASR_LLM_1B** (Cell 17 second pass) — not cached on this box, needs a
       fresh multi-GB download; `MODEL_NAME` currently pinned to 300M, swap back when ready.
5. [ ] **W&B** logging — intentionally left `WANDB_MODE=disabled` (no account configured here);
       the per-epoch print statements already cover the loss/WER/CER visibility requirement.
6. [x] ~~Swap **Cell 8 dataset**~~ Done for the smoke-test path (1/1/1 real apc North-Levantine
       samples). The non-smoke (full-corpus) path in `load_splits` is written but not yet
       exercised — full training run over the real 517-row corpus is still open.

---

## ENVIRONMENT — critical, non-obvious

- **No GPU** on this box; torch on `/root/.../.venv` is CPU. `omnilingual-asr` runs on CPU fine.
- **Two filesystems**: overlay `/` is **5 GB (~330 MB free)** — too small for anything big.
  `/workspace` is a **RunPod net volume (365 TB)** BUT has a **per-user disk quota** — writing
  there eventually fails with `OSError [Errno 122] Disk quota exceeded` (that's why `peft` could
  not be installed and why log/ckpt writes to `/workspace` silently produced 0-byte files).
  → **Write logs/checkpoints to the overlay** (`/tmp/.../scratchpad` or the project dir), read
  model/venv from `/workspace`.
- The notebook targets `/workspace/asr_env` (`ROOT`) — this box IS effectively the RunPod box
  minus the GPU.

### The working CPU venv (already built) — use this to re-run anything
```
/workspace/venv_omni/bin/python        # CPU torch 2.8 + fairseq2n 0.6+cpu + omnilingual-asr + jiwer
```
**Install gotcha for a fresh venv:** default PyPI `fairseq2n==0.6` HARD-REJECTS CPU torch
(`_check_torch_version` demands a CUDA build). Install the CPU variant instead:
```
pip install torch==2.8.0 torchaudio==2.8.0 --index-url https://download.pytorch.org/whl/cpu
pip install omnilingual-asr torch==2.8.0+cpu torchaudio==2.8.0+cpu --extra-index-url https://download.pytorch.org/whl/cpu
pip install --force-reinstall --no-deps "fairseq2n==0.6+cpu" --extra-index-url https://fair.pkg.atmeta.com/fairseq2/whl/pt2.8.0/cpu
pip install jiwer
```
`peft` is NOT installed (quota) and NOT needed — OmniASR uses manual LoRA.

### Real Arabic data on disk
- `/workspace/asr/Palestinian-ASR/omnilingual_selected/apc_north_levantine_all_splits/` — HF
  `datasets` arrow, **North-Levantine (Palestinian) Arabic**, audio = **FLAC bytes**, text col
  `raw_text`. Decode with **`soundfile`** (libsndfile), NOT torchcodec/ffmpeg (ffmpeg libs missing).
- `other_arabic_dialects/` also present (13 GB total under `omnilingual_selected`).
- Prebuilt 4-sample test fixture: `/workspace/asr_env/g5_fixture/{audio.npz, meta.json}`.

---

## FILES (all in `/root/asr/Palestinian-ASR/`)

| file | what |
|---|---|
| `asr_model_agnostic_finetune.ipynb` | the harness — **fixed**. Cell 6 = OmniASRAdapter + `_LoRALinear` |
| `DISCOVERY.md` | real API findings + assumption verdicts |
| `SMOKE_RESULTS.md` | all evidence (plumbing + real gates + train path) + remaining GPU list |
| `HANDOFF.md` | this file |
| `run_gates_cpu.py` | G1–G5 on real 300M (CPU). `/workspace/venv_omni/bin/python run_gates_cpu.py` |
| `train_lora_cpu.py` | manual-LoRA attach→backward→save/reload on real 300M (CPU) |
| `smoke_cpu_plumbing.py` | harness control-flow test with a fake adapter (no model needed) |
| `mock_seq2seq_test.py` | Seq2SeqBatch wiring test with fairseq2 mocked (no model/GPU) |

Reproduce the real gates: `/workspace/venv_omni/bin/python run_gates_cpu.py`
(logs must go to overlay, e.g. `> /tmp/.../scratchpad/x.log`, not `/workspace`).

---

## QUICK "is it working?" answer
Load / preprocess / inference / loss / LoRA-train are **proven on real 300M weights (CPU)**.
NOT yet proven: GPU precision/optimizer plumbing (bf16 autocast, 8-bit Adam, grad-ckpt), the
assembled multi-epoch loop, and the 1B model. So: high confidence, not a 100% guarantee until a
GPU run of the full `TrainAPI.run` loop passes.
