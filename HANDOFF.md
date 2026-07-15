# HANDOFF — OmniASR LoRA fine-tuning notebook

_Last updated: 2026-07-15. Read this first next session._

Target: validate/repair `asr_model_agnostic_finetune.ipynb` (model-agnostic ASR LoRA harness)
for `omnilingual-asr/omniASR_LLM_300M` → then `_1B`, language `arb_Arab`.

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

## STATUS: what REMAINS (needs a real GPU) ⏳

None of these can run on this box (no GPU). Priority order for next GPU session:

1. [ ] Run the notebook top-to-bottom on GPU for **300M** with a tiny split (`SMOKE_TEST=True`).
       Watch: bf16 autocast path, `bitsandbytes.AdamW8bit`, gradient checkpointing.
2. [ ] Full **`TrainAPI.run`** multi-epoch loop end-to-end (early-stop, best-WER ckpt via the
       `save_pretrained`→`torch.save` fallback, per-epoch `_validate` beam-search generate).
       Individual pieces are verified; the assembled loop on real weights is not.
3. [ ] Confirm **manual-LoRA ckpt round-trips through Cell 16** (`load_adapter` fails → `torch.load`
       `adapter.pt` → `load_state_dict(strict=False)`; verified in isolation, not via Cell 16).
4. [ ] Repeat for **omniASR_LLM_1B** (Cell 17 second pass).
5. [ ] **W&B** logging (was stubbed off here).
6. [ ] Swap **Cell 8 dataset**: currently Common Voice placeholder. Real apc/QASR data is on
       `/workspace/asr/Palestinian-ASR/omnilingual_selected/` (see below).

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
