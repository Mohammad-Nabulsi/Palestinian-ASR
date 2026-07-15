# OmniASR Adapter — Discovery & Repair Notes

**Date:** 2026-07-15
**Scope:** Validate/repair the `OmniASRAdapter` (Cell 6) in `asr_model_agnostic_finetune.ipynb`
against the real `omnilingual-asr` API.

## ⚠️ Environment constraint (why gates G1/G2/G5 were not executed here)

This machine **cannot run the real model**:
- **No GPU** — no `/dev/nvidia*`, nothing in `lspci`, `torch.cuda.is_available() == False`
  (the installed torch in `.venv` is `2.11.0+cpu`).
- **No disk headroom** — root overlay is 5.0 GB, ~330 MB free. Installing `omnilingual-asr`
  pulls `fairseq2==0.6` + `fairseq2n` + `torch 2.8.0` (888 MB wheel) + ~5 GB of CUDA wheels.
  It physically does not fit, and I did not delete the user's venvs to make room.

Therefore Phase 0 discovery was done by **reading the published source** at
`github.com/facebookresearch/omnilingual-asr@main` (authoritative, not guessed), and the
harness *plumbing* was proven with a **CPU fake-adapter smoke run** (see
`smoke_cpu_plumbing.py` / `SMOKE_RESULTS.md`). The three real-model gates (G1 LOAD,
G2 CACHE, G5 BASE PRED) **must still be run on the RunPod GPU box** — they are the only
gates that require the actual weights. The notebook targets `/workspace/asr_env` (a RunPod
network volume), which is where the real run is intended to happen anyway.

---

## Source files read

| file | what it told us |
|---|---|
| `src/omnilingual_asr/models/inference/pipeline.py` | `ASRInferencePipeline`: attrs, tokenizer, `transcribe()` signature, batch construction |
| `src/omnilingual_asr/models/wav2vec2_llama/model.py` | `Wav2Vec2LlamaModel.forward` — **computes the loss internally**; input syntax |
| `workflows/recipes/wav2vec2/asr/criterion.py` | training criterion — how loss is obtained from the model |
| `src/omnilingual_asr/models/wav2vec2_llama/factory.py` | fairseq2 attention/FFN module classes → real LoRA target names |
| `src/omnilingual_asr/models/wav2vec2_llama/lang_ids.py` | `supported_langs` — confirms `arb_Arab` |
| `src/omnilingual_asr/models/wav2vec2_llama/hub.py` | model hub accessor (fairseq2 asset card) |

---

## Assumption ledger (the 6 unverified guesses + the blocker)

| # | Assumption in notebook | Verdict | Reality |
|---|---|---|---|
| 1 | `pipeline.model` / `._model` holds the nn.Module | ✅ **correct** | `pipeline.model` is the `Wav2Vec2LlamaModel` (set in `__init__` via `load_model(model_card)`). `._model` does not exist; `getattr(p,"model",None) or p._model` works but the fallback is dead code. |
| 2 | `pipeline.tokenizer` exists | ✅ **correct** | `self.tokenizer = load_tokenizer(model_card)`. Also exposes `token_encoder`, `token_decoder`. |
| 3 | `model(**batch)` accepts `labels`, returns `.loss` | ❌ **wrong API, but loss IS supervised** | See Phase 1 below. Real call is `loss = model(seq2seq_batch)`; there is **no** `input_values/attention_mask/labels` kwarg interface. |
| 4 | LoRA targets `q_proj/k_proj/v_proj/o_proj/gate_proj/up_proj/down_proj` + PEFT/Unsloth wrapping | ❌ **wrong names AND wrong mechanism** | Correct names: `q_proj,k_proj,v_proj,output_proj,gate_proj,inner_proj` (228 layers). **The real-model run then revealed these are `fairseq2.nn.projection.Linear` (MRO `Linear→Projection→Module`), NOT a `torch.nn.Linear` subclass** — so `isinstance(mod, nn.Linear)` is False, the old derivation silently found 0 and fell back, and **PEFT `get_peft_model`/Unsloth cannot wrap them** (they dispatch LoRA on `torch.nn.Linear`). The notebook's PEFT `apply_lora` would fail on OmniASR. **Fix: manual LoRA injection (`_LoRALinear`) into any module with a 2-D `.weight`; detection is duck-typed, not isinstance.** |
| 5 | `FAIRSEQ2_CACHE_DIR` controls checkpoint cache | ⚠️ **unverified / likely insufficient** | Checkpoints resolve through fairseq2's asset store (`load_model`/`load_tokenizer`) and are fetched via `huggingface_hub` (`hf-xet` is a dep). fairseq2's asset cache defaults to `~/.cache/fairseq2/assets`; HF blobs land under `HF_HOME`/`HF_HUB_CACHE`. **On the GPU box, after the first load run `find ~ -name '*300M*' -o -path '*fairseq2*'` to locate the real path**, then set that env var. Do not rely on `FAIRSEQ2_CACHE_DIR` alone until confirmed. |
| 6 | `tokenizer.create_encoder(lang=...)` encodes text | ❌ **wrong** | Encoder takes **no** `lang`: `tokenizer.create_encoder()` → callable `enc(text) -> LongTensor`. Decode: `tokenizer.create_decoder(skip_special_tokens=True)`. Language is **not** a tokenizer arg — it is passed to `transcribe(..., lang=[...])` and/or lives in `batch.example["lang"]`; the model inserts the lang token into the decoder syntax itself. |

---

## Phase 1 — the blocker (assumption #3), resolved

**Branch taken: YES — `forward` computes a supervised loss. Do NOT hand-roll cross-entropy.**

Evidence from `model.py::Wav2Vec2LlamaModel.forward(batch, return_logits=False, return_decoder_inputs=False)`:

```
# builds syntax:  target audio [<special> lang] <bos> target text <eos>
inputs   = self.create_default_syntax(batch, device)      # BOS/EOS + lang handled here
embedded = self.embed_inputs_training(inputs, dtype)      # (training branch)
dec_out  = self.llama_decoder(decoder_inputs, layout)
logits   = self.final_proj(dec_out)
targets, targets_layout = batch.as_target_input()
loss = self.compute_loss(logits, ..., targets, ...,
                         pad_idx=self.target_vocab_info.pad_idx,
                         eos_idx=self.target_vocab_info.eos_idx, ...)
return loss                                               # default return is the loss Tensor
```

And `criterion.py` confirms the intended training call:

```
if isinstance(self._model.base_module, Wav2Vec2LlamaModel):
    return self._model.module(batch)          # -> CTC/LM loss Tensor
```

Consequences / label conventions (all handled **inside** the model — we must NOT do them ourselves):
- **BOS/EOS**: `create_default_syntax` prepends `<bos>` and appends `<eos>` (`add_eos`);
  `bos_idx/eos_idx` come from `tokenizer.vocab_info`.
- **Lang token**: injected by the model when `lang_embeddings_p > 0` / for LID models, read from
  `batch.example["lang"]`. So we must put `example={"lang": [LANG]*N}` on the batch.
- **Padding / loss mask**: `compute_loss` uses `pad_idx` and an internal `loss_mask`
  (`loss=True` only on the text + eos spans). We must **not** pass `-100`; padding is `pad_idx`.
- **Targets** are plain token ids from `tokenizer.create_encoder()(text)` — no manual shifting.

### What the training input must be — a fairseq2 `Seq2SeqBatch`

```python
from fairseq2.datasets.batch import Seq2SeqBatch
batch = Seq2SeqBatch(
    source_seqs      = wav,            # (N, T_audio) float, in model dtype/device
    source_seq_lens  = audio_lens,     # true lengths (N,)
    target_seqs      = tok_ids,        # (N, L) int64, padded with tokenizer pad_idx
    target_seq_lens  = label_lens,     # true label lengths (N,)
    example          = {"lang": [LANG]*N},
)
loss = model(batch)                    # scalar Tensor
```

`source_seqs`/`target_seqs` padding follows the pipeline's own `Collater` (audio pad 0,
text pad = `tokenizer.vocab_info.pad_idx`). The `-100` collate in the old notebook is a
transformers convention and is wrong here.

---

## Inference (`generate`) — mostly correct already

`ASRInferencePipeline.transcribe(inp, *, lang=None, batch_size=2) -> List[str]` accepts:
- `List[str|Path]` audio file paths, **or**
- `List[dict]` pre-decoded `{"waveform": ndarray/tensor, "sample_rate": int}` (no temp files needed),
- `lang`: `List[str|None]` same length as `inp`.

The notebook's temp-`.wav` approach works, but passing dicts is cleaner and avoids disk I/O.
`transcribe` is `@torch.inference_mode()` and resamples to 16 kHz + normalizes internally, so
generate does not need to pre-resample. Constraint: `MAX_ALLOWED_AUDIO_SEC = 40`.

Note: the base `omniASR_LLM_300M/1B` are standard LLM-ASR (call `.transcribe()`); only
`omniASR_LLM_7B_ZS` is zero-shot and needs `.transcribe_with_context()`.

---

## Fixes applied to the notebook

1. **Cell 4 (ConfigAPI)** — LoRA `target_modules` corrected to fairseq2 names
   (`q_proj,k_proj,v_proj,output_proj,gate_proj,inner_proj`).
2. **Cell 6 (OmniASRAdapter)**:
   - `_encode` → `tokenizer.create_encoder()` (no `lang`); decode via `create_decoder(skip_special_tokens=True)`.
   - `collate` → pads labels with the tokenizer `pad_idx` (not `-100`) and keeps true lengths.
   - `train_step` → overridden to build a `Seq2SeqBatch` and call `loss = model(batch)`.
   - `generate` → uses pre-decoded dict inputs to `transcribe`.
   - `_derive_target_modules()` helper to read real Linear leaf names from `named_modules()`.
   - device-agnostic (`DEVICE`, not hard-coded `"cuda"`).
3. **Whole notebook** — `DEVICE = "cuda" if torch.cuda.is_available() else "cpu"`; every
   `.to("cuda")` in PredictAPI / TrainAPI replaced with `.to(DEVICE)`; `ROOT` overridable via
   `ASR_ENV_ROOT` env var so it runs off-RunPod.

## Still TODO on the GPU box (cannot be done here)

- **G1 LOAD** — load 300M, print param count / dtype / device.
- **G2 CACHE** — locate the real cache path (see #5), time cold vs warm load.
- **G3 round-trip** — confirm `decode(encode(text))` recovers normalized Arabic on the real tokenizer.
- **G5 BASE PRED** — confirm non-empty Arabic-script hypotheses (empty/latin ⇒ wrong lang token).
- Re-derive `target_modules` from the real `named_modules()` and confirm PEFT wraps > 0 params.
- Confirm `Seq2SeqBatch` training `loss` is finite and decreases on the tiny set.
