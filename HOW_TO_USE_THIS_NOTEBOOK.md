# How to use `asr_lora_finetune_unified.ipynb`

One notebook that LoRA-fine-tunes any of four Arabic ASR model families on any dataset in
this repo's Arrow format. You do not edit training logic to switch models or data — you
only ever touch **Cell 2**.

## TL;DR

1. Pick a kernel that matches the model you want (table below).
2. Open Cell 2, set `MODEL_NAME` and `DATASET_DIR`.
3. Run all cells top to bottom.

That's it for the common case. Everything below explains what those two knobs actually
control and what else you can change if the defaults don't fit.

## 1. Kernel setup — do this first

Each model family needs its own Python environment because their dependencies genuinely
conflict (OmniASR's `fairseq2n` pins `transformers==4.57.6`/`numpy<2`; Qwen and Cohere need
`transformers>=5.13`/`numpy2`). The notebook itself doesn't change — only which kernel you
launch it with.

| `MODEL_NAME` | kernel | venv |
|---|---|---|
| `omnilingual-asr/omniASR_LLM_300M`, `omnilingual-asr/omniASR_LLM_1B` | `venv_omni_gpu` | `/workspace/venv_omni_gpu` |
| `nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0`, `..._pc_v1.0` | `venv_nemo_gpu` | `/workspace/venv_nemo_gpu` |
| `Qwen/Qwen3-ASR-0.6B-hf` | `venv_qwen_gpu` | `/workspace/venv_qwen_gpu` |
| `CohereLabs/cohere-transcribe-arabic-07-2026` (gated) | `venv_qwen_gpu` | `/workspace/venv_qwen_gpu` |

Currently only `venv_omni_gpu` is registered as a Jupyter kernel
(`jupyter kernelspec list`). If you pick a Qwen/Cohere or NeMo model and the kernel isn't
in the switcher yet, register it once from that venv:

```bash
/workspace/venv_nemo_gpu/bin/python -m ipykernel install --user --name=venv_nemo_gpu --display-name "venv_nemo_gpu"
/workspace/venv_qwen_gpu/bin/python -m ipykernel install --user --name=venv_qwen_gpu --display-name "venv_qwen_gpu"
```

then reload the kernel picker in your notebook UI.

Cell 2 prints which kernel it *should* be running under (`KERNEL_BY_MODEL`) — if you picked
the wrong one, imports will fail loudly inside `load_base()` rather than silently doing the
wrong thing.

## 2. Cell 2 — the two knobs

```python
MODEL_NAME  = "Qwen/Qwen3-ASR-0.6B-hf"
DATASET_DIR = "/workspace/asr/Palestinian-ASR/omnilingual_selected/apc_north_levantine_all_splits"
```

### Changing the model

Set `MODEL_NAME` to one of the six registry keys in the table above (they live in
`KERNEL_BY_MODEL`/`REGISTRY` in Cells 2 and 7). Restart the kernel to match, then re-run from
Cell 1. Nothing else in the notebook needs editing — LoRA target modules, batch size, audio
length caps, etc. are all looked up per-model from `ConfigAPI` (Cell 4).

The Cohere model is a **gated** HF repo. If you haven't accepted its license with the account
whose token is active (`hf auth login` or `HF_TOKEN` env var), Cell 2 prints a warning and the
download will 401 later in Cell 11.

### Changing the dataset

Set `DATASET_DIR` to any directory loadable with `datasets.load_from_disk(...)`. Cell 8
(`load_splits`) expects:

- a flat `Dataset` (not a `DatasetDict`) — the notebook does its own train/val/test split
- an `audio` column whose entries are `{"bytes": <FLAC bytes>}` (decoded manually via
  `soundfile`, not `datasets`' built-in `Audio(decode=True)` — see the comment in Cell 8 for
  why: this repo's `datasets` version otherwise pulls in `torchcodec`, which risks breaking
  the pinned CUDA torch build)
- a `text` column (or `raw_text`, which gets renamed to `text` automatically)
- a `duration` column in seconds (used to filter for the smoke-test band and to compute
  dataset hours in the final summary)

`apc_north_levantine_all_splits` (the current default) and
`other_arabic_dialects` under `omnilingual_selected/` are both in this shape already — point
`DATASET_DIR` at any sibling directory built the same way to swap corpora.

If your data doesn't have a `duration` column or uses a different audio encoding, you'll need
to adapt `load_splits`/`_materialize_audio` in Cell 8 — that's the one place dataset shape is
assumed.

## 3. `SMOKE_TEST` — also in Cell 2

```python
SMOKE_TEST = True   # tiny subsets + 2 epochs
```

- `True`: 1 train / 1 val / 1 test example, capped at the model's `max_audio_seconds`, 2
  epochs. This is for verifying the whole pipeline (load → LoRA → train → eval → save) runs
  end-to-end on your kernel/data combo before committing GPU time to a real run. Takes minutes.
- `False`: real 80/10/10 split of the whole dataset, shuffled with `SEED=42`, trained for up
  to `TrainConfigSpec.num_epochs` (50 by default) with early stopping on validation WER
  (`patience=3`, or 4 under smoke — see Cell 4/13).

Always do one `SMOKE_TEST=True` pass on a new model+data combo before flipping it to `False`.

## 4. Changing hyperparameters

Per-model LoRA and training defaults live in `ConfigAPI` in Cell 4 (`_LORA` and `_TRAIN`
dicts, keyed by `MODEL_NAME`). To change something for a real run — batch size, learning
rate, LoRA rank, epochs, early-stopping patience — edit the relevant `LoRAConfigSpec`/
`TrainConfigSpec` entry for your model in Cell 4, or just reassign fields after Cell 13:

```python
train_spec.learning_rate = 5e-5
train_spec.num_epochs = 20
lora_spec.r = 16
```

do this **before** Cell 13's `adapter.apply_lora(lora_spec)` call for LoRA changes (rank/
alpha/target_modules only take effect at that call), or before Cell 15 (`TrainAPI.run`) for
training-loop changes.

## 5. Running it — cell by cell

| Cell | What it does |
|---|---|
| 1 | Imports. No pip install here on purpose — your kernel's venv already has the right stack. |
| 2 | **The two knobs.** Also sets up MLflow tracking (`sqlite:///.../mlflow.db`) and paths under `ASR_ENV_ROOT` (default `/workspace/asr_env`). |
| 3 | Arabic text normalization + WER/CER via `jiwer`. |
| 4 | `ConfigAPI` — per-model LoRA/training hyperparameter defaults. |
| 5 | `ModelAdapter` ABC — the abstraction every model family implements (`load_base`, `preprocess`, `collate`, `generate`, `apply_lora`, checkpoint save/load). |
| 6–9 | The four concrete adapters: OmniASR, FastConformer-CTC, Qwen3-ASR, CohereAsr. You don't need to touch these unless adding a new model. |
| 10 | `REGISTRY` mapping `MODEL_NAME -> adapter class`. |
| 11 | Loads and materializes `DATASET_DIR` into train/val/test splits. |
| 12 | `PredictAPI` — cached inference (keyed on model + split + dataset fingerprint + stage). |
| 13 | `EvaluateAPI` — cached WER/CER (keyed additionally on a hash of the actual predictions, so a metric never gets silently re-served after the data changes). |
| 14 | Builds the adapter and loads the base (pre-LoRA) model — this is the slow "download/load weights" step. |
| 15 | Baseline predict + eval on `test` before any fine-tuning (`stage="base"`). |
| 16 | Applies LoRA to the loaded model. |
| 17 | `TrainAPI` — the custom train loop (not `Seq2SeqTrainer`, since the fairseq2 OmniASR module isn't a `PreTrainedModel`). Per-epoch train/val loss + val WER/CER, early stopping, best-WER checkpoint, periodic numbered checkpoints for resume. |
| 18 | **Runs training.** This is the long-running cell. |
| 19 | Reloads the best checkpoint, re-predicts + re-evaluates on `test` (`stage="tuned"`), writes a JSON summary (base vs. tuned WER/CER delta) and logs it to MLflow. |
| 20 | A one-shot `smoke(...)` wrapper that redoes load→LoRA→train→eval for whichever `MODEL_NAME` is currently set — useful for a quick end-to-end re-check without manually re-running 11–19. |

For a first run on a new model/kernel, just run all cells top to bottom with
`SMOKE_TEST=True`. For a real training run, set `SMOKE_TEST=False`, adjust hyperparameters
if needed (§4), and run all cells again.

## 6. Where outputs go

All under `ASR_ENV_ROOT` (default `/workspace/asr_env`, overridable via that env var):

- `preds/` — cached prediction JSONs (`<model>__<split>__<fingerprint>__<stage>.json`)
- `metrics/` — cached WER/CER JSONs, plus `<model>__SUMMARY.json` (base vs. tuned, dataset
  stats, the exact LoRA/train config used)
- `checkpoints/<model>/best/` — best-validation-WER LoRA adapter
- `checkpoints/<model>/ckpt_step*/` — periodic checkpoints (optimizer/scheduler/RNG state
  included) for resuming an interrupted run; `TrainConfigSpec.save_total_limit` controls how
  many are kept
- `mlflow.db` + `mlruns/` — full metric history, browsable with
  `mlflow ui --backend-store-uri sqlite:////workspace/asr_env/mlflow.db`

## 7. Resuming an interrupted training run

`TrainAPI.run(..., resume=True)` (the default, Cell 18) automatically picks up the latest
`ckpt_step*` under `checkpoints/<model>/` if one exists — optimizer, scheduler, RNG state,
epoch/step counters, and MLflow run id are all restored. Just re-run the notebook the same
way; there's no separate resume flag to set.

## 8. Adding a fifth model family

Subclass `ModelAdapter` (pattern in Cells 6–9), implement `load_base`, `preprocess`,
`collate`, `generate` at minimum (override `apply_lora`/`save_checkpoint`/`load_checkpoint`
only if the model's LoRA-able layers aren't plain `torch.nn.Linear`, like OmniASR's fairseq2
projections), add its hyperparameters to `ConfigAPI._LORA`/`_TRAIN`, its kernel to
`KERNEL_BY_MODEL`, its language code to `LANG_BY_MODEL`, and register it in `REGISTRY`.

## 9. Known gotchas

- **OmniASR audio length ceiling**: `max_audio_seconds=40.0` for both OmniASR sizes is a
  verified hard inference limit, not a tunable default — clips longer than that will train
  fine but crash at `generate()`/transcribe time.
- **CTC vs. seq2seq loss**: `ConformerCTCAdapter.loss_type = "ctc"`; the other three are
  `"seq2seq"`. This only matters if you're reading `adapter.loss_type` elsewhere — the train
  loop itself is loss-type-agnostic (`train_step` returns a scalar either way).
- **CohereAsr chunking**: if a training clip is long enough that the processor internally
  chunks it into multiple feature rows, `collate()` raises rather than silently corrupting
  batch alignment — keep training clips under Cohere's `max_audio_clip_s`.
- **Cache invalidation**: `PredictAPI`/`EvaluateAPI` caches are keyed on dataset fingerprint
  and (for metrics) a hash of the actual predictions — safe to re-run cells repeatedly without
  wasting compute, and safe against stale metrics after swapping `DATASET_DIR`.
