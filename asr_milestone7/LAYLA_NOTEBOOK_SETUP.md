# Layla Setup For `asr_milestone7`

The current notebook at [notebooks/train_asr_plug_play.ipynb](/home/MohammadNabulsi/whisper/asr_milestone7/notebooks/train_asr_plug_play.ipynb) is a smoke notebook.

Today it does all of these things on purpose:

- loads `configs/smoke_config.yaml`
- forces `smoke_mode=True`
- generates fake smoke data
- loops over all three architecture families

Because of that, changing only the YAML file is not enough for a real Layla run.

## What Is Ready

You can now generate milestone-7-compatible JSONL manifests from three Layla source batches with:

```bash
python3 asr_milestone7/scripts/make_layla_jsonl_manifests.py
```

By default this creates:

- `asr_milestone7/data/layla_manifests/train.jsonl`
- `asr_milestone7/data/layla_manifests/val.jsonl`
- `asr_milestone7/data/layla_manifests/test.jsonl`

using these three Layla batches:

- `normalized_output_appended.json` as `train`
- `normalized_layla_batch_130_appended_131.json` as `val`
- `normalized_pasted_132_133_appended.json` as `test`

There is also a matching config at [configs/omni_300m_layla_3shards.yaml](/home/MohammadNabulsi/whisper/asr_milestone7/configs/omni_300m_layla_3shards.yaml).

## Exact Model Name

Use this exact value, because the registry matches model names literally:

```yaml
model_name: Omni ASR 300M
```

`omni lingual 0.3B` is the right idea, but the code in [asr_pipeline/registry.py](/home/MohammadNabulsi/whisper/asr_milestone7/asr_pipeline/registry.py) currently recognizes `Omni ASR 300M`.

## Notebook Changes Needed

If you want to reuse the existing notebook, update these parts:

1. Change the config path from `configs/smoke_config.yaml` to `configs/omni_300m_layla_3shards.yaml`.
2. Remove the override that forces `smoke_mode=True`.
3. Remove the fake data generation cell.
4. Replace the `smoke_representative_models` list with only:

```python
["Omni ASR 300M"]
```

## Current Blocker

Even after the dataset/config changes, a true Omni run still will not finish yet.

The non-smoke Omni code paths in these files are still placeholders:

- [asr_pipeline/adapters/omni.py](/home/MohammadNabulsi/whisper/asr_milestone7/asr_pipeline/adapters/omni.py)
- [asr_pipeline/train.py](/home/MohammadNabulsi/whisper/asr_milestone7/asr_pipeline/train.py)

That means:

- `smoke_mode: true` will only test plumbing with deterministic fake predictions/training behavior.
- `smoke_mode: false` will stop at `NotImplementedError` for real model loading, prediction, and training.

## Bottom Line

The directory can now be configured with Layla train/val/test manifests and the correct Omni 300M model name, but real Omni fine-tuning still needs adapter implementation before the notebook can run end-to-end on the real data.
