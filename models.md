# Models

This project stores all downloaded model checkpoints locally under:

```text
models/
```

Download logs are written to:

```text
.logs/
```

---

## Downloaded Models

The following models are downloaded using:

```bash
python download_models.py
```

| Model | Local Directory | Status |
|--------|-----------------|--------|
| Whisper Medium | `models/whisper_medium/` | ✅ Downloaded |
| Whisper Large-v3 | `models/whisper_large_v3/` | ✅ Downloaded |
| OmniASR-LLM-300M | `models/omni_asr_llm_300m/` | ✅ Downloaded |
| OmniASR-LLM-1B | `models/omni_asr_llm_1b/` | ✅ Downloaded |
| Qwen3-ASR-0.6B | `models/qwen3_asr_0_6b/` | ✅ Downloaded |
| Qwen3-ASR-1.7B | `models/qwen3_asr_1_7b/` | ✅ Downloaded |

To download all models:

```bash
python download_models.py
```

To download a single model:

```bash
python download_models.py whisper_medium
```

Logs for each model are stored separately, for example:

```text
.logs/whisper_medium.log
.logs/qwen3_asr_1_7b.log
```

---

## NVIDIA Conformer-CTC Large Arabic

The NVIDIA Arabic Conformer model was intentionally **not downloaded**.

An initial attempt to download it through Hugging Face failed because the repository does not exist there (HTTP 404). The model is distributed through NVIDIA's ecosystem rather than as a standard Hugging Face repository.

Since this project currently focuses on models that can be trained using the Hugging Face ecosystem, support for the NVIDIA Conformer model has been deferred to a future stage of the project.

When support is added, a dedicated download and training pipeline will be implemented separately.

---

## Directory Structure

```text
models/
├── whisper_medium/
├── whisper_large_v3/
├── omni_asr_llm_300m/
├── omni_asr_llm_1b/
├── qwen3_asr_0_6b/
└── qwen3_asr_1_7b/

.logs/
├── whisper_medium.log
├── whisper_large_v3.log
├── omni_asr_llm_300m.log
├── omni_asr_llm_1b.log
├── qwen3_asr_0_6b.log
└── qwen3_asr_1_7b.log
```