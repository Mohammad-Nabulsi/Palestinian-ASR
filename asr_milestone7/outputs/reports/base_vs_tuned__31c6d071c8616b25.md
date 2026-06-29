# Base vs tuned ASR comparison

- Model: `Qwen/Qwen3-ASR-0.6B`
- Family: `qwen`
- Run: `milestone7_smoke_qwen`
- Base metrics: `/home/MohammadNabulsi/whisper/asr_milestone7/outputs/metrics/base/qwen__Qwen__Qwen3-ASR-0.6B__milestone7_smoke_qwen__base__0926ec75e87c6dfe.json`
- Tuned metrics: `/home/MohammadNabulsi/whisper/asr_milestone7/outputs/metrics/tuned/qwen__Qwen__Qwen3-ASR-0.6B__milestone7_smoke_qwen__tuned__d2f733b326412d24.json`

| metric | base | tuned | absolute improvement | relative improvement |
|---|---:|---:|---:|---:|
| wer | 0.600000 | 0.000000 | 0.600000 | 1.000000 |
| cer | 0.954545 | 0.000000 | 0.954545 | 1.000000 |
| normalized_wer | 0.600000 | 0.000000 | 0.600000 | 1.000000 |
| normalized_cer | 0.863636 | 0.000000 | 0.863636 | 1.000000 |
| loose_wer | 0.600000 | 0.000000 | 0.600000 | 1.000000 |
| loose_cer | 0.863636 | 0.000000 | 0.863636 | 1.000000 |

Positive improvement means the tuned error rate is lower than the base error rate.
