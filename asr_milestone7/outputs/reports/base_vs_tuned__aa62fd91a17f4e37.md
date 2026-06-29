# Base vs tuned ASR comparison

- Model: `openai/whisper-medium`
- Family: `whisper`
- Run: `milestone7_smoke_whisper`
- Base metrics: `/home/MohammadNabulsi/whisper/asr_milestone7/outputs/metrics/base/whisper__openai__whisper-medium__milestone7_smoke_whisper__base__e733443202e3e3d5.json`
- Tuned metrics: `/home/MohammadNabulsi/whisper/asr_milestone7/outputs/metrics/tuned/whisper__openai__whisper-medium__milestone7_smoke_whisper__tuned__8f4a94631ef16563.json`

| metric | base | tuned | absolute improvement | relative improvement |
|---|---:|---:|---:|---:|
| wer | 0.600000 | 0.000000 | 0.600000 | 1.000000 |
| cer | 1.090909 | 0.000000 | 1.090909 | 1.000000 |
| normalized_wer | 0.600000 | 0.000000 | 0.600000 | 1.000000 |
| normalized_cer | 1.000000 | 0.000000 | 1.000000 | 1.000000 |
| loose_wer | 0.600000 | 0.000000 | 0.600000 | 1.000000 |
| loose_cer | 1.000000 | 0.000000 | 1.000000 | 1.000000 |

Positive improvement means the tuned error rate is lower than the base error rate.
