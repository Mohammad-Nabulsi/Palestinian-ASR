# Base vs tuned ASR comparison

- Model: `Omni ASR 300M`
- Family: `omni`
- Run: `milestone7_smoke_omni`
- Base metrics: `/home/MohammadNabulsi/whisper/asr_milestone7/outputs/metrics/base/omni__Omni_ASR_300M__milestone7_smoke_omni__base__eb380d86e9877afc.json`
- Tuned metrics: `/home/MohammadNabulsi/whisper/asr_milestone7/outputs/metrics/tuned/omni__Omni_ASR_300M__milestone7_smoke_omni__tuned__0ba601a017c6fce9.json`

| metric | base | tuned | absolute improvement | relative improvement |
|---|---:|---:|---:|---:|
| wer | 1.000000 | 0.000000 | 1.000000 | 1.000000 |
| cer | 1.000000 | 0.000000 | 1.000000 | 1.000000 |
| normalized_wer | 1.000000 | 0.000000 | 1.000000 | 1.000000 |
| normalized_cer | 1.000000 | 0.000000 | 1.000000 | 1.000000 |
| loose_wer | 1.000000 | 0.000000 | 1.000000 | 1.000000 |
| loose_cer | 1.000000 | 0.000000 | 1.000000 | 1.000000 |

Positive improvement means the tuned error rate is lower than the base error rate.
