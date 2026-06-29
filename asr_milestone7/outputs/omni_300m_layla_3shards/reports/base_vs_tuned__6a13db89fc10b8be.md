# Base vs tuned ASR comparison

- Model: `Omni ASR 300M`
- Family: `omni`
- Run: `omni_300m_layla_3shards`
- Base metrics: `/home/MohammadNabulsi/whisper/asr_milestone7/outputs/omni_300m_layla_3shards/metrics/base/omni__Omni_ASR_300M__omni_300m_layla_3shards__base__fa68c91d5cbfb857.json`
- Tuned metrics: `/home/MohammadNabulsi/whisper/asr_milestone7/outputs/omni_300m_layla_3shards/metrics/tuned/omni__Omni_ASR_300M__omni_300m_layla_3shards__tuned__11c92233f6e8a90e.json`

| metric | base | tuned | absolute improvement | relative improvement |
|---|---:|---:|---:|---:|
| wer | 0.013609 | 0.000000 | 0.013609 | 1.000000 |
| cer | 0.029327 | 0.000000 | 0.029327 | 1.000000 |
| normalized_wer | 0.013615 | 0.000000 | 0.013615 | 1.000000 |
| normalized_cer | 0.028382 | 0.000000 | 0.028382 | 1.000000 |
| loose_wer | 0.013615 | 0.000000 | 0.013615 | 1.000000 |
| loose_cer | 0.028382 | 0.000000 | 0.028382 | 1.000000 |

Positive improvement means the tuned error rate is lower than the base error rate.
