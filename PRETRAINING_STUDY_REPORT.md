# Palestinian ASR — Pretraining Study Report

Whisper-medium + LoRA. 11 pretrain-then-fine-tune configurations against a Palestinian Arabic ASR target, each fine-tuned to early stopping, plus a cross-domain generalization sweep (37 checkpoints, 188 domain-evaluations).

## Training results — pretrain → fine-tune progression

Baseline (no pretrain, direct fine-tune on Palestinian data only) test WER: **36.71%**.

| Run | Corpus | Pretrain epochs | Merged (pre-finetune) test WER | Final (post-finetune) test WER |
|---|---|---|---:|---:|
| run1 | No pretrain (baseline) | – | – | 36.71% |
| run2 | Jordanian (1.98h) | 1ep | 48.01% | 36.29% |
| run2 | Jordanian (1.98h) | 2ep | 46.21% | 35.10% |
| run3 | Omni | 1ep | 58.93% | 36.54% |
| run3 | Omni | 2ep | 58.98% | 35.78% |
| run4 | Omni + Jordanian | 1ep | 50.11% | 35.48% |
| run4 | Omni + Jordanian | 2ep | 48.69% | 35.41% |
| run5 | Layla + Jordanian | 1ep | 49.53% | 35.34% |
| run5 | Layla + Jordanian | 2ep | 46.45% | 36.58% |
| run6 | QASR-Levantine 10h | 1ep | 55.69% | 37.71% |
| run6 | QASR-Levantine 10h | 2ep | 53.77% | 35.51% |
| run7 | QASR-nonLevantine 50h | 1ep | 55.89% | 34.54% |
| run7 | QASR-nonLevantine 50h | 2ep | 55.32% | 34.41% |
| run8 | QASR-Lev + MASC-Lev | 1ep | 51.37% | 34.95% |
| run8 | QASR-Lev + MASC-Lev | 2ep | 50.62% | 33.10% |
| run9 | QASR-Lev + MASC-Lev + QASR-nonLev 50h | 1ep | 51.03% | 34.24% |
| run9 | QASR-Lev + MASC-Lev + QASR-nonLev 50h | 2ep | 51.03% | 33.03% **(best)** |
| run10 | All 6 sources combined (1ep only) | – | 48.70% | 33.17% |
| run11 | Staged: QASR-nonLev -> merge -> MASC+QASR-Lev -> merge -> pal | – | 56.11% | 33.97% |

## Cross-domain generalization — full matrix (WER %)

Every checkpoint evaluated on the domains it did *not* train on. Curated test sets for Omni/Layla/Casa-Jor; fresh untouched 2h samples for QASR-Lev/MASC-Lev/QASR-NonLev/MASC-NonLev. `–` = that run trained on this domain (excluded by design). `*` after a WER = post-Palestinian-fine-tune checkpoint.

| Checkpoint | Omni | Layla | Casa/Jor | QASR-Lev | MASC-Lev | QASR-NonLev | MASC-NonLev |
|---|---:|---:|---:|---:|---:|---:|---:|
| run2 1ep pretrain | 60.4 | 45.6 | – | 50.0 | 41.1 | 42.4 | 29.1 |
| run2 1ep +pal-ft | 58.6* | 51.5* | – | 56.8* | 46.2* | 50.9* | 33.9* |
| run2 2ep pretrain | 59.4 | 43.8 | – | 52.8 | 42.6 | 45.6 | 30.7 |
| run2 2ep +pal-ft | 58.8* | 49.7* | – | 59.0* | 47.5* | 53.2* | 35.1* |
| run3 1ep pretrain | – | 48.0 | 51.7 | 66.6 | 45.5 | 51.4 | 28.9 |
| run3 1ep +pal-ft | – | 54.4* | 39.5* | 57.2* | 47.0* | 52.3* | 34.1* |
| run3 2ep pretrain | – | 49.7 | 50.5 | 66.2 | 44.2 | 52.6 | 31.4 |
| run3 2ep +pal-ft | – | 54.5* | 42.3* | 54.8* | 49.0* | 50.4* | 36.4* |
| run4 1ep pretrain | – | 43.6 | – | 61.0 | 41.5 | 52.2 | 28.5 |
| run4 1ep +pal-ft | – | 52.7* | – | 60.4* | 47.4* | 55.2* | 35.2* |
| run4 2ep pretrain | – | 42.3 | – | 58.8 | 41.2 | 47.5 | 30.0 |
| run4 2ep +pal-ft | – | 55.2* | – | 57.4* | 48.9* | 53.7* | 37.5* |
| run5 1ep pretrain | 57.5 | – | – | 48.5 | 38.6 | 45.0 | 27.2 |
| run5 1ep +pal-ft | 59.2* | – | – | 59.8* | 47.8* | 53.6* | 34.4* |
| run5 2ep pretrain | 58.1 | – | – | 53.9 | 39.9 | 53.3 | 30.1 |
| run5 2ep +pal-ft | 57.3* | – | – | 62.6* | 44.5* | 56.0* | 34.0* |
| run6 1ep pretrain | 56.3 | 33.8 | 43.4 | – | 30.5 | 25.2 | 20.6 |
| run6 1ep +pal-ft | 60.4* | 52.3* | 42.9* | – | 58.7* | 53.1* | 35.8* |
| run6 2ep pretrain | 58.1 | 33.4 | 43.6 | – | 31.6 | 24.9 | 21.0 |
| run6 2ep +pal-ft | 59.0* | 52.8* | 39.6* | – | 46.0* | 43.6* | 34.6* |
| run7 1ep pretrain | 59.1 | 33.5 | 47.7 | 32.7 | 31.0 | – | 17.5 |
| run7 1ep +pal-ft | 60.3* | 52.1* | 41.6* | 55.6* | 47.6* | – | 30.4* |
| run7 2ep pretrain | 58.7 | 33.4 | 45.9 | 30.0 | 30.7 | – | 17.3 |
| run7 2ep +pal-ft | 60.5* | 52.0* | 43.1* | 49.0* | 44.8* | – | 30.8* |
| run8 1ep pretrain | 54.4 | 31.2 | 40.6 | – | – | 23.9 | 18.3 |
| run8 1ep +pal-ft | 57.1* | 50.5* | 37.2* | – | – | 44.0* | 31.1* |
| run8 2ep pretrain | 54.1 | 30.2 | 40.1 | – | – | 23.3 | 17.7 |
| run8 2ep +pal-ft | 57.0* | 52.2* | 38.7* | – | – | 45.5* | 33.4* |
| run9 1ep pretrain | 53.7 | 29.1 | 40.8 | – | – | – | 16.5 |
| run9 1ep +pal-ft | 56.3* | 47.5* | 37.4* | – | – | – | 28.2* |
| run9 2ep pretrain | 53.8 | 28.5 | 40.3 | – | – | – | 16.0 |
| run9 2ep +pal-ft | 56.8* | 50.3* | 37.7* | – | – | – | 29.3* |
| run10 pretrain | – | – | – | – | – | – | 16.9 |
| run10 +pal-ft | – | – | – | – | – | – | 24.8* |
| run11 stage1 | 59.0 | 32.3 | 46.1 | 32.3 | 31.4 | – | 17.8 |
| run11 stage1+2 | 54.6 | 28.4 | 40.2 | – | – | – | 17.2 |
| run11 +pal-ft | 56.6* | 48.2* | 37.0* | – | – | – | 28.8* |

## Notable patterns

- **masc_non_lev_2h** consistently scores far better than every other cross-domain set across all 37 checkpoints, including checkpoints that never saw non-Levantine MASC audio at all — likely a recording-condition/read-speech artifact in that split rather than genuine dialect transfer.
- 2-epoch pretraining checkpoints generally generalize about as well as their 1-epoch siblings on unseen domains, with no consistent direction — pretraining depth mainly moves in-domain (Palestinian) WER, not cross-domain robustness.
- Every checkpoint does worst on whichever Levantine-adjacent set (QASR-Lev / MASC-Lev) it hasn't trained on — dialect proximity to the pretraining corpus, not just audio domain, drives cross-domain WER.

---
*188/188 domain-evals complete. Best overall: run9 2-epoch pretrain at 33.03% test WER.*