# Results

Every number below is WER % unless stated. `as-scored / folded` — folded means ة→ه and ى→ي
applied to both reference and hypothesis before scoring. Our references contain zero ta
marbuta, so the unfolded number penalises models that spell it correctly.

Machine-readable copy: `RUN_REGISTRY.json` (regenerate with `scripts/research/run_registry.py`)
and `eval_sweep/matrix.json` on R2.

---

## 1. The 50 h selection ablation — the cleanest experiment

Three runs, same 50 h budget, same LoRA config, same LR, same **shared** test set, all
speaker- and recording-disjoint from it.

**Produced by:** `scripts/research/run_50h_selection_ablation.sh`
→ `scripts/train_whisper_medium_lora_sequence.py`

| | train data | config difference |
|---|---|---|
| `sim_ascending` | `sim_sets/train_ascending.parquet` | `--within-chunk-order file` (curriculum preserved) |
| `sim_shuffled` | `sim_sets/train_shuffled.parquet` | `--within-chunk-order shuffle` |
| `random_50h` | `sim_sets/random_train.parquet` | random 50 h, own random 5 h val |

Shared: `--batch-size 8 --lr 2e-5 --warmup-ratio 0.1 --n-chunks 1 --chunk-hours 50
--sequence 1,1 --test-eval end`, LoRA r=32 α=32 dropout 0.05 on q,k,v,o,fc1,fc2, bf16,
gradient checkpointing, OneCycleLR.

### Training curve

| run | epoch 1 val | epoch 2 val | test |
|---|---|---|---|
| sim_ascending | 23.63 | 21.96 | 26.29 |
| sim_shuffled | 23.08 | 21.80 | 26.41 |
| random_50h | 31.71 | 30.53 | 26.57 |

`random_50h` validated on its **own** random val set, so 30.53 is not comparable to 21.80 —
different corpora, not a worse model. The test column is the controlled comparison.

### Full evaluation matrix (as-scored / folded)

| test set | base | Cohere | v3 300 h | sim_ascending | sim_shuffled | random_50h |
|---|---|---|---|---|---|---|
| in-domain sim test (4,145) | 45.45 / 34.55 | 36.95 / **23.97** | *15.07 — LEAKED* | 26.30 / 26.01 | 26.44 / 26.10 | 26.61 / 26.31 |
| casa_pal (330) | 59.91 / 59.70 | — | — | 61.22 / 57.46 | 60.93 / **57.08** | 62.29 / 58.56 |
| casa_jor (848) | 51.04 / 49.97 | — | — | 51.09 / **47.78** | 51.22 / **47.78** | 52.25 / 48.99 |
| layla told (105) | 52.94 / 52.69 | — | — | 54.06 / 50.46 | 53.40 / **49.79** | 56.47 / 52.78 |
| omni all-spk (250) | 66.84 / 66.42 | — | **62.27 / 57.92** | 63.36 / 58.97 | 63.09 / 58.72 | 68.04 / 63.98 |

CER, folded: in-domain base 16.68 → tuned 10.75; omni base 31.44 → tuned 21.20.

### Reading it

- **Selection works OOD.** +5.0 points on Omni and +2.3 on Layla over random. Random
  selection fails outright on Layla (+0.10 vs base).
- **Ordering does nothing.** 26.30 vs 26.44, and shuffled wins on four of five sets.
- **In-domain is blind to selection.** All three land within 0.3 points. Judge on OOD.
- The similarity target was built from Layla+Omni, so the Layla/Omni gains are partly
  circular. The Casablanca gains (1.1–1.2) are not.

---

## 2. The 300 h step-guided run

**Produced by:** `scripts/train_whisper_medium_lora_sequence.py`, 6 × 50 h bands ascending by
composite DID score, 2 epochs, 12 stages, lr 1e-4, 10.8 GPU-hours.
Run dir `~/v3_run`, 12 checkpoints.

| stage | cum h | val WER | val CER |
|---|---|---|---|
| s1_c1 | 50 | 39.27 | 17.49 |
| s2_c2 | 100 | 37.25 | 17.07 |
| s3_c3 | 150 | 35.34 | 16.35 |
| s4_c4 | 200 | 34.64 | 16.25 |
| s5_c5 | 250 | 33.72 | 15.91 |
| s6_c6 | 300 | **30.61** | 14.72 |
| s7_c1 | 350 | 32.53 ← epoch 2 restarts on band 1 | 15.96 |
| s8_c2 | 400 | 31.15 | 15.10 |
| s9_c3 | 450 | 30.93 | 15.10 |
| s10_c4 | 500 | 30.62 | 14.96 |
| s11_c5 | 550 | 30.43 | 15.11 |
| s12_c6 | 600 | **29.90** | 14.69 |

Final test **30.85** on its own 5.07 h test set.

The +1.9 jump at s7 (epoch 2 restarting on the weakest band) is the one place in the project
where ordering visibly moves validation WER. Note it did **not** translate into a better final
model than shuffling, per §1.

### Related runs on the same corpus

| run | description | val | test |
|---|---|---|---|
| `v3_best50h_c6` | highest-scoring 50 h only, 2 epochs | 32.53 | 32.98 |
| `v3_continuation` | continued on c5+c6 then c6 | 29.53 | 29.47 **(contaminated — see HANDOFF §4)** |

`v3_best50h_c6` is the direct refutation of "quality beats volume": training on the *best*
50 h is worse than training on all 300 h, everywhere.

---

## 2b. Random-matched 300 h ablation — COMPLETE

Started 2026-09-21 15:11. **Identical to the curated 300 h run except the training data.**

| | |
|---|---|
| script | `scripts/train_whisper_medium_lora_sequence.py` (unchanged behaviour: `within_chunk_order=shuffle`, weighting off) |
| launcher | `scripts/research/run_random_300h_ablation.sh` |
| train | `~/rnd300_data/parquet/train_ordered.parquet` 256,331 clips / 301.82 h / 1,468 units |
| rank meta | `~/rnd300_data/rank_meta.json`, 6 x 50 h ascending |
| val / test | `~/v3_data/parquet/{val,test}.parquet` - the **same** sets the 300 h run used |
| config | `--sequence 1,2,3,4,5,6,1,2,3,4,5,6 --n-chunks 6 --chunk-hours 50 --batch-size 8 --lr 1e-4 --warmup-ratio 0.1 --test-eval end` |
| difference vs curated | composite score 0.221 vs 0.384; ≥0.8 fraction 17.3% vs 34.3% |

| stage | cum h | random | curated | delta |
|---|---|---|---|---|
| s1_c1 | 50 | 40.73 | 39.27 | +1.46 |
| s2_c2 | 100 | 41.99 | 37.25 | +4.74 |
| s3_c3 | 150 | 41.01 | 35.34 | +5.67 |
| s4_c4 | 200 | 37.51 | 34.64 | +2.87 |
| s5_c5 | 250 | 35.02 | 33.72 | +1.30 |
| s6_c6 | 300 | 33.63 | 30.61 | +3.02 |
| s7_c1 | 350 | 34.38 | 32.53 | +1.85 |
| s8_c2 | 400 | 34.68 | 31.15 | +3.53 |
| s9_c3 | 450 | 34.83 | 30.93 | +3.90 |
| s10_c4 | 500 | 33.90 | 30.62 | +3.28 |
| s11_c5 | 550 | 33.07 | 30.43 | +2.64 |
| **s12_c6** | **600** | **32.70** | **29.90** | **+2.80** |
| **final test** | | **34.72** | **30.85** | **+3.87** |

Ran 34,984 s (9.7 h). The curated corpus is better at **all 12 of 12 stages** — the
consistency matters more than any single number. s1_c1 was scored separately because a
resume skipped its in-loop validation.

**This is the first clear positive result for the DID score.** At 300 h, selecting speakers by
0.3*text + 0.7*acoustic Levantine score is worth **3.87 WER points on test** and 2.80 on val
against a corpus matched on units (1,468), hours (301), source mix and per-unit size — with
only the score differing (composite 0.384 vs 0.221).

It also contradicts the 50 h experiment, where selection showed nothing in-domain. Two
readings, and we cannot yet separate them: the effect may need scale to appear, or the 50 h
sets were selected by *embedding similarity* while these were selected by *DID score* — two
different criteria measured on two different benchmarks.

Caveat that limits the claim: the v3 test set is itself score-selected (composite 0.733 vs
the pool's 0.190), so it favours score-selected training data by construction. The honest
statement is "picking high-DID-score data helps on a high-DID-score benchmark". The OOD sweep
on this adapter is the unbiased test and has not been run yet.

---

## 3. Zero-shot dialect ID over the naturalistic sets

**Produced by:** `scripts/research/eval/ood_did_scan.py` — 8,385 clips, 15 splits, 71 s.
Models: `badrex/mms-300m-arabic-dialect-identifier` (acoustic),
`IbrahimAmin/marbertv2-arabic-written-dialect-classifier` (text).

| set | hrs | audio Levantine mean | audio median | text mean |
|---|---|---|---|---|
| casa_pal test | 0.98 | 0.589 | 0.746 | 0.854 |
| casa_jor test | 0.98 | 0.449 | 0.303 | 0.879 |
| layla (train/val/test) | 6.70 | 0.381 / 0.360 / 0.221 | 0.026 / 0.037 / 0.012 | ~0.87 |
| omni (train/val/test) | 7.91 | 0.978 / 0.990 / 0.984 | 0.998 | ~0.95 |

Three findings:

1. **The text classifier is nearly useless here** — 86–96% argmax LEV on *every* set including
   Jordanian and Layla. It carries 30% of the selection weight and contributes far less signal.
2. **Layla reads as Gulf** — argmax Gulf 64%, Levantine 34%, despite being nominally Jordanian.
   Layla is also the one set where fine-tuning hurts.
3. **High dialect score does not predict our WER.** Omni is 0.982 (saturated Levantine) and is
   our *worst* test set.

Near-binary behaviour reproduces here: 54% / 59% / 82% / 98% of clips at 0.00 or 1.00 for
casa_pal / casa_jor / layla / omni.

---

## 4. Score distributions across corpora

**Produced by:** `scripts/research/selection/build_sample_weights.py` and the scan JSONLs.

| corpus | acoustic mean/median | text mean | **composite** mean/median | ≥0.8 |
|---|---|---|---|---|
| entire QASR+MASC pool (1,325,241 clips) | 0.206 / 0.010 | 0.154 | 0.190 / 0.037 | 14.3% |
| random matched 300 h | 0.247 / 0.017 | 0.160 | 0.221 / 0.061 | 17.3% |
| v3 curated train | 0.428 / 0.225 | 0.282 | 0.384 / 0.304 | 34.3% |
| v3 curated test | 0.784 / 0.991 | 0.612 | 0.733 / 0.787 | 72.0% |

Even the hand-picked 300 h is only 34.3% ≥0.8 with a median of 0.304 — the near-binary
classifier showing up at corpus scale.

**Warning for the planned ablation:** the v3 test set is itself score-selected (composite
0.733). A curated-vs-random comparison on it is partly circular. Judge on the four
independent OOD sets.

---

## 5. Speaker-embedding audit

**Produced by:** `scripts/research/audit/verify_speaker_clusters.py` (`microsoft/wavlm-base-plus-sv`).

Calibration on this data — **same voice 0.94, different voice same recording 0.755,
random pair 0.73**. The embedding space is a narrow cone; 0.73 is the floor, not 0.

- QASR `speaker_id` is `<recording_id>_speakerN_align` — diarisation output, **recording-scoped**.
  All 1,671 ids sit inside exactly one recording; the same anchor across 50 broadcasts gets 50
  different ids. It cannot give cross-recording disjointness.
- Within a recording it *is* one voice: same-id 0.942 vs different-id-same-recording 0.755.
- 1 of 27 clusters was impure.
- **Within one verified voice, 0.00-scoring and 1.00-scoring clips match each other at 0.94–0.98.**
  The dialect score swings floor-to-ceiling while the voice does not move.

Cosine to the Omni-test centroid: omni/test 0.964, omni/train 0.885, layla/test 0.786,
casa_jor 0.649, casa_pal 0.592. Casablanca is below the random-pair floor.

---

## 6. Retrieval: how much QASR is Omni-like

**Produced by:** `scripts/research/selection/score_pool.py` (202 h QASR pool on disk).

| cumulative | clips | sim range | mean |
|---|---|---|---|
| 0–10 h | 7,860 | 0.984–0.899 | 0.928 |
| 10–20 h | 8,791 | 0.899–0.774 | 0.850 |
| 20–30 h | 9,280 | 0.774–0.682 | 0.716 |
| 30–40 h | 8,917 | 0.682–0.648 | 0.663 |
| 40–50 h | 8,793 | 0.648–0.625 | 0.636 |

Only **9.9 h** clears 0.90 and **21.3 h** clears 0.75. Past hour 26 the retrieved clips are
*less* similar to Omni than two random recordings are to each other. **50 h of Omni-like QASR
does not exist in this pool.**

Scope caveat: this searched the 202 h already selected by Levantine score, i.e. 16% of QASR's
1,241.7 h, filtered on a different axis than the one being searched.
