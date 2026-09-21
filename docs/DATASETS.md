# Datasets

Every corpus and split, what built it, and what it was used for.
Paths are on the 5090 node unless marked R2.

---

## 1. Source corpora (full pools)

| corpus | clips | hours | speaker unit | distinct units |
|---|---|---|---|---|
| QASR | 1,041,377 | 1,241.7 | `recording_id` / real `speaker_id` | 3,545 recordings / 26,719 speaker_ids |
| MASC-C | 283,871 | 319.6 | `video_id` | 5,020 |
| **total** | **1,325,248** | **1,561.3** | | 31,739 |

On R2: `backup/transfer/curated_corpus/train/{qasr,masc}/{lev,non_lev}/*.parquet.zst`
(QASR 116 GiB, MASC 41.7 GiB). Not on the node — stream and delete per shard.

**Scan metadata is on the node and covers the whole pool** —
`~/scans/audio_v2/acoustic_full_scan/{qasr,masc}_{lev,non_lev}/row_probabilities.jsonl`
(acoustic DID) and `~/scans/text_v1/dialect_id_scans/text_dialect_scan_marbertv2_*`
(text DID). This is how corpora get *designed* without downloading audio.

### uid formats — a recurring source of bugs
- QASR: `qasr:<UUID>:<UUID_underscored>_utt_<n>_align` — genuinely per utterance.
- MASC: the corpus parquet stores the **bare `video_id`**, so uid is *video-level* and not
  unique per clip. Per-utterance MASC keys must be `(video_id, text)`.
- QASR `speaker_id` = `<recording_id>_speakerN_align`. Recording-scoped; not a person identity.
  Map at `~/spkmap/uid_speaker.jsonl` (1,055,971 rows), built by
  `scripts/research/selection/build_speaker_map.py`.

---

## 2. v3 curated corpus — the 300 h step-guided run

`~/v3_data/parquet/{train,val,test}.parquet` + `~/v3_data/audio` (34 GB, on-disk WAVs).

Selected by **0.3 × MARBERTv2 text + 0.7 × acoustic** Levantine score, averaged per speaker
(`recording_id` for QASR, `video_id` for MASC), top 310 h, speaker-disjoint, 5 h val + 5 h test
carved from the highest scorers, train sorted ascending by score.

| split | clips | hours | units | real speaker_ids | composite score |
|---|---|---|---|---|---|
| train | 263,941 | 301.04 | 1,468 (541 qasr + 927 masc) | 5,029 | 0.384 |
| val | 4,185 | 5.10 | 67 (7 qasr + 60 masc) | 175 | — |
| test | 4,650 | 5.07 | 113 (5 qasr + 108 masc) | 190 | 0.733 |

Built by `scripts/extract_full_acoustic_speaker_split.py` +
`scripts/build_full_acoustic_speaker_split.py`.
Clip length: mean 4.00 s (QASR) / 4.34 s (MASC).

**Note the test set is score-selected (composite 0.733 vs pool 0.190).** It favours
score-selected training data by construction.

---

## 3. Similarity sets — the 50 h ablation

`~/sim_sets/*.parquet`, audio reuses `~/v3_data/audio`. R2 `levantine_similarity_v1/sets/`.

Built by `scripts/research/selection/{build_target_vector,score_pool,carve_sim_sets}.py`.
Pool = the 202 h QASR + 99 h MASC in v3 train (263,941 clips, 5,029 speakers).

| set | speakers | QASR | MASC | clips | hours | centered sim |
|---|---|---|---|---|---|---|
| test (shared by all 3 runs) | 113 | 113 | 0 | 4,145 | 5.05 | +0.878…+0.670 |
| val (similarity) | 97 | 97 | 0 | 4,086 | 5.08 | +0.669…+0.613 |
| train_ascending / train_shuffled | 753 | 380 | 373 | 47,521 | 50.13 | +0.765…+0.152 |
| random_train | 638 | 466 | 172 | 44,063 | 50.01 | +0.004 |
| random_val | 85 | 62 | 23 | 4,773 | 5.47 | −0.064 |

`train_ascending` and `train_shuffled` are the *same rows* in different order.
Disjoint on **speaker_id and recording_id**: test↔train, test↔val, train↔val,
random_train↔test, random_train↔random_val, random_val↔test all 0/0.

Guard: `MIN_CLIPS=10` for split eligibility (5,029 speakers → 3,330), because a 1-clip
speaker tops any mean-score ranking by noise alone.

**Leakage warning:** these were carved from v3 *train*, so any model trained on the full v3
corpus has seen this test set. Only the three 50 h runs may be scored on it.

---

## 4. Naturalistic evaluation sets

`~/nat_resplit/parquet/*.parquet` + `~/nat_resplit/audio`.
Built by `scripts/research/data/resplit_naturalistic.py`.

| collection | clips | hours | speakers | mean clip | note |
|---|---|---|---|---|---|
| casa_pal train/val/test | 664/334/330 | 1.01/0.49/0.47 | — | 5.2–5.5 s | 44.1 kHz — **must resample** |
| casa_jor train/test | 847/848 | 1.00/0.98 | — | 4.2 s | no val split (0 rows) |
| layla train/val/test | 533/250/239 | 3.52/1.61/1.56 | 40/27/27 | 23.5 s | 109 speakers, 13 regions |
| omni train/val/test | 764/252/295 | 4.60/1.54/1.77 | 5/2/3 | 21.7 s | only 10 speakers total |

All speaker-disjoint by ID (verified). Layla's `speaker_key` column is the **shard filename**
and is useless; the real speaker is in the uid:
`layla_<region>_<SPK>_<SPK>_Laylawetheeb_<style>__seg<NN>`.

### Derived sets

- **`~/layla_told/layla_told_{train,val,test}.parquet`** — Layla with read-aloud dropped.
  109 speakers (all survive), 458 clips, **3.01 h** (train 1.59 / val 0.73 / test 0.69).
  All 109 speakers recorded both styles, so a paired read-vs-told comparison is available.
  Built by `scripts/research/data/build_layla_told.py`.
- **`~/sim_sets/omni_test_all_speakers.parquet`** — 25 fixed random clips from each of all 10
  Omni speakers. 250 clips, 1.50 h. Replaces the 3-speaker omni test.
  Built by `scripts/research/data/build_omni_all_speakers_test.py`.
- **`~/casa_orig/`** — Casablanca with its published splits intact (test 667/848, validation
  667/847). Built by `scripts/research/data/build_casa_original.py`.

### Omni per-speaker (total 7.91 h, mean **47.4 min/speaker**)

| split | speakers | clips | hours |
|---|---|---|---|
| train | spk02, spk03, spk06, spk08, spk10 | 764 | 4.60 |
| val | spk05, spk09 | 252 | 1.54 |
| test | spk01, spk04, spk07 | 295 | 1.77 |

**Open risk:** spk10 (train) and spk07 (test) match at x-vector 0.981, above the 0.94
same-voice reference. If they are one person, Omni's numbers leak. Needs a human to listen.

---

## 5. Random 300 h ablation corpus — designed, not extracted

`~/random300_matched/{train,val,test}_manifest.json` + `config.json`.
R2 `levantine_similarity_v1/random300_matched/`.
Built by `scripts/research/selection/build_random_300h.py` (seed 4242).

Matched to v3 on count, hours, source mix and per-unit size; random with respect to dialect
score. Unit = QASR `recording_id` / MASC `video_id`, same as v3.

| | v3 curated | random matched |
|---|---|---|
| units | 1,468 | **1,468** |
| QASR / MASC units | 541 / 927 | **541 / 927** |
| hours | 301.04 | **300.97** |
| QASR / MASC hours | 202.04 / 98.99 | 201.81 / 99.17 |
| min per unit | 12.3 | **12.3** |
| composite score | 0.384 | **0.221** |
| ≥0.8 | 34.3% | 17.3% |

**Disjoint from v3 val and test** — 0 shared units, 0 shared clips — so it can be scored on
v3's own benchmark and compared to 29.90 / 30.85.

Two things to know when reading a result from it:
- A uniform random draw gives 2.8 min/unit, so 1,468 units would yield only **82 h**.
  Size-matching is required, and it biases the draw *up* to composite 0.221 vs the pool's
  0.190 — against the hypothesis, so it cannot manufacture a win for curation.
- 12.9% clip overlap with v3 *train* (both draw from one pool). Harmless; evaluation sets are clean.

**Audio WAS extracted on 2026-09-21** into `~/rnd300_data/` (24 GB):
`parquet/train.parquet` 256,331 clips / **301.82 h**, 82,781 MASC + 173,550 QASR,
47 MASC rows dropped by the silence guard, zero QASR drops.
`parquet/train_ordered.parquet` is the same rows sorted into 6 ascending bands,
`rank_meta.json` defines them:

| band | speakers | hours | score range |
|---|---|---|---|
| 1 | 426 | 49.76 | 0.000 - 0.074 |
| 2 | 199 | 50.08 | 0.074 - 0.114 |
| 3 | 199 | 49.92 | 0.114 - 0.171 |
| 4 | 182 | 50.23 | 0.171 - 0.247 |
| 5 | 239 | 49.60 | 0.247 - 0.380 |
| 6 | 223 | 52.22 | 0.381 - 0.826 |

Built by `scripts/research/selection/build_rank_meta_6band.py` from
`~/random300_matched/assignments.json` via `scripts/extract_full_acoustic_speaker_split.py`.

**val/test for this run are v3's own sets**, not new ones. `~/rnd300_data/audio/{val,test}`
are symlinks into `~/v3_data/audio` — required, because `--audio-dir` applies to every
parquet the trainer opens, and forgetting it crashed this run once.

---

## 6. Vectors and score maps

All on R2 `levantine_similarity_v1/vectors/`.

| file | contents |
|---|---|
| `target_vectors.npz` | 109 Layla + 10 Omni speaker vectors (512-d), `target`, `target_balanced`, both centroids |
| `pool_scores.npz` | per-clip centered similarity for all 263,941 pool clips, plus the global mean and centered target |
| `speaker_scores.json` | per-speaker aggregates for all 5,029 speakers |
| `sample_weights_composite.json` | 1,323,517 keys → 0.3·text + 0.7·acoustic score, for `--sample-weights` |
| `uid_speaker.jsonl` (in `meta/`) | 1,055,971 QASR uid → speaker_id / recording_id |

**Target vector construction:** Layla told clips sliced into 5 s chunks, the 109 speakers
dealt round-robin into 5 groups with group *k* contributing only its *k*-th chunk (so the
target is not 109 people reciting the same sentence); Omni 10 s per speaker; average per
speaker, then average all 119 equally.

**Mean-centering is mandatory.** Raw cosine cannot separate Layla from QASR (0.739 vs 0.737)
because the embedding space is a narrow cone. Centering gives +0.103 vs −0.167 and 4.6× the
spread. See `scripts/research/audit/diagnose_target_collapse.py`.
