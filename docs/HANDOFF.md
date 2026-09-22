# Palestinian/Levantine ASR — single handoff

**Last updated:** 2026-09-21 (evening) · supersedes every earlier handoff note in this repo and on R2.

Read this file first. It is the only document that is kept current.
Details live in [RESULTS.md](RESULTS.md), [DATASETS.md](DATASETS.md), [RUNBOOK.md](RUNBOOK.md).

---

## 1. The problem

Build an ASR system for Palestinian/Levantine Arabic. There is almost no transcribed
Palestinian speech, but there are thousands of hours of transcribed **broadcast** Arabic
(QASR, MASC-C) containing Levantine speakers. The project mines that broadcast audio with
dialect identification, then LoRA fine-tunes `openai/whisper-medium` on what it finds.

The research question has narrowed over time to: **does it matter which hours you pick?**

---

## 2. What we know for sure

Ordered by how well established each one is.

### 2.1 Fine-tuning works; the base model is far behind
On our in-domain test set, folded WER **45.45 → 26.01**. Every fine-tune beats base
Whisper-medium on every set except Layla, where a randomly-selected corpus does nothing.

### 2.1b Selection by DID score helps at 300 h
Curated 300 h beats a random 300 h matched on units, hours, source mix and per-unit size by
**3.87 test WER** (30.85 vs 34.72), ahead at all 12 stages. At 50 h the same idea showed
nothing in-domain, so the effect appears to need scale — or the two experiments differ
because one selected by DID score and the other by embedding similarity.

### 2.2 Data **selection** helps out-of-domain; **ordering** does not
The cleanest experiment in the project. Three runs, identical 50 h budget, identical config,
same shared test set:

| test set | similarity-selected | random | selection gains |
|---|---|---|---|
| in-domain | −8.55 | −8.25 | 0.3 |
| casa_jor | −2.18 | −0.98 | 1.2 |
| casa_pal | −2.23 | −1.14 | 1.1 |
| layla told | −2.22 | **+0.10** | 2.3 |
| omni all | −7.45 | −2.44 | **5.0** |

(folded WER change vs base; negative = better)

Ascending-by-score ordering vs the same data shuffled: **26.30 vs 26.44** in-domain, and
shuffled is marginally *better* on four of five sets. Curriculum ordering has now failed
twice under controlled conditions. **Do not spend more GPU on ordering.**

Caveat: the similarity target was built *from* Layla+Omni, so winning on Layla+Omni is
partly circular. What it establishes is that targeted selection works when you can define
the target.

### 2.3 The dialect classifier is near-binary, and speaker-mean selection hides it
Roughly 60–98% of any speaker's clips score exactly 0.00 or 1.00. Verified not to be a
speaker-mixing artifact: within a single **acoustically verified** QASR speaker (x-vector
same-voice 0.94 vs random 0.73), clips scoring 0.00 and 1.00 match each other at 0.94–0.98.
Same voice, same microphone, minutes apart, opposite scores.

### 2.4 Volume buys very little past 50 h
300 h step-guided reaches **57.92** on Omni; a similarity-selected **50 h** reaches **58.72**.
Six times the data for 0.8 points.

### 2.5 Orthography is a large fraction of the apparent gap
Our references contain **zero** ta marbuta. Folding ة→ه and ى→ي moves base Whisper
45.45 → 34.55 (10.9 points of pure spelling) while our tuned models move 0.3. Always report
both numbers.

### 2.6 Cohere still wins in-domain
`CohereLabs/cohere-transcribe-arabic-07-2026` zero-shot: **23.97** folded on our test set vs
our best 26.01. A purpose-built Arabic system beats our fine-tuned Whisper on our own data.

### 2.7 Omni and Casablanca are acoustically very far apart
Casablanca scores 0.59–0.68 cosine to the Omni centroid — at or below the random-pair floor
of 0.73. This explains the OOD scatter better than dialect labels do.

### 2.8 Length is an unresolved confound
Training clips average **4.0 s**; Layla and Omni average **22–24 s**, pressed against
Whisper's 30 s window. Casablanca at 4–5 s matches training. This has never been controlled
for and may explain much of the OOD gap. **This is the most valuable untested hypothesis.**

---

## 3. What is NOT established

- That the DID composite score (0.3·text + 0.7·acoustic) is useful at all. Band selection,
  ascending ordering, and best-band-only training all produced null or negative results.
- That any of this transfers to real Palestinian conversational speech. 0.75% of the corpus
  is naturalistic; every test set is broadcast or read speech.
- Whether Omni's `spk07`/`spk10` are the same person (x-vector 0.981, above the same-voice
  reference). If they are, Omni's numbers leak. **Unresolved — needs a human to listen.**

---

## 4. Known-bad numbers — do not quote these

| number | why it is wrong |
|---|---|
| 300 h run scoring **15.07** on the new sim test set | 100% leakage: that test set was carved from v3's *training* pool. |
| `v3_continuation` run (29.46 in-domain) | Two concurrent processes shared one output dir; one skipped a stage. Provenance ambiguous. |
| Old `data_lev_custom_split_v1` val WER 23.75 | Per-row split; ~99–100% of val recordings also appear in training. |
| Any Layla WER from before the re-split | Computed on a 1.00 h subset, not the 6.70 h corpus. |

---

## 5. Bugs found and fixed (each would have silently corrupted results)

1. **MASC-C decoded as digital silence.** `sf.read(dtype="int16")` on float32 WAVs returns
   zeros. 38% of a 200 h train set and 66% of its test set were silence paired with real
   transcripts. Fixed by native-dtype decode + peak-amplitude guard + abort-on-majority-silent.
2. **`--init-from` clobbered the LR schedule**, inheriting the donor checkpoint's max_lr.
   A run degraded 29.90 → 33.13. Fixed in `3ae22a1`; LR bounds are now logged at startup.
3. **Sample-rate guard silently dropped all Casablanca.** Casablanca is 44.1 kHz; a
   `sr != 16000` skip scored 0 of 1,328 Palestinian clips and exited 0. Fixed with resampling.
4. **MASC weight keys never matched.** The corpus stores MASC uid as a bare `video_id`;
   the weight map prefixed it `masc_c:`. Every MASC clip trained at a uniform fallback weight.
5. **Speaker ranking topped by singletons.** A 1-clip speaker beats any real speaker on mean
   score. `MIN_CLIPS=10` now gates split eligibility.
6. **Target vector collapse.** Averaging 119 speaker vectors produced a generic-speech
   direction; raw cosine could not separate Layla from QASR (0.739 vs 0.737). Fixed by
   mean-centering — spread improved 4.6×.

---

## 6. Current state

### DONE — random-matched 300 h ablation (finished 2026-09-22 02:03, 9.7 h)
**Result: the curated corpus wins by 3.87 test WER points (30.85 vs 34.72) and is ahead at
all 12 of 12 stages.** First clear evidence the DID score is worth something at scale.
See RESULTS.md 2b, including why the v3 test set biases this in curation's favour. Run dir `~/runs/rnd300_matched`,
log `~/runs/rnd300_matched/train.log`.

Trains the size-matched **random** 300 h corpus through the **exact** loop that produced the
curated 300 h run — same script, same 6 x 50 h ascending banding, same 2 epochs, same
`--lr 1e-4`, same v3 val and v3 test. Only the speaker selection differs
(composite 0.221 vs 0.384).

Final: random val **32.70** / test **34.72** vs curated **29.90** / **30.85**.

Two incidents on this run, both handled, both worth knowing:
* It crashed at 15:58 with `FileNotFoundError` on a val WAV. `--audio-dir` applies to
  **every** parquet, and val/test are v3's, whose `audio_path` values live under
  `~/v3_data/audio`. Fixed by symlinking `val` and `test` into `~/rnd300_data/audio`;
  all 4,185 val and 4,650 test paths verified to resolve.
* On resume it went to `stage_pos=1` and **skipped stage 1's validation**. No training was
  lost. The `s1_c1` adapter was scored separately against the same val set to recover the
  number above. If you see a missing stage in any resumed run, this is why.

Everything below is built, verified and backed up to R2 under
`backup/transfer/levantine_similarity_v1/`.

### Ready to launch, awaiting a decision

| experiment | cost | what it settles |
|---|---|---|
| **Sample-weighted 50 h** | ~2 h | Whether a continuous score beats banding. Mechanism implemented and verified; recommend weighting by *similarity*, not DID score. |
| **Length ablation** | ~1 h | Concatenate short QASR clips to ~24 s and re-score. Needs no aligner and no new data. **Highest value per hour.** |

### Recommended order
1. OOD sweep on the new adapter (casa_pal, casa_jor, layla_told, omni_all) — ~30 min. The v3
   test set is itself score-selected, so it is biased toward the curated arm; the OOD sets are not.
2. Length ablation — cheapest remaining, and it may reframe every OOD number in this report.
3. Weighted run, using the **similarity** score rather than the DID score.

---

## 7. Where everything lives

| what | where |
|---|---|
| Code | this repo, branch `feat/v3-310h-ascending-curriculum`, `scripts/research/` |
| Machine setup, SSH, credentials policy | [RUNBOOK.md](RUNBOOK.md) |
| Every dataset and its provenance | [DATASETS.md](DATASETS.md) |
| Every run, config and score | [RESULTS.md](RESULTS.md) + `RUN_REGISTRY.json` |
| Vectors, scores, manifests, adapters | R2 `backup/transfer/levantine_similarity_v1/` |
| Working data on the node | `~/v3_data`, `~/sim_sets`, `~/nat_resplit`, `~/runs` |

`RUN_REGISTRY.json` is machine-readable and regenerated by
`scripts/research/run_registry.py`. Regenerate it after any new run.

---

## 8. Standing rules

- **Never commit secrets.** Credentials live in mode-600 files on the node; see RUNBOOK.
- **Never brute-force the node password** — ~3 bad tries triggers a ~24 h fail2ban ban.
- RTX 5090 is Blackwell **sm_120** and needs **cu128**. cu124/cu126 installs silently lack
  kernels. Verify with a real bf16 matmul, not just `torch.cuda.is_available()`.
- The node has ~45 GB RAM. Corpora above ~200 h must use on-disk WAVs (`--audio-dir`),
  not audio embedded in parquet.
- Report WER **as-scored and folded**. One number alone is misleading here.
- Every new split must be verified disjoint on **both** speaker and recording id. QASR
  `speaker_id` is `<recording_id>_speakerN`, so it is recording-scoped, not a person identity.
