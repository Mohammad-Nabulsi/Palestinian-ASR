# Speaker-grouped dialect ranking and hour-block selection

Follow-up to the speaker-disjointness audit (`scripts/r2_speaker_group_disjointness.py`,
which confirmed the current `data_lev_custom_split_v1` splits leak: ~100% of val's and
~99% of test's QASR recordings also have rows in train). This document covers the
ranking a *speaker-disjoint* split is cut from, the numbers it produces, and how it is
wired into the pipeline.

| | |
|---|---|
| scoring | [pipeline/speaker_scoring.py](pipeline/speaker_scoring.py) |
| pipeline stage | [pipeline/stages/speaker_select.py](pipeline/stages/speaker_select.py) |
| split routing | [pipeline/stages/split.py](pipeline/stages/split.py) |
| analysis CLI | [scripts/speaker_dialect_ranking.py](scripts/speaker_dialect_ranking.py) |
| outputs | `outputs/speaker_split_v1/` (production selection), `outputs/speaker_rank_*/` (score comparisons) |

## Where it sits in the pipeline

    dialect(text) --> dialect(audio) --> speaker_select --> split

An earlier iteration put speaker grouping *between* the two dialect passes, averaging
text scores per speaker before the audio model ran. That placement cannot work: deciding
which speakers we are most confident about needs both models' verdicts on the same
utterance, and it left `split` assigning rows by hash — which is what allowed one
recording into train and test at once. `speaker_select` now runs after both passes and
emits `speaker_assignments.json`; `split` routes every row to its speaker's split.

Rows whose speaker was ranked below the last block follow `unassigned_split` — `drop`
by default, because a speaker the selection did not choose should not silently appear in
a set. Leaves with no speaker key at all (omni, layla, casa/*) keep the old ratio hash.

## The score

    score = mean p(LEV)_text  x  mean p(Levantine)_audio  x  WilsonLB(agreed rows / rows)

`agreed` counts utterances where **both** models clear 0.80 on that same clip — the two
models landing on Levantine together, rather than each averaging high over different
rows. The Wilson lower bound is what makes sample count matter: a speaker with one lucky
utterance scores near zero however high that utterance is, while agreement repeated over
dozens of rows survives the discount. `score_sum_align` (mean of the two probabilities
instead of their product) selects 133 of the same 141 speakers, so that choice is not
load-bearing.

Per-block row floors keep the eval sets well-evidenced without throwing short speakers
away: val and test require 20+ dual-scored rows per speaker, train takes anyone. A
two-utterance speaker therefore skips the eval blocks and lands in train.

## Data source, and why not the curated parquet

`data_lev_custom_split_v1` stores only the scalar `text_lev` / `audio_lev` probability
per row, which is not enough for hard voting — an argmax over a 5-class softmax can be
LEV at p=0.3. The raw dialect-ID scans do carry each row's full distribution, so this
analysis reads those instead:

    R2: backup/transfer/dialect_id_scans.tar.zst    (see DIALECT_ID_SCANS.md in the bucket)

Its four **audio** runs already carry *both* modalities inline per row — audio
`top_label`/`label_scores{Levantine,MSA,Egyptian,Gulf,Maghrebi}`, text
`text_top_label`/`text_label_scores{LEV,GLF,EGY,MSA,MAGHREB}`, plus `duration_sec` — so
an audio-scan row **is** a row with both classifications (step 1). The two **text** runs
(1,881,995 rows, every clean QASR + MASC row) are read separately for each speaker's
unconditioned text statistics.

Grouping key (step 2) is the same as the disjointness script: QASR `recording_id`,
MASC-C `video_id`, both parsed out of `uid`.

## Pool (steps 1–2)

| | rows | speakers | hours |
|---|---:|---:|---:|
| audio-scan rows read | 250,731 | | |
| — usable (`status: ok`) | **224,386** | **8,068** | **276.6** |
| — dropped `skipped_short` | 26,345 | | |
| of which QASR | 160,574 | 3,545 recordings | 191.9 |
| of which MASC-C | 63,812 | 4,523 videos | 84.8 |
| rows whose **audio** argmax is Levantine | 67,097 | | **81.1** |

Duplicates across the four candidate-band runs: 0 — the bands are genuinely
complementary, so no speaker is double-weighted.

## Soft vs hard alignment (steps 3–4)

Soft = mean probability over the speaker's rows, thresholded at 0.80. Hard = majority
vote over the per-row argmax label. Speakers = 8,068.

| view | agreement | soft∧hard lev | hard-only lev | neither | rank corr (ρ) |
|---|---:|---:|---:|---:|---:|
| text, on the dual-scored pool | 79.9% | 6,443 | 1,625 | 0 | — (degenerate) |
| text, on the full text scan | 94.6% | 58 | 432 | 7,578 | 0.986 |
| audio | 89.4% | 158 | 857 | 7,053 | 0.955 |

**The text hard vote is meaningless on the dual-scored pool.** The audio model was only
ever run on rows the text model had already put at p(LEV) ≥ 0.5 *with LEV as argmax* —
all 224,386 of them — so every row votes LEV and every speaker comes out 100% text-hard-LEV.
The informative text hard vote is the one over the full text scan, which sees each
speaker's non-LEV rows too; the script reports both and ranks on the full-scan one.

Neither modality has a "soft yes / hard no" speaker: mean p ≥ 0.8 always implies a
majority argmax. The disagreement is entirely the other direction — speakers whose rows
mostly *say* Levantine without the mean clearing 0.80.

## Ranking agreement (step 6)

Spearman ρ between the three rankings:

| pair | all 8,068 speakers | speakers with ≥20 rows (n=3,478) |
|---|---:|---:|
| text hard vs audio hard | **0.055** | **0.271** |
| combined vs text hard | 0.026 | 0.288 |
| combined vs audio hard | 0.955 | 0.992 |

Two things fall out of this. First, the text and audio dialect classifiers rank speakers
**almost independently** — agreeing on ~5% of the rank ordering across the full pool, and
only ~27% once tiny groups are removed. Second, because the pool's text probabilities are
already conditioned to be high (≥0.5 by construction) while audio probabilities span the
full range, the product in step 5 is driven almost entirely by the audio term. Sorting by
`combined_prob` is, in practice, sorting by audio.

## The production selection

`outputs/speaker_split_v1/`, from the config the pipeline now runs:

    --rank-by score_product_align --block-hours 8 8 200 \
    --block-names val test train --block-min-rows 20 20 0

| block | speakers | hours | utterances | agreed hours | audio argmax Levantine | rows/speaker | score range |
|---|---:|---:|---:|---:|---:|---|---|
| val | 73 | 8.03 | 6,550 | 5.25 (65%) | 81% | 20 / 50 / 331 | 0.322 – 0.698 |
| test | 68 | 8.07 | 6,642 | 4.40 (55%) | 72% | 20 / 90 / 319 | 0.239 – 0.321 |
| train | 3,978 | 200.01 | 165k | 38.68 (19%) | 33% | 1 / 29 / 419 | 0.000 – 0.734 |
| unassigned | 3,949 | 60.53 | — | — | — | — | below the train cut |

Source mix: val 39 MASC + 34 QASR, test 24 MASC + 44 QASR, train 1,453 MASC + 2,525 QASR.
Train's *maximum* score exceeds val's because high-scoring speakers with fewer than 20
rows skip the eval blocks and land there.

**A 200h train block cannot be selective.** The dual-scored pool is 276.6h total, so 200h
is 72% of it: the train block runs from 0.734 down to ~0, and by audio argmax it is 38%
MSA against 33% Levantine. It is "everything else that isn't val or test", not a
confidence-selected set. If train is meant to be selective, it has to be much smaller —
the cumulative curve below gives the trade — or the pool has to grow (the audio model
never ran on rows the text model scored below 0.5).

### Disjointness

Verified rather than assumed — `verify_disjoint` cross-checks every block pair and is a
hard failure in both the stage and the CLI:

    pairwise shared speakers: val vs test 0, val vs train 0, test vs train 0
    speakers in more than one block: none
    speaker keys seen under two sources: none

The end-to-end wiring is tested the same way: a fixture run of `speaker_select` -> `split`
asserts that no recording appears in two split directories and that every written row
landed in the split its speaker was assigned.

### Cumulative hours down the ranking (raw mean-product ranking)

| target | speakers needed | score at cut |
|---:|---:|---:|
| 1h | 82 | 0.808 |
| 2h | 101 | 0.790 |
| 4h | 139 | 0.738 |
| 8h | 195 | 0.685 |
| 16h | 319 | 0.622 |
| 24h | 440 | 0.567 |
| 50h | 738 | 0.475 |
| 100h | 1,525 | 0.303 |
| 200h | 4,023 | 0.093 |

16h (8+8) is ~6% of the 276.6h dual-scored pool, so there is ample headroom to widen the
blocks or add a third one.

## Caveats that affect how far these numbers can be pushed

1. **QASR grouping is per *recording*, not per person.** The scan `uid` only exposes
   `recording_id`. A broadcast anchor recurs across episodes, so recording-disjoint
   blocks still let the same physical person appear in more than one split. The real
   `speaker_id` (and `normalizedName`) exists in the QASR ingest schema
   (`preprocess/qasr_segment_to_arrow.py`) — closing this needs a uid → speaker_id join
   against the curated QASR shards, after which `speaker_select` groups on that instead
   and nothing else has to change.
2. **MASC-C has no speaker id at all**; `video_id` is the closest proxy, and one channel
   can publish many videos of the same presenter.
3. **The pool is conditioned on text ≥0.5.** The audio model never ran on the rest, so a
   speaker's audio statistics only ever describe their text-selected utterances. The
   `audio_coverage` field per speaker (dual-scored rows / all text-scored rows) shows how
   thin that slice is.
4. **The 81.1h ceiling.** Only 81.1h of the 276.6h pool has audio actually voting
   Levantine, which bounds any selection built on audio agreement.

## Reproducing

```bash
# fetch + unpack the scans (pyarrow's open_input_stream auto-decompresses .zst)
python - <<'PY'
import os, shutil, pyarrow.fs as fs
s3 = fs.S3FileSystem(access_key=os.environ["R2_ACCESS_KEY_ID"],
                     secret_key=os.environ["R2_SECRET_ACCESS_KEY"],
                     endpoint_override=os.environ["R2_ENDPOINT"], scheme="https")
with s3.open_input_stream("backup/transfer/dialect_id_scans.tar.zst") as src, open("scans.tar", "wb") as dst:
    shutil.copyfileobj(src, dst, length=8 * 1024 * 1024)
PY
tar -xf scans.tar

cd scripts && python speaker_dialect_ranking.py \
    --audio-scan ../dialect_id_scans/dialect_scan_badrex_mms300m_*/row_probabilities.jsonl \
    --text-scan  ../dialect_id_scans/text_dialect_scan_marbertv2_written_clean_*/row_probabilities.jsonl \
    --rank-by score_product_align \
    --block-hours 8 8 200 --block-names val test train --block-min-rows 20 20 0 \
    --emit-samples 50 --out-dir ../outputs/speaker_split_v1
```

Runtime is ~4 minutes, dominated by parsing the 1.88M-line text scans. Each run writes
`summary.json`, `selection.csv` and `speaker_assignments.json` (tracked in git),
`samples.jsonl` when `--emit-samples` is set, and three full per-speaker JSONL rankings
(untracked, ~7MB each).

In the pipeline the same selection comes from the `speaker_select` stage, which reads the
audio stage's `row_probabilities.jsonl` directly — no text scan needed, since it ranks on
the dual-scored pool alone:

```bash
python -m pipeline run --config configs/full.yaml --only speaker_select
python -m pipeline run --config configs/full.yaml --only split
```
