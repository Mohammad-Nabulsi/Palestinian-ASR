# Speaker-grouped dialect ranking and hour-block selection

Follow-up to the speaker-disjointness audit (`scripts/r2_speaker_group_disjointness.py`,
which confirmed the current `data_lev_custom_split_v1` splits leak: ~100% of val's and
~99% of test's QASR recordings also have rows in train). This document covers the
ranking that a *speaker-disjoint* split can be cut from, and what the numbers came out
to be.

Script: `scripts/speaker_dialect_ranking.py`. Outputs: `outputs/speaker_rank_*/`.

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

## Hour blocks (steps 7–8)

Blocks are consecutive slices of one ranking, so they are speaker-disjoint by
construction. Three configurations were run, all carving 8h + 8h:

| config | ranked by | pool | 1st 8h block | 2nd 8h block | rows/speaker (min, mean) |
|---|---|---:|---|---|---|
| **A** (as specified) | `combined_prob` | 8,068 spk / 276.6h | 195 spk, 8.06h | 125 spk, 8.10h | 1, 34.7 |
| **B** | `combined_prob`, ≥20 rows/spk | 3,478 spk / 224.1h | 86 spk, 8.08h | 73 spk, 8.09h | 20, 76.5 |
| **C** | `combined_vote_lb` (Wilson) | 8,068 spk / 276.6h | 87 spk, 8.34h | 76 spk, 8.12h | 6, 82.2 |

Source mix of the selected blocks:

| config | 1st block | 2nd block |
|---|---|---|
| A | 166 MASC (3.11h) + 29 QASR (4.94h) | 87 MASC (2.49h) + 38 QASR (5.61h) |
| B | 51 MASC (2.41h) + 35 QASR (5.67h) | 33 MASC (1.97h) + 40 QASR (6.12h) |
| C | 65 MASC (2.26h) + 22 QASR (6.08h) | 44 MASC (1.03h) + 32 QASR (7.09h) |

Restricting each selection to rows whose audio argmax is actually Levantine costs about
1.3–2.6h per block (e.g. config B: 8.08h → 6.73h and 8.09h → 5.91h).

**Config A ranks tiny groups to the top.** Its top five are MASC videos with 1, 1, 2, 9
and 2 dual-scored rows — a few seconds each — whose full-text LEV vote fraction is
0.20, 0.05, 0.50, 0.71 and 0.31. A raw mean has no notion of evidence, so one lucky
utterance outranks a speaker with 280 consistent ones. That is why A needs 195 speakers
to reach 8h where B and C need ~86.

Config C fixes this without a hard cutoff by ranking on the product of **Wilson lower
bounds** of the two vote fractions, which grows with group size. Its top entries are
groups of 12–280 rows. A and C agree on only 133 of their selected speakers (A picks 320
for 16h, C picks 163), so the choice of score materially changes the eval set.

Recommendation: use **C** (or B if a hard minimum is preferred) for anything that will be
reported as an eval set; A is the literal reading of the spec but is not defensible for
val/test.

### Cumulative hours down the primary ranking (config A)

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
   (`preprocess/qasr_segment_to_arrow.py`) and in `pipeline/stages/speaker_aggregate.py`'s
   grouping — closing this needs a uid → speaker_id join against the curated QASR shards.
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
    --rank-by combined_vote_lb --block-hours 8 8 --block-names train val \
    --out-dir ../outputs/speaker_rank_C_wilson
```

Runtime is ~4 minutes, dominated by parsing the 1.88M-line text scans. Each run writes
`summary.json`, `selection.csv` (the speakers that landed in a block — tracked in git)
and three full per-speaker JSONL rankings (untracked, ~7MB each).
