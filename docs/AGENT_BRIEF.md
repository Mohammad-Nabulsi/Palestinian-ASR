# Agent brief — start here

Paste-able bootstrap for a fresh agent picking up the Palestinian/Levantine ASR project.
Everything below is verified as of **2026-09-23**.

---

## 1. Read these first, in this order

| file | what it gives you |
|---|---|
| `docs/HANDOFF.md` | the single source of truth: what is established, what is not, known-bad numbers, bugs found, what to run next |
| `docs/RESULTS.md` | every run, its config, the code that produced it, the full WER matrix |
| `docs/DATASETS.md` | every corpus and split, provenance, uid-format traps, leakage warnings |
| `docs/RUNBOOK.md` | machine access, how to launch training and evals, R2 usage, gotchas |

Do not trust any earlier handoff note in this repo or on R2. These four supersede all of them.

---

## 2. Where everything lives

**GitHub** — `https://github.com/Mohammad-Nabulsi/Palestinian-ASR`
branch `feat/v3-310h-ascending-curriculum`, current commit `60ef86a`.
Research code is under `scripts/research/{selection,data,eval,audit}/`.

**Cloudflare R2** — `R2:backup/transfer/levantine_similarity_v1/` (99 objects, ~1.07 GiB)

```
docs/                     the four documents above
code/scripts_research/    mirror of the repo's research scripts
vectors/                  target vectors, per-clip pool scores, speaker scores, sample weights
meta/uid_speaker.jsonl    1,055,971 QASR uid -> real speaker_id / recording_id
sets/                     every split parquet + index (+ sets/v3_eval/ = the benchmark)
random300_matched/        manifests, config, assignments, rank_meta for the random-300h corpus
results/<run>/            train.log, all_results.json per run
adapters/<run>/           the trained LoRA adapters
RUN_REGISTRY.json         machine-readable config+data+results for every run
```

Source corpora (not on the node): `R2:backup/transfer/curated_corpus/train/{qasr,masc}/{lev,non_lev}/`
— QASR 116 GiB, MASC 41.7 GiB. Stream shard-by-shard and delete; never download wholesale.

**On the GPU node** — `~/v3_data` (34 GB, the curated 300 h), `~/rnd300_data` (24 GB, the random
300 h), `~/sim_sets`, `~/nat_resplit`, `~/layla_told`, `~/runs/<run>/`, `~/Palestinian-ASR`.

---

## 3. The GPU node

| | |
|---|---|
| GPU | 1 x RTX 5090, 32 GB, Blackwell **sm_120** — needs **cu128** wheels, cu124/cu126 silently lack kernels |
| CPU RAM | ~45 GB — the binding constraint. Above ~200 h you MUST use on-disk WAVs (`--audio-dir`) |
| Disk | 1.8 T, ~1.3 T free |
| Node | `nablusi@172.16.121.35` (RFC1918 — only reachable through the gateway) |
| Gateway | `student@htclogin.najah.edu` (212.14.244.143), password auth only |
| Python | `~/anaconda3/envs/work/bin/python` |

### Access
Two hops, always. The node has no public address.

On the workstation, `~/.ssh/config` already defines `nnu-gw` and `nnunode` (ProxyJump chain),
and `~/nnu.sh` wraps it. Credentials live in mode-600 files **outside any repo**:

| secret | file |
|---|---|
| gateway password | `~/.nnu_gw.txt` (workstation) |
| node key | `~/.ssh/nnu_ed25519` (workstation) |
| R2 keys | `~/.r2.env` (node) — `source` it before any rclone |
| HF token | `~/.hf.env` (node) — needed for gated models e.g. Cohere |

`~/.ssh/nnu_askpass.sh` feeds the gateway password to ssh non-interactively:
```bash
SSH_ASKPASS=~/.ssh/nnu_askpass.sh SSH_ASKPASS_REQUIRE=force DISPLAY=:0 ssh nnunode
```

**A key cannot be installed on the gateway** — the shared `student` home is not writable
(`Permission denied` on `~/.ssh`). Every connection is password-authenticated.

**If the direct route fails**, add `ProxyJump <host>` to the `nnu-gw` block and go via a machine
that can reach An-Najah. `209.20.159.144`, `RAG` (51.112.42.49) and `cor2dev2` have all worked.

### CURRENT BLOCKER (2026-09-23)
`htclogin.najah.edu` is **down or unreachable**: ICMP and all TCP ports including 22 time out,
verified from four unrelated source IPs with two SSH clients over 12+ hours. Earlier it accepted
TCP on 22 but never completed the SSH handshake. This is not a block on us — An-Najah IT
confirmed no ban. Nothing can run until that host is back. Do not burn hours re-testing; probe
occasionally and move on.

---

## 4. Running and monitoring a job

Always detach — an SSH drop must not kill a 10-hour run:
```bash
~/nnu.sh 'cd ~ && nohup <cmd> > ~/thing.log 2>&1 & echo started'
```

Training (this is the exact shape that produced every 300 h run):
```bash
~/nnu.sh 'cd ~/Palestinian-ASR && nohup ~/anaconda3/envs/work/bin/python -u \
  scripts/train_whisper_medium_lora_sequence.py \
    --train-parquet ~/rnd300_data/parquet/train_ordered.parquet \
    --rank-meta     ~/rnd300_data/rank_meta.json \
    --val-parquet   ~/v3_data/parquet/val.parquet \
    --test-parquet  ~/v3_data/parquet/test.parquet \
    --audio-dir     ~/rnd300_data/audio \
    --sequence 1,2,3,4,5,6,1,2,3,4,5,6 --n-chunks 6 --chunk-hours 50 \
    --batch-size 8 --lr 1e-4 --warmup-ratio 0.1 \
    --test-eval end --heartbeat-min 20 \
    --run-name NAME --out-dir ~/runs/NAME > ~/NAME.log 2>&1 & echo started'
```

Flags that matter:
- `--within-chunk-order file` — **required** for any score-ordered curriculum; the default
  (`shuffle`) reshuffles and destroys the ordering
- `--lr 1e-4` matches the 300 h baseline; `2e-5` is the better-calibrated value for new work
- `--audio-dir` — mandatory above ~200 h (45 GB RAM ceiling)
- `--sample-weights ~/sample_weights_composite.json --weight-floor 0.25 --weight-gamma 1.0
  --weight-norm batch` — optional per-sample loss weighting; ratio between a score-1.0 and a
  score-0.0 clip is exactly `1/floor`

Monitoring:
```bash
~/nnu.sh 'tail -3 ~/runs/NAME/train.log; nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
```
`pgrep -af NAME` matches its own command line — a single hit that IS your grep means nothing is
running. Trust GPU utilisation instead. A 12-stage 300 h run takes ~10 h and writes a
`SUMMARY <stage>: {...}` JSON line plus a checkpoint after every stage, so it resumes cleanly.

Evaluating:
```bash
~/nnu.sh 'set -a; source ~/.hf.env; set +a; export HF_HOME=$HOME/hf_cache
  ~/anaconda3/envs/work/bin/python ~/eval_model.py --parquet <test.parquet> --audio-dir <dir> \
    --model whisper-lora --adapter ~/runs/NAME/checkpoints/s12_c6/adapter \
    --out ~/eval_sweep/NAME__set.json --batch-size 16 --tag NAME'
```
Then `scripts/research/eval/fold_report.py` builds the matrix **as-scored and ta-marbuta-folded**
— always report both; folding alone moves base Whisper 45.45 -> 34.55.

After any new run: rerun `scripts/research/run_registry.py` and push
`RUN_REGISTRY.json`, the train log and the adapter to R2.

---

## 5. What is pending

1. **OOD sweep on the `rnd300_matched` adapter** — casa_pal, casa_jor, layla_told, omni_all.
   ~30 min. The script is ALREADY staged on the node at `~/ood_rnd300.sh`; just
   `nohup bash ~/ood_rnd300.sh > ~/ood_rnd300.log 2>&1 &`. This is the unbiased test: the v3 test
   set is itself score-selected (composite 0.733 vs the pool's 0.190), so it favours the curated
   arm by construction. **Run this before quoting the headline result as general.**
2. **Length ablation** — training clips average 4.0 s; Layla and Omni average 22–24 s. Concatenate
   short QASR clips to ~24 s and re-score. ~1 h, needs no new data, and may reframe every OOD
   number in the report.
3. **Install NetBird on the node** — it dials outbound to the user's self-hosted control plane,
   giving a stable `100.96.x.x` address immune to gateway outages. Do this first thing when
   access returns; it prevents a repeat of the current situation.

---

## 6. Standing rules

- **Never commit secrets.** `detect-secrets` is configured (`.secrets.baseline`); scan the staged
  set before every commit. Credentials go in mode-600 files, never in a repo, log, or chat.
- **Never brute-force the node password** — ~3 bad tries is a ~24 h fail2ban ban.
- Verify every new split disjoint on **both** speaker and recording id. QASR `speaker_id` is
  `<recording_id>_speakerN` — recording-scoped, not a person.
- MASC WAVs are 32-bit float: `sf.read(dtype="int16")` silently returns zeros. Decode native and
  peak-check. Casablanca is 44.1 kHz; an `sr != 16000` skip silently drops it all.
- MASC uid in the corpus parquet is the bare `video_id` — video-level, not per clip. Per-utterance
  joins must key on `(video_id, text)`.
- Report outcomes faithfully. This project has had six bugs that silently corrupted results;
  HANDOFF.md §5 lists them. Assume the next one exists.

---

## 7. The one-paragraph state of the science

Fine-tuning Whisper-medium on mined broadcast Arabic works: in-domain folded WER 45.45 -> 26.01.
**Selection matters, ordering does not.** At 300 h, selecting speakers by the DID composite score
beats a corpus matched on units, hours, source mix and per-unit size by **3.87 test WER**
(30.85 vs 34.72), ahead at all 12 stages. At 50 h, embedding-similarity selection beat random by
**5.0 points on Omni** but nothing in-domain. Curriculum ordering has failed twice under control
(26.30 vs 26.44, shuffled marginally better). Cohere zero-shot still beats us in-domain
(23.97 vs 26.01). The dialect classifier is near-binary — within one acoustically verified
speaker, clips score 0.00 and 1.00 minutes apart. The biggest untested confound is **utterance
length**: we train on 4 s clips and test OOD on 22–24 s ones.
