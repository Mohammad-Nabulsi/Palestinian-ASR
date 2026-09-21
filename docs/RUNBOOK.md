# Runbook — the GPU node

> **No credential values are in this file, and none may ever be committed to this repo.**
> This documents *where* each secret lives and *how* to use it. Values live only in
> mode-600 files outside any git tree. See §6.

---

## 1. The machine

An-Najah National University HPC box.

| | |
|---|---|
| GPU | 1 × RTX 5090, 32 GB, Blackwell **sm_120** |
| CPU RAM | ~45 GB — this is the binding constraint, not the GPU |
| Disk | 1.8 T, ~1.3 T free at `/` |
| Node | `nablusi@172.16.121.35` (key-based login) |
| Gateway | `student@htclogin.najah.edu` (password) |
| Python | `~/anaconda3/envs/work/bin/python` |
| Repo on node | `~/Palestinian-ASR` |

**Two hops:** you cannot reach the node directly. The gateway is a jump host.

### Critical warnings
- **Never brute-force the node password.** ~3 bad attempts triggers a ~24 h fail2ban ban.
  Always use the key.
- The 5090 is **sm_120** and needs **cu128** wheels. cu124/cu126 installs silently lack
  kernels — `torch.cuda.is_available()` returns True and matmuls fail or fall back. Verify
  with `torch.cuda.get_arch_list()` *and* a real bf16 matmul:
  `scripts/research/../verify_gpu.py` pattern.
- 45 GB RAM means any corpus above ~200 h must use on-disk WAVs (`--audio-dir`), never audio
  embedded in the parquet. A 200 h embedded run was SIGKILLed once for exactly this.

---

## 2. Connecting

A helper `~/nnu.sh` on the Windows workstation wraps the two-hop SSH. It builds a
ProxyCommand through the gateway using `plink -pwfile` (non-interactive) and then keys into
the node.

Shape of it (values redacted — see §6 for where they come from):

```bash
PROXY="'/c/Program Files/PuTTY/plink.exe' -ssh -batch \
  -hostkey <GATEWAY_HOST_KEY> -pwfile <PATH_TO_GATEWAY_PASSWORD_FILE> \
  student@htclogin.najah.edu -nc %h:%p"

ssh -o BatchMode=yes -o ConnectTimeout=25 \
    -i ~/.ssh/nnu_ed25519 -o IdentitiesOnly=yes \
    -o StrictHostKeyChecking=accept-new \
    -o ProxyCommand="$PROXY" nablusi@172.16.121.35 "$@"
```

Usage:
```bash
~/nnu.sh 'nvidia-smi --query-gpu=memory.used,utilization.gpu --format=csv,noheader'
```

`scp` takes the same `-o ProxyCommand` and works for pushing scripts up and pulling
results down.

Every invocation prints `Keyboard-interactive authentication prompts from server:` on
stderr. It is harmless; pipe through `grep -v keyboard-interactive`.

### Long jobs
The SSH wrapper will time out long before a training run finishes. Always detach:

```bash
~/nnu.sh 'cd ~ && nohup <cmd> > ~/thing.log 2>&1 & echo started'
# then poll
~/nnu.sh 'tail -3 ~/thing.log; nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'
```

Note that `pgrep -af <name>` matches its own command line — a single hit that *is* your
grep means nothing is running. Check GPU utilisation to be sure.

---

## 3. Running a training job

```bash
~/nnu.sh 'cd ~/Palestinian-ASR && nohup ~/anaconda3/envs/work/bin/python -u \
  scripts/train_whisper_medium_lora_sequence.py \
    --train-parquet ~/sim_sets/train_ascending.parquet \
    --rank-meta     ~/sim_sets/train_ascending_rank.json \
    --val-parquet   ~/sim_sets/val.parquet \
    --test-parquet  ~/sim_sets/test.parquet \
    --audio-dir     ~/v3_data/audio \
    --sequence 1,1 --n-chunks 1 --chunk-hours 50 \
    --batch-size 8 --lr 2e-5 --warmup-ratio 0.1 \
    --within-chunk-order file --test-eval end --heartbeat-min 20 \
    --run-name my_run --out-dir ~/runs/my_run > ~/my_run.log 2>&1 & echo started'
```

### Flags that matter
| flag | why |
|---|---|
| `--within-chunk-order file` | keeps parquet row order — **required** for any score-ordered curriculum, otherwise the loader reshuffles and the curriculum is destroyed |
| `--lr 2e-5` | the calibrated value. **Not 1e-4.** |
| `--audio-dir` | on-disk WAVs; mandatory above ~200 h |
| `--rank-meta` | generate with `scripts/research/selection/build_rank_meta.py`; with `--n-chunks 1` the whole set trains as one continuous block |
| `--sample-weights` | optional uid→score map; see below |
| `--heartbeat-min 20` | liveness lines so a stalled run is visible |

### Per-sample loss weighting
```
--sample-weights ~/sample_weights_composite.json \
--weight-floor 0.25 --weight-gamma 1.0 --weight-norm batch
```
`w = floor + (1−floor)·score^gamma`, normalised so the mean weight is 1. The ratio between a
score-1.0 and a score-0.0 clip is exactly `1/floor`. `--weight-norm batch` divides by each
batch's own mean so step size is identical to an unweighted run — use it for clean A/Bs.

### First thing to check in any run log
```
grep -E "within_chunk_order|LR bounds|sample weighting|chunk 0" ~/runs/<run>/train.log
```
`LR bounds after init` exists because `--init-from` once silently inherited a donor run's LR.

---

## 4. Evaluating

```bash
~/nnu.sh 'set -a; source ~/.hf.env; set +a; export HF_HOME=$HOME/hf_cache
  ~/anaconda3/envs/work/bin/python ~/Palestinian-ASR/scripts/research/eval/eval_model.py \
    --parquet ~/sim_sets/test.parquet --audio-dir ~/v3_data/audio \
    --model whisper-lora --adapter ~/runs/my_run/checkpoints/s2_c1/adapter \
    --out ~/eval_sweep/my_run__newtest.json --batch-size 16 --tag my_run'
```

`--model` is `whisper-base` (untuned medium), `whisper-lora`, or `cohere`.
**Cohere needs `HF_HOME` and the token exported**, or it hits the gated repo and fails.

Then build the matrix, which reports as-scored and ta-marbuta-folded:
```bash
~/nnu.sh '~/anaconda3/envs/work/bin/python ~/Palestinian-ASR/scripts/research/eval/fold_report.py'
```

Sweep everything: `scripts/research/eval/eval_sweep.sh` (4 models × 5 sets).
Refresh the registry afterwards: `scripts/research/run_registry.py`.

---

## 5. R2 (Cloudflare)

`rclone` is at `~/bin/rclone` on the node and is **not** on `PATH` by default.

```bash
export PATH=$PATH:~/bin
set -a; source ~/.r2.env; set +a      # exports the R2 env vars rclone reads
rclone lsf  R2:backup/transfer/levantine_similarity_v1/
rclone copy <local> R2:backup/transfer/levantine_similarity_v1/<dest>/
```

`rclone` prints `NOTICE: Config file ... not found - using defaults` every call — it is
configured purely from environment variables, so this is expected.

### Layout
```
backup/transfer/
  curated_corpus/train/{qasr,masc}/{lev,non_lev}/   source shards (158 GiB)
  levantine_similarity_v1/
    vectors/      target vectors, pool scores, speaker scores, sample weights
    meta/         uid -> speaker_id map, carve log
    sets/         every split parquet + index
    random300_matched/
    adapters/     the three 50h LoRA adapters
    results/      matrix.json, per-run logs
    did_scan/
    RUN_REGISTRY.json
```

---

## 6. Credentials — where they live

**Nothing here may be committed.** All values sit in mode-600 files outside any git tree.

| secret | lives at | used for |
|---|---|---|
| Gateway password | `~/.nnu_gw.txt` (both workstation and node) | `plink -pwfile` ProxyCommand |
| Node SSH key | `~/.ssh/nnu_ed25519` (workstation) | login to 172.16.121.35 |
| R2 access key / secret / endpoint | `~/.r2.env` on the node | `source` before any `rclone` |
| HuggingFace token | `~/.hf.env` on the node | gated repos (Cohere) |
| GitHub token | workstation only, outside the repo | `git push` |

If any is missing, recreate it as a mode-600 file — do not paste values into a script, a
log, a commit, or a chat.

```bash
umask 077 && printf '%s' '<value>' > ~/.nnu_gw.txt
```

> **Rotate these.** The gateway password, the R2 access/secret pair, the HuggingFace token
> and a GitHub token were all pasted into chat transcripts during development. Treat them as
> exposed and reissue them. The gateway password is a *shared* university credential — it
> must never reach a public repo.

---

## 7. Gotchas worth knowing before you lose a day

- **MASC WAVs are 32-bit float.** `sf.read(dtype="int16")` silently returns zeros. Decode at
  native dtype and scale manually. Always peak-check for silence.
- **Casablanca is 44.1 kHz**, everything else is 16 kHz. An `sr != 16000` skip will silently
  discard the entire Palestinian set and exit 0.
- **MASC uid is video-level** in the corpus parquet. Joins needing per-utterance MASC must key
  on `(video_id, text)`.
- **Scan `row_idx` is unreliable** — observed 605 of 606 out of range. Join by uid or by
  `(speaker, text)`.
- **QASR `speaker_id` is recording-scoped.** Disjointness must be checked on *both*
  speaker_id and recording_id.
- **Windows-side Python writing to `/tmp`** fails silently in this setup. Use git-bash
  heredocs, `scp` the file, or the Write tool.
- Do not run two training jobs into the same `--out-dir`. It happened once: they halved each
  other's speed and mixed checkpoints, and one silently skipped a stage.
