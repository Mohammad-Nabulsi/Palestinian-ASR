# Dataset Download and Preparation Order

This file records how each dataset in this workspace has been downloaded or prepared so far.

## 1. QASR

QASR was downloaded with bash commands into a local `QASR/` directory.

```bash
mkdir -p QASR && cd QASR

wget -c -O qasr_annotation_v1.0.tar.bz2 "https://arabicspeechdata.blob.core.windows.net/data/qasr_annotation_v1.0.tar.bz2?sp=rl&st=2025-03-08T09:23:55Z&se=2030-03-08T17:23:55Z&spr=https&sv=2022-11-02&sr=c&sig=D8KB7c2B5f4c6ikXd4nHl8QR620lTk3C0SMaB1rZeN0%3D"

wget -c -O qasr_wav_v1.0.tar.bz2.part_aa "https://arabicspeechdata.blob.core.windows.net/data/qasr_wav_v1.0.tar.bz2.part_aa?sp=rl&st=2025-03-08T09:23:55Z&se=2030-03-08T17:23:55Z&spr=https&sv=2022-11-02&sr=c&sig=D8KB7c2B5f4c6ikXd4nHl8QR620lTk3C0SMaB1rZeN0%3D"

wget -c -O qasr_wav_v1.0.tar.bz2.part_ab "https://arabicspeechdata.blob.core.windows.net/data/qasr_wav_v1.0.tar.bz2.part_ab?sp=rl&st=2025-03-08T09:23:55Z&se=2030-03-08T17:23:55Z&spr=https&sv=2022-11-02&sr=c&sig=D8KB7c2B5f4c6ikXd4nHl8QR620lTk3C0SMaB1rZeN0%3D"
```

The pasted `part_ab` line in the earlier note was corrupted, so the cleaned command above should be used instead.

**Confirmed (2026-07-26): the archive has exactly 4 split parts, not 2.** Listed directly from the
container (the SAS token above is container-level, `sr=c`, so `GET ...?restype=container&comp=list&<same query params>`
enumerates every blob — no guessing needed):

| part | bytes |
|---|---|
| `qasr_wav_v1.0.tar.bz2.part_aa` | 48,318,382,080 |
| `qasr_wav_v1.0.tar.bz2.part_ab` | 48,318,382,080 |
| `qasr_wav_v1.0.tar.bz2.part_ac` | 48,318,382,080 |
| `qasr_wav_v1.0.tar.bz2.part_ad` | 25,648,868,607 |

Total compressed: 170,604,014,847 bytes (~159GiB). Only `part_aa`+`part_ab` were ever downloaded to
this box historically, which is exactly why QASR audio extraction only ever produced 961/3,545, then
2,021/3,545 matched wav files — the archive was genuinely incomplete on disk, not corrupted. Download
the missing parts with the same pattern:

```bash
wget -O qasr_wav_v1.0.tar.bz2.part_ac "https://arabicspeechdata.blob.core.windows.net/data/qasr_wav_v1.0.tar.bz2.part_ac?sp=rl&st=2025-03-08T09:23:55Z&se=2030-03-08T17:23:55Z&spr=https&sv=2022-11-02&sr=c&sig=D8KB7c2B5f4c6ikXd4nHl8QR620lTk3C0SMaB1rZeN0%3D"
wget -O qasr_wav_v1.0.tar.bz2.part_ad "https://arabicspeechdata.blob.core.windows.net/data/qasr_wav_v1.0.tar.bz2.part_ad?sp=rl&st=2025-03-08T09:23:55Z&se=2030-03-08T17:23:55Z&spr=https&sv=2022-11-02&sr=c&sig=D8KB7c2B5f4c6ikXd4nHl8QR620lTk3C0SMaB1rZeN0%3D"
```

**Do not add `-c` to those `wget` commands.** A/B tested directly against this endpoint (alternating
the flag on/off, fresh output file each run): plain `wget` sustains ~15-18MB/s, `wget -c` on a file
that doesn't exist yet (so there's nothing to actually resume) crawls at ~0.3-0.6MB/s — about 30x
slower, apparently from however this specific Azure Blob endpoint handles the Range-request probe
`-c` sends. A full restart of a ~45min download is cheaper than a "resumable" one running 30x slower,
so download fresh (delete any partial file first) rather than resuming.

After all 4 parts are downloaded, extract with parallel decompression across all CPUs (do not
materialize a combined `qasr_wav_v1.0.tar.bz2` first — just `cat` the parts straight into the pipe,
no need for the 159GB of scratch space that would take):

```bash
cd QASR
cat qasr_wav_v1.0.tar.bz2.part_aa qasr_wav_v1.0.tar.bz2.part_ab \
    qasr_wav_v1.0.tar.bz2.part_ac qasr_wav_v1.0.tar.bz2.part_ad \
  | pv -i 120 -f -s 170604014847 \
  | pbzip2 -dc -p$(nproc) \
  | tar -xf - -C wav_all
tar -xjf qasr_annotation_v1.0.tar.bz2
```

Note: `pbzip2 -p$(nproc)` only actually parallelizes if the source file was itself compressed with
`pbzip2` (multi-stream). This QASR archive was compressed with plain single-threaded `bzip2`, so in
practice `pbzip2 -d` here runs single-core regardless of `-p` — decompression throughput will be
capped at single-core bzip2 speed (~15-20MB/s observed), not 4x that. `-p$(nproc)` is harmless to
leave in (no downside), just don't expect it to multiply the extraction speed.

Confirmed result of the full 4-part extraction (2026-07-26): 3,545/3,545 wav files extracted,
matching all 3,545 xml transcripts exactly — QASR audio is now 100% complete, written to
`QASR/wav_all/` (220GB) on network storage. The older `QASR/wav_extracted/` (125GB, from the
961-then-2,021-file partial extractions) and `QASR/alt/` (59GB, the original 961-file extraction)
are now redundant subsets of `wav_all/` and can be deleted to reclaim ~184GB once you've verified
`wav_all/` independently.

Suggested run order for QASR:

1. Create `QASR/`.
2. Download `qasr_annotation_v1.0.tar.bz2`.
3. Download all 4 `qasr_wav_v1.0.tar.bz2.part_a{a,b,c,d}` files (plain `wget`, no `-c` — see above).
4. Extract directly via the `cat | pv | pbzip2 -dc | tar -xf -` pipeline above into a single output
   folder (e.g. `wav_all/`) — no need to materialize the concatenated `.tar.bz2` on disk first.

## 2. Layla

Layla was downloaded on a local machine first, then pushed to the remote VM into:

- `/home/MohammadNabulsi/whisper/Layla`

Suggested run order for Layla:

1. Download the ZIP on the local machine.
2. Transfer or push the ZIP to the remote VM under `Layla/`.
3. Unzip it in place so the extracted dataset lives under `Layla/Layla Witheeb Jordanian Arabic Acoustic Dataset/`.

Useful command on the VM:

```bash
cd /home/MohammadNabulsi/whisper/Layla
unzip "Layla Witheeb Jordanian Arabic Acoustic Dataset.zip"
```

## 3. Casablanca

Casablanca was downloaded using the notebook:

- `/home/MohammadNabulsi/whisper/downlaod_notebooks/casablanca_download_with_logging.ipynb`

After download, Casablanca was restructured with:

- `/home/MohammadNabulsi/whisper/downlaod_notebooks/restructure_casablanca.py`

The restructuring targets the dataset directory:

- `/home/MohammadNabulsi/whisper/casablanca`

Suggested run order for Casablanca:

1. Run `downlaod_notebooks/casablanca_download_with_logging.ipynb`.
2. Confirm the raw country folders are present under `casablanca/`.
3. Run `downlaod_notebooks/restructure_casablanca.py`.
4. Verify the grouped layout under `casablanca/far_dialects/`, `casablanca/levant/`, and `casablanca/relevant_arabic/`.

## 4. Omni

Omni was downloaded using the notebook:

- `/home/MohammadNabulsi/whisper/downlaod_notebooks/omni.ipynb`

Suggested run order for Omni:

1. Run `downlaod_notebooks/omni.ipynb`.
2. Confirm the output dataset is written under `/home/MohammadNabulsi/whisper/omnilingual_selected`.

## Current Notebook / Script Locations

- Casablanca download notebook: `/home/MohammadNabulsi/whisper/downlaod_notebooks/casablanca_download_with_logging.ipynb`
- Omni download notebook: `/home/MohammadNabulsi/whisper/downlaod_notebooks/omni.ipynb`
- Casablanca restructure script: `/home/MohammadNabulsi/whisper/downlaod_notebooks/restructure_casablanca.py`
