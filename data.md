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

If the archive has more split parts, download them with the same pattern, for example:

```bash
wget -c -O qasr_wav_v1.0.tar.bz2.part_ac "<same URL pattern, replacing part_ab with part_ac>"
wget -c -O qasr_wav_v1.0.tar.bz2.part_ad "<same URL pattern, replacing part_ab with part_ad>"
```

After all parts are downloaded, concatenate them in order:

```bash
cat qasr_wav_v1.0.tar.bz2.part_* > qasr_wav_v1.0.tar.bz2
```

Suggested run order for QASR:

1. Create `QASR/`.
2. Download `qasr_annotation_v1.0.tar.bz2`.
3. Download every `qasr_wav_v1.0.tar.bz2.part_*` file.
4. Concatenate the audio parts into `qasr_wav_v1.0.tar.bz2`.
5. Extract the archives if needed:

```bash
tar -xjf qasr_annotation_v1.0.tar.bz2
tar -xjf qasr_wav_v1.0.tar.bz2
```

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
