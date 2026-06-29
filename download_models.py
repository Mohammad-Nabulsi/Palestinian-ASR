#!/usr/bin/env python3
import os, sys, logging
from pathlib import Path
from tqdm.auto import tqdm
from huggingface_hub import snapshot_download

ROOT = Path.cwd()
MODELS_DIR = ROOT / "models"
LOGS_DIR = ROOT / ".logs"

MODELS = {
    "whisper_medium": "openai/whisper-medium",
    "whisper_large_v3": "openai/whisper-large-v3",
    "omni_asr_llm_300m": "facebook/omniASR-LLM-300M",
    "omni_asr_llm_1b": "facebook/omniASR-LLM-1B",
    "qwen3_asr_0_6b": "Qwen/Qwen3-ASR-0.6B",
    "qwen3_asr_1_7b": "Qwen/Qwen3-ASR-1.7B",
    # NVIDIA Arabic Conformer-CTC is usually loaded through NeMo/NGC.
    # This HF id may exist depending on access/version:
    "nvidia_conformer_ctc_large_arabic": "nvidia/stt_ar_conformer_ctc_large",
}

def setup_logger(name: str):
    LOGS_DIR.mkdir(exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    fh = logging.FileHandler(LOGS_DIR / f"{name}.log", mode="w")
    fh.setFormatter(logging.Formatter("%(asctime)s | %(levelname)s | %(message)s"))
    logger.addHandler(fh)
    return logger

def download_one(local_name: str, repo_id: str):
    logger = setup_logger(local_name)
    out_dir = MODELS_DIR / local_name
    out_dir.mkdir(parents=True, exist_ok=True)

    logger.info(f"START {repo_id} -> {out_dir}")
    print(f"\nDownloading {local_name}: {repo_id}")

    try:
        snapshot_download(
            repo_id=repo_id,
            local_dir=str(out_dir),
            local_dir_use_symlinks=False,
            resume_download=True,
        )
        logger.info("DONE")
        print(f"✅ Done: {local_name}")
    except Exception as e:
        logger.exception(f"FAILED: {e}")
        print(f"❌ Failed: {local_name}. See .logs/{local_name}.log")

def main():
    MODELS_DIR.mkdir(exist_ok=True)
    LOGS_DIR.mkdir(exist_ok=True)

    selected = sys.argv[1:]
    items = MODELS.items() if not selected else [(k, MODELS[k]) for k in selected]

    for name, repo in tqdm(list(items), desc="Models"):
        download_one(name, repo)

if __name__ == "__main__":
    main()