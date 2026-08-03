# Cell 1 — Environment
# No single pip line here on purpose -- see the kernel table above. Whichever venv this
# kernel points at already has that model family's stack installed; the other three
# families' packages are imported lazily inside their adapter's load_base(), so an
# unused adapter class being *defined* here never requires its packages to be present.

import os, json, gc, math, time, random, hashlib, shutil, warnings
from pathlib import Path
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Optional, Callable

import numpy as np, torch, torch.nn as nn
warnings.filterwarnings("ignore")
print(torch.__version__, torch.cuda.is_available())

# --- stdout/stderr tee: mirror every print() into a persistent log file so progress can
# be monitored from outside the notebook process without a Jupyter connection. ---
import sys, datetime
# Every artifact this notebook writes (logs, mlflow db, checkpoints, preds, metrics) is
# namespaced by PAL_RUN so the four runs of the pal study never overwrite each other --
# and, critically, so TrainAPI's resume-from-latest-checkpoint logic can never pick up a
# checkpoint belonging to a different run. Read from the env here (Cell 1 runs before the
# config cell) and re-read identically in Cell 2.
PAL_RUN = os.environ.get("PAL_RUN", "run1_pal_only")
# How many epochs the FINAL fine-tuning stage gets (stage 2 of runs 2-4; the single stage of
# run1_pal_only_1ep). >1 continues an already-trained run rather than starting over: TrainAPI
# resumes from that stage's last epoch checkpoint, restoring the adapter, the AdamW-8bit
# optimizer state, the scheduler and the RNG state, then trains the remaining epochs. Stage 1
# is unaffected and stays at 1 epoch -- on a resumed run it is skipped entirely (its epoch-1
# checkpoint already satisfies its budget), so the merged base is rebuilt without retraining.
PAL_STAGE2_EPOCHS = int(os.environ.get("PAL_STAGE2_EPOCHS", "1"))
# Stage 2 with the shared early-stopping default (<=50 epochs, patience 3 on palVal WER)
# instead of a fixed budget -- mirrors run1_pal_only's SINGLE_STAGE_EPOCHS=None behavior, but
# for stage 2 of a two-stage run. PAL_STAGE2_EPOCHS is ignored when this is set.
PAL_STAGE2_EARLYSTOP = os.environ.get("PAL_STAGE2_EARLYSTOP", "0") == "1"
# Checkpoints stay keyed on PAL_RUN alone (that is what makes the resume find them); only the
# RESULTS are namespaced by epoch budget, so each epoch's numbers are stored separately.
RUN_TAG = PAL_RUN
if PAL_STAGE2_EARLYSTOP:
    RUN_TAG = f"{PAL_RUN}_earlystop"
elif PAL_STAGE2_EPOCHS != 1:
    RUN_TAG = f"{PAL_RUN}_s2ep{PAL_STAGE2_EPOCHS}"
_LOG_DIR = Path(f"/root/Palestinian-ASR/Runs/whisper_medium_pal/{RUN_TAG}/logs")
_LOG_DIR.mkdir(parents=True, exist_ok=True)
_LOG_PATH = _LOG_DIR / f"train_{datetime.datetime.now():%Y%m%d_%H%M%S}.log"

class _Tee:
    def __init__(self, *streams): self.streams = streams
    def write(self, data):
        for s in self.streams:
            s.write(data); s.flush()
    def flush(self):
        for s in self.streams: s.flush()

_log_fh = open(_LOG_PATH, "a", buffering=1)
sys.stdout = _Tee(sys.__stdout__, _log_fh)
sys.stderr = _Tee(sys.__stderr__, _log_fh)
print(f"[log] mirroring stdout/stderr to {_LOG_PATH}")


# ================================================================
# THE ONLY TWO THINGS YOU CHANGE
# ================================================================
MODEL_NAME  = "openai/whisper-medium"
# The `pal` view (see scripts/build_pal_stage_datasets.py), cut from
# data_curated_levant_binary_v1 and materialized to LOCAL disk (not the /workspace network
# mount) so training I/O is fast:
#   train = casa/pal {train + val}          664 rows  (~0.99h)
#   val   = 30% of casa/pal test            199 rows   -> palVal
#   test  = remaining 70% of casa/pal test  465 rows   -> palTest
DATASET_DIR = "/root/Palestinian-ASR/data_pal_v1"

# ---- pal study: which of the four runs is this? (PAL_RUN env var) ----
#   run1_pal_only : LoRA fine-tune straight on palTrain, eval palVal + palTest.
#   run1_pal_only_1ep : identical, but a FIXED 1-epoch budget instead of the shared
#                   <=50-epochs-with-early-stopping default. run1_pal_only early-stopped at
#                   epoch 7 (best at 4), which makes its WER incomparable to the 1-epoch
#                   stage 2 of runs 2-4 -- this variant is the budget-matched baseline.
#   run2_jor      : stage 1 = 1 epoch LoRA on ALL of Jordanian Casablanca (train+val+test),
#                   merge the adapter into the base weights, eval; stage 2 = 1 epoch LoRA on
#                   palTrain on top of the merged model, eval again.
#   run3_omni     : same two-stage shape, stage 1 = ALL of omni (train+val+test).
#   run4_omni_jor : same two-stage shape, stage 1 = omni + Jordanian Casablanca together.
# Stage-1 corpora are single-split ({dir}/train/) views built by the same script.
STAGE1_BY_RUN = {
    "run1_pal_only": None,
    "run1_pal_only_1ep": None,
    "run2_jor":      "/root/Palestinian-ASR/data_stage1_v1/jor",
    "run3_omni":     "/root/Palestinian-ASR/data_stage1_v1/omni",
    "run4_omni_jor": "/root/Palestinian-ASR/data_stage1_v1/omni_jor",
    # New variant: reuse the already-trained epoch-1-of-2 stage-1 checkpoint from the
    # matching *_s1ep2 run (see PAL_STAGE1_CKPT below) instead of training stage 1 fresh.
    # Same stage-1 corpus as the parent run, purely for accurate dataset_hours/STAGE1_DIR
    # bookkeeping in the summary -- the corpus itself is not retrained on.
    "run2_jor_s1e1of2":      "/root/Palestinian-ASR/data_stage1_v1/jor",
    "run3_omni_s1e1of2":     "/root/Palestinian-ASR/data_stage1_v1/omni",
    "run4_omni_jor_s1e1of2": "/root/Palestinian-ASR/data_stage1_v1/omni_jor",
    # layla+jor study (no omni): stage 1 = ALL of layla + ALL of Jordanian Casablanca
    # (train+val+test), built by scripts/build_stage1_layla_jor.py.
    #   run5_layla_jor           : stage 1 = 1 epoch, merge, stage 2 = early-stopping default
    #                               (PAL_STAGE2_EARLYSTOP=1).
    #   run5_layla_jor_s1ep2     : stage 1 = 2 epochs, merge the LAST epoch, stage 2 = 1 fixed
    #                               epoch (PAL_STAGE1_EPOCHS=2).
    #   run5_layla_jor_s1ep2_s2ep2 : same 2-epoch stage-1 merge, reused via PAL_STAGE1_CKPT
    #                               from run5_layla_jor_s1ep2's saved stage-1 checkpoint
    #                               (no retraining), stage 2 = 2 fixed epochs.
    "run5_layla_jor":            "/root/Palestinian-ASR/data_stage1_v1/layla_jor",
    "run5_layla_jor_s1ep2":      "/root/Palestinian-ASR/data_stage1_v1/layla_jor",
    "run5_layla_jor_s1ep2_s2ep2": "/root/Palestinian-ASR/data_stage1_v1/layla_jor",
}
if PAL_RUN not in STAGE1_BY_RUN:
    raise SystemExit(f"PAL_RUN={PAL_RUN!r} not one of {list(STAGE1_BY_RUN)}")
STAGE1_DIR = STAGE1_BY_RUN[PAL_RUN]
# When set, stage 1 is NOT trained -- its adapter is loaded directly from this checkpoint dir
# (typically another run's saved epochNNN/ dir) and merged. Used to ask "what if we'd merged
# after epoch 1 of a 2-epoch stage-1 schedule, instead of epoch 2?" using the epoch001/
# checkpoint already saved by the corresponding *_s1ep2 run, without retraining stage 1.
PAL_STAGE1_CKPT = os.environ.get("PAL_STAGE1_CKPT")
# Stage-1 epoch budget. Historically 1 for runs 2-4 (no early stopping -- with a single epoch
# there is nothing to stop early or select between); the layla_jor study's *_s1ep2 runs set
# this to 2 via PAL_STAGE1_EPOCHS to train a full 2-epoch stage 1 before merging. run1 uses
# the shared TrainConfigSpec defaults (up to 50 epochs, early stopping on val WER with
# patience 3) like every other model family's run in this project.
STAGE_EPOCHS = int(os.environ.get("PAL_STAGE1_EPOCHS", "1"))
# Single-stage runs only: None -> the shared TrainConfigSpec default (<=50 epochs, early
# stopping on palVal WER with patience 3); an int -> exactly that many epochs, no early
# stopping. Ignored by the two-stage runs, which are always STAGE_EPOCHS per stage.
SINGLE_STAGE_EPOCHS = {"run1_pal_only": None,
                       "run1_pal_only_1ep": PAL_STAGE2_EPOCHS}.get(PAL_RUN)
# Stage 2 of the two-stage runs. Stage 1 is always STAGE_EPOCHS (1).
STAGE2_EPOCHS = PAL_STAGE2_EPOCHS
# ================================================================

KERNEL_BY_MODEL = {
    "omnilingual-asr/omniASR_LLM_1B":                    "omni_gpu (/workspace/venv_omni_gpu)",
    "omnilingual-asr/omniASR_LLM_300M":                  "omni_gpu (/workspace/venv_omni_gpu)",
    "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0": "nemo_gpu (/workspace/venv_nemo_gpu)",
    "nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0":  "nemo_gpu (/workspace/venv_nemo_gpu)",
    "Qwen/Qwen3-ASR-0.6B-hf":                            "qwen_gpu (/workspace/venv_qwen_gpu)",
    "CohereLabs/cohere-transcribe-arabic-07-2026":       "qwen_gpu (/workspace/venv_qwen_gpu)",
    "openai/whisper-large-v3":                            "qwen_gpu (/workspace/venv_qwen_gpu)",
    "openai/whisper-medium":                             "qwen_gpu (/workspace/venv_qwen_gpu)",
}
# Each adapter's own native language-conditioning code (OmniASR script codes vs. plain ISO
# codes elsewhere) -- kept out of the two knobs above since it's determined by MODEL_NAME,
# not something you'd want to pick independently.
LANG_BY_MODEL = {
    "omnilingual-asr/omniASR_LLM_1B":                    "arb_Arab",
    "omnilingual-asr/omniASR_LLM_300M":                  "arb_Arab",
    "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0": "ar",
    "nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0":  "ar",
    "Qwen/Qwen3-ASR-0.6B-hf":                            "ar",
    "CohereLabs/cohere-transcribe-arabic-07-2026":       "ar",
    "openai/whisper-large-v3":                            "ar",
    "openai/whisper-medium":                             "ar",
}
LANG = LANG_BY_MODEL[MODEL_NAME]

SMOKE_TEST    = False                # tiny subsets + 2 epochs
SEED          = 42

DEVICE        = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32

ROOT          = Path(os.environ.get("ASR_ENV_ROOT", "/workspace/asr_env"))
MODEL_CACHE   = ROOT / "models"
PRED_DIR      = ROOT / "preds"
METRIC_DIR    = ROOT / "metrics"
CKPT_DIR      = ROOT / "checkpoints" / f"whisper_medium_pal__{PAL_RUN}"
for d in (MODEL_CACHE, PRED_DIR, METRIC_DIR, CKPT_DIR): d.mkdir(parents=True, exist_ok=True)

os.environ["HF_HOME"] = str(MODEL_CACHE / "hf")
import mlflow
# mlflow 3.x deprecated the plain "file:" tracking backend (raises unless
# MLFLOW_ALLOW_FILE_STORE=true) -- sqlite is the currently-recommended local backend and
# also lets `mlflow ui --backend-store-uri ...` browse runs later. Artifact location is
# set explicitly on first creation since sqlite backends don't default one on their own.
# Local disk, NOT ROOT (/workspace, a FUSE-mounted network volume) -- SQLite needs real
# flock()/fcntl() locking semantics, which FUSE network filesystems commonly emulate
# poorly or not at all. mlflow.start_run() hard-hung indefinitely (0% GPU, one CPU-pegged
# thread busy-retrying the lock, confirmed via the mlflow.db file's mtime never advancing)
# against /workspace/asr_env/mlflow.db before this fix.
_LOCAL_MLFLOW_DIR = Path(f"/root/Palestinian-ASR/Runs/whisper_medium_pal/{RUN_TAG}")
_LOCAL_MLFLOW_DIR.mkdir(parents=True, exist_ok=True)
MLFLOW_TRACKING_URI = os.environ.get("MLFLOW_TRACKING_URI", f"sqlite:///{_LOCAL_MLFLOW_DIR / 'mlflow.db'}")
mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
_MLFLOW_EXPERIMENT = "arabic-asr-unified"
if mlflow.get_experiment_by_name(_MLFLOW_EXPERIMENT) is None:
    mlflow.create_experiment(_MLFLOW_EXPERIMENT, artifact_location=f"file:{ROOT / 'mlruns' / _MLFLOW_EXPERIMENT}")
mlflow.set_experiment(_MLFLOW_EXPERIMENT)

if MODEL_NAME.startswith("CohereLabs/"):
    # The repo is GATED: verify credentials up front so the failure mode is obvious.
    from huggingface_hub import get_token
    _tok = get_token() or os.environ.get("HF_TOKEN")
    if not _tok:
        print("[WARN] No Hugging Face token found (hf auth login / HF_TOKEN). "
              f"{MODEL_NAME} is a gated repo -- the model download will fail with 401 "
              "until a token whose account accepted the license is available.")

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed()
print(f"MODEL_NAME={MODEL_NAME}")
print(f"kernel should be: {KERNEL_BY_MODEL.get(MODEL_NAME, '?? not in KERNEL_BY_MODEL')}")
print(f"DEVICE={DEVICE} | dtype={COMPUTE_DTYPE} | ROOT={ROOT} | DATASET_DIR={DATASET_DIR}")
print(f"PAL_RUN={PAL_RUN} | STAGE1_DIR={STAGE1_DIR} | STAGE_EPOCHS={STAGE_EPOCHS} | CKPT_DIR={CKPT_DIR}")
print(f"SINGLE_STAGE_EPOCHS={SINGLE_STAGE_EPOCHS} (None = early-stopping default) | "
      f"STAGE2_EPOCHS={STAGE2_EPOCHS} | PAL_STAGE2_EARLYSTOP={PAL_STAGE2_EARLYSTOP} | "
      f"PAL_STAGE1_CKPT={PAL_STAGE1_CKPT} | RUN_TAG={RUN_TAG}")

import re, unicodedata, jiwer

_DIAC = re.compile(r"[\u0617-\u061A\u064B-\u0652\u0640]")
_PUNC = re.compile(r"[^\w\s\u0621-\u064A]")

def normalize_ar(t: str) -> str:
    """Diacritic strip, tatweel removal, alef/ya/ta-marbuta unification."""
    if t is None: return ""
    t = unicodedata.normalize("NFKC", str(t))
    t = _DIAC.sub("", t)
    t = re.sub("[\u0622\u0623\u0625\u0671]", "\u0627", t)   # alef variants -> alef
    t = t.replace("\u0649", "\u064A")                          # alef maqsura -> ya
    t = t.replace("\u0629", "\u0647")                          # ta marbuta -> ha
    t = t.replace("\u0624", "\u0648").replace("\u0626", "\u064A")
    t = _PUNC.sub(" ", t)
    return re.sub(r"\s+", " ", t).strip()

def compute_wer_cer(preds, refs, normalize=True):
    if normalize:
        preds = [normalize_ar(p) for p in preds]
        refs  = [normalize_ar(r) for r in refs]
    keep = [(p, r) for p, r in zip(preds, refs) if r.strip()]
    if not keep: return {"wer": float("nan"), "cer": float("nan"), "n": 0}
    p, r = zip(*keep)
    return {"wer": jiwer.wer(list(r), list(p)),
            "cer": jiwer.cer(list(r), list(p)),
            "n": len(r)}

@dataclass
class LoRAConfigSpec:
    r: int = 32
    lora_alpha: int = 32
    lora_dropout: float = 0.05
    bias: str = "none"
    target_modules: Optional[List[str]] = None
    modules_to_save: Optional[List[str]] = None
    task_type: Optional[str] = None

@dataclass
class TrainConfigSpec:
    num_epochs: int = 50
    early_stopping_patience: int = 3
    metric_for_best: str = "wer"
    greater_is_better: bool = False
    per_device_train_batch_size: int = 4
    per_device_eval_batch_size: int = 4
    gradient_accumulation_steps: int = 4
    learning_rate: float = 1e-4
    warmup_ratio: float = 0.05
    weight_decay: float = 0.0
    max_grad_norm: float = 1.0
    optim: str = "adamw_bnb_8bit"
    bf16: bool = True
    gradient_checkpointing: bool = False
    dataloader_num_workers: int = 0
    max_audio_seconds: float = 30.0
    max_label_tokens: int = 256
    save_total_limit: int = 3
    save_steps: Optional[int] = None      # None -> auto: max(200, steps_per_epoch // 3)
    dataloader_pin_memory: bool = True
    dataloader_persistent_workers: bool = True
    dataloader_prefetch_factor: int = 4

class ConfigAPI:
    """Single source of truth for per-model hyperparameters, across all four families.

    NOTE on target_modules per family (real API differences, not guesses -- see each
    adapter class docstring below for the verification evidence):
      * OmniASR (fairseq2, not transformers): StandardMultiheadAttention
        (q_proj/k_proj/v_proj/output_proj) + GLUFeedForwardNetwork (gate_proj/inner_proj).
      * FastConformer-CTC (NeMo): linear_q/linear_k/linear_v/linear_out (attention) +
        linear1/linear2 (feed-forward).
      * Qwen3-ASR (transformers-native): standard q/k/v/o_proj + gate/up/down_proj.
      * CohereAsr (transformers-native, gated): target_modules=None -> discovered at
        runtime in CohereAsrAdapter.apply_lora() via named_modules() introspection.
    """
    _LORA = {
        "omnilingual-asr/omniASR_LLM_1B": LoRAConfigSpec(
            r=32, lora_alpha=32,
            target_modules=["q_proj","k_proj","v_proj","output_proj","gate_proj","inner_proj"]),
        "omnilingual-asr/omniASR_LLM_300M": LoRAConfigSpec(
            r=32, lora_alpha=32,
            target_modules=["q_proj","k_proj","v_proj","output_proj","gate_proj","inner_proj"]),
        "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0": LoRAConfigSpec(
            target_modules=["linear_q","linear_k","linear_v","linear_out","linear1","linear2"]),
        "nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0": LoRAConfigSpec(
            target_modules=["linear_q","linear_k","linear_v","linear_out","linear1","linear2"]),
        "Qwen/Qwen3-ASR-0.6B-hf": LoRAConfigSpec(
            # Capped to r=32/alpha=32 (the same blanket default every other model family in
            # this file uses) by explicit instruction, to keep LoRA capacity uniform across
            # every model in the comparison. Published Qwen3-ASR-specific guidance
            # (arXiv:2607.08208, "Diarization-Guided Qwen-ASR Adaptation") suggests r=64/
            # alpha=128 performs better for this model family specifically -- noted here for
            # the record in case a later run wants to isolate that variable, but not what
            # this run uses. learning_rate=5e-5 (below) is kept from that same reference.
            r=32, lora_alpha=32,
            target_modules=["q_proj","k_proj","v_proj","o_proj","gate_proj","up_proj","down_proj"]),"CohereLabs/cohere-transcribe-arabic-07-2026": LoRAConfigSpec(target_modules=None),
        "openai/whisper-large-v3": LoRAConfigSpec(
            # r=32/alpha=32/lr=1e-4 (see TrainConfigSpec entry below) is this project's own
            # prior empirical "Whisper study" (per the Cell 4 markdown), not a borrowed
            # default -- kept as-is. Brief 2026-08-01 literature check found published Whisper
            # LoRA configs cluster around r=32 with alpha/r in [1.0, 2.0] (i.e. alpha in
            # [32, 64]) and lr around 1e-4, so this project's r=32/alpha=32/lr=1e-4 sits
            # squarely inside that published range -- no change made.
            target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"]),
        "openai/whisper-medium": LoRAConfigSpec(
            # Same encoder/decoder attention + MLP naming as whisper-large-v3 (both are the
            # stock HF WhisperForConditionalGeneration architecture, just fewer layers/dims
            # for medium) -- identical target_modules.
            # r=16 per explicit instruction for the pal study and every run after it (was 32,
            # the blanket default the other families still use). lora_alpha is deliberately
            # left at 32 -- "same configs, rank 16" -- which makes scaling alpha/r = 2.0,
            # still inside the published Whisper-LoRA range of alpha/r in [1.0, 2.0] noted on
            # the whisper-large-v3 entry above. Halving the rank halves the trainable-parameter
            # count, which is the point on a train split this small (palTrain is ~1h).
            r=16,
            target_modules=["q_proj","k_proj","v_proj","out_proj","fc1","fc2"]),
    }
    _TRAIN = {
        "omnilingual-asr/omniASR_LLM_1B": TrainConfigSpec(
            per_device_train_batch_size=2, gradient_accumulation_steps=8,
            gradient_checkpointing=True, dataloader_num_workers=4,
            # 40s = OmniASR's HARD inference ceiling (see the dedicated notebook's Cell 4
            # comment for the full verified-on-GPU rationale).
            max_audio_seconds=40.0),
        "omnilingual-asr/omniASR_LLM_300M": TrainConfigSpec(
            # Same pattern validated live for Qwen: same effective batch (16), less
            # gradient accumulation, more real per-step parallelism -- batch=4/accum=4 ->
            # batch=8/accum=2. Scaled conservatively (2x real batch, matching Qwen's own 2x
            # step) since this model is actually ~1.63B params (~2.7x Qwen's 0.6B, verified
            # via a real CPU run -- see OmniASRAdapter's docstring) despite the "300M" name,
            # and already carries gradient_checkpointing=True + a raised max_audio_seconds=
            # 40.0, both signs of past memory sensitivity. No live GPU to confirm (Qwen
            # currently owns the only GPU) -- re-tune upward empirically on first launch.
            per_device_train_batch_size=8, gradient_accumulation_steps=2,

            gradient_checkpointing=True, dataloader_num_workers=4,
            max_audio_seconds=40.0),
        "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0": TrainConfigSpec(
            per_device_train_batch_size=2, gradient_accumulation_steps=4),
        "nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0": TrainConfigSpec(
            per_device_train_batch_size=2, gradient_accumulation_steps=4),
        "Qwen/Qwen3-ASR-0.6B-hf": TrainConfigSpec(
            # 0.6B model + LoRA is light; batch 8 x accum 2 (effective 16) uses the 24GB
            # card far better than batch 2 x accum 4 while leaving headroom for a possible
            # concurrent second run. learning_rate=5e-5 per arXiv:2607.08208 (see LoRA note
            # above) rather than the shared 1e-4 default.
            # Same effective batch (16) as before, just less gradient accumulation and
            # more real per-step parallelism -- batch=8/accum=2 used 18.57GB during full
            # training; empirically confirmed headroom to push further (see eval note below).
            per_device_train_batch_size=16, gradient_accumulation_steps=1,
            learning_rate=5e-5, dataloader_num_workers=4,
            # Validation runs generate() (autoregressive, sequential decode steps) on top of
            # a loss forward pass -- GPU sat at only ~44-51% util during eval even at batch=16
            # (14.1GB/24GB used, ~10GB free), so pushed further. batch=4 (shared default) made
            # epoch-1 validation take 3+ hours vs ~1h for the entire training portion of the
            # same epoch; batch=16 alone got a real, sustained 2.67 rows/s (~88min projected).
            per_device_eval_batch_size=32),
        "CohereLabs/cohere-transcribe-arabic-07-2026": TrainConfigSpec(
            per_device_train_batch_size=1, gradient_accumulation_steps=8),   # 2B model
        "openai/whisper-large-v3": TrainConfigSpec(
            # Same pattern validated live for Qwen: same effective batch (8), less gradient
            # accumulation, more real per-step parallelism -- batch=2/accum=4 -> batch=4/
            # accum=2. Scaled conservatively (2x real batch, matching Qwen's own 2x step,
            # not Qwen's full magnitude) since whisper-large-v3 is ~2.5x Qwen's param count
            # and there's no live GPU to confirm headroom (Qwen currently owns the only GPU).
            # Re-tune upward empirically on first real launch, watching nvidia-smi.
            per_device_train_batch_size=4, gradient_accumulation_steps=2,   # 1.5B model

            # Same "generate() is autoregressive/sequential-decode-bound, not batch-bound"
            # argument that let Qwen (0.6B) push eval batch to 8x the shared default of 4
            # applies here too -- but whisper-large-v3 is ~2.5x Qwen's param count (~1.5B vs
            # 0.6B) and its decoder cross-attends over a FIXED 1500-frame encoder output (the
            # feature extractor always pads/truncates log-mel to 30s -> 128x3000), so per-
            # sample eval memory is at least predictable (no risk of a stray long clip
            # spiking a batch), but still much heavier per-sample than Qwen's raw-waveform
            # frontend. No live GPU to confirm headroom the way Qwen's 32 was confirmed
            # (~18GB/24GB peak) -- conservatively 2x the shared default of 4 rather than
            # matching Qwen's 8x; a human should re-tune upward empirically once this
            # actually launches, watching nvidia-smi the same way Qwen's was tuned.
            per_device_eval_batch_size=8),
        "openai/whisper-medium": TrainConfigSpec(
            # whisper-medium is ~769M params -- well under half of whisper-large-v3 (~1.5B)
            # and closer to Qwen3-ASR-0.6B (~0.6B) in size, so this follows Qwen's actual
            # confirmed-on-GPU pattern directly: batch=8/accum=2 (effective batch 16),
            # dataloader_num_workers=4. Per explicit instruction, NOT the conservative
            # batch=4/accum=2 used for whisper-large-v3 above.
            # batch=4/accum=4 rather than batch=8/accum=2. SAME effective batch (16), so the
            # optimization schedule is unchanged -- only the activation memory of a single
            # forward/backward is halved. batch=8 measured 19.82GB of a 23.52GB card and OOM'd
            # in the first epoch's forward pass because a concurrent FastConformer run holds
            # 3.65GB on the same GPU (2026-08-01 run1 log); activations dominate that figure
            # (bf16 weights are ~1.5GB and the r=16 LoRA optimizer state is ~35MB), so halving
            # the real batch brings the peak to roughly 11GB and leaves real headroom.
            per_device_train_batch_size=4, gradient_accumulation_steps=4,
            dataloader_num_workers=4,
            # Per explicit instruction: eval batch fixed at 16 (between whisper-large-v3's
            # conservative 8 and Qwen's 32 -- medium's decoder still cross-attends over the
            # same fixed 1500-frame encoder output as large-v3, so kept below Qwen's raw-
            # waveform-frontend headroom rather than matching it outright). Lowered 16 -> 8
            # for the same reason as the train batch above: eval batch 16 does survive
            # (baseline prediction over palTest completed at 16 before the training OOM), but a
            # per-epoch validation that OOMs costs the whole epoch, and eval batch size affects
            # nothing but throughput.
            per_device_eval_batch_size=8),
    }
    @classmethod
    def lora(cls, name)  -> LoRAConfigSpec:  return cls._LORA.get(name, LoRAConfigSpec())
    @classmethod
    def train(cls, name) -> TrainConfigSpec: return cls._TRAIN.get(name, TrainConfigSpec())

print(json.dumps(asdict(ConfigAPI.lora(MODEL_NAME)), indent=2))
print(json.dumps(asdict(ConfigAPI.train(MODEL_NAME)), indent=2))

from abc import ABC, abstractmethod

class ModelAdapter(ABC):
    name: str
    loss_type: str = "seq2seq"          # or "ctc"
    supports_unsloth: bool = False

    def __init__(self, model_name: str, lang: str = LANG):
        self.model_name = model_name; self.lang = lang
        self.model = None; self.processor = None

    @abstractmethod
    def load_base(self): ...
    @abstractmethod
    def preprocess(self, example: Dict) -> Dict: ...
    @abstractmethod
    def collate(self, features: List[Dict]) -> Dict[str, Any]: ...
    @abstractmethod
    def generate(self, batch: Dict) -> List[str]: ...

    def apply_lora(self, spec: "LoRAConfigSpec"):
        """Default: standard PEFT LoRA wrapping self.model in place -- correct whenever the
        target projections are real torch.nn.Linear (Qwen3-ASR's case). Adapters whose LoRA
        can't go through this path override it: OmniASRAdapter (manual injection -- fairseq2
        Linear isn't a torch.nn.Linear subclass), ConformerCTCAdapter (keeps the PEFT wrapper
        in self.peft, not self.model, so the raw NeMo API stays reachable on self.model),
        CohereAsrAdapter (target_modules is discovered at runtime first)."""
        if self.supports_unsloth:
            try:
                from unsloth import FastModel
                self.model = FastModel.get_peft_model(
                    self.model, r=spec.r, lora_alpha=spec.lora_alpha,
                    lora_dropout=spec.lora_dropout, bias=spec.bias,
                    target_modules=spec.target_modules, use_gradient_checkpointing="unsloth")
                print("[lora] unsloth"); return self.model
            except Exception as e:
                print(f"[lora] unsloth unavailable ({e}); falling back to PEFT")
        from peft import LoraConfig, get_peft_model
        kw = dict(r=spec.r, lora_alpha=spec.lora_alpha, lora_dropout=spec.lora_dropout,
                  bias=spec.bias, target_modules=spec.target_modules)
        if spec.modules_to_save: kw["modules_to_save"] = spec.modules_to_save
        if spec.task_type:       kw["task_type"] = spec.task_type
        self.model = get_peft_model(self.model, LoraConfig(**kw))
        self.model.print_trainable_parameters()
        print("[lora] peft"); return self.model

    def train_step(self, batch) -> torch.Tensor:
        out = self.model(**batch)
        return out.loss if hasattr(out, "loss") else out["loss"]

    def trainable_parameters(self) -> List["torch.nn.Parameter"]:
        return [p for p in self.model.parameters() if p.requires_grad]

    def save_checkpoint(self, dir_):
        """Default: self.model IS the PEFT-wrapped module. Concrete adapters whose LoRA
        lives elsewhere (self.peft, manual injection) override this."""
        Path(dir_).mkdir(parents=True, exist_ok=True)
        self.model.save_pretrained(str(dir_))

    def load_checkpoint(self, dir_):
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        sd = load_file(str(Path(dir_) / "adapter_model.safetensors"))
        set_peft_model_state_dict(self.model, sd)

import math

class _LoRALinear(nn.Module):
    """Manual LoRA wrapper. OmniASR's projections are `fairseq2.nn.projection.Linear`, which is
    NOT a `torch.nn.Linear` subclass, so neither PEFT nor Unsloth can wrap them. We inject a
    low-rank side path ourselves: y = base(x) + scaling * dropout(x) @ A^T @ B^T, B zero-init so
    the initial delta is 0. Works on any module exposing a 2-D `.weight`."""
    def __init__(self, base: nn.Module, r: int, alpha: int, dropout: float):
        super().__init__()
        self.base = base
        for p in self.base.parameters(): p.requires_grad_(False)
        out_f, in_f = base.weight.shape
        dt, dev = base.weight.dtype, base.weight.device
        self.lora_A = nn.Parameter(torch.zeros(r, in_f, dtype=dt, device=dev))
        self.lora_B = nn.Parameter(torch.zeros(out_f, r, dtype=dt, device=dev))
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        self.scaling = alpha / r
        self.drop = nn.Dropout(dropout)
    def forward(self, x):
        out = self.base(x)
        delta = self.drop(x) @ self.lora_A.t() @ self.lora_B.t()
        return out + self.scaling * delta


class OmniASRAdapter(ModelAdapter):
    """fairseq2 wav2vec2_llama. Verified against omnilingual-asr@main source AND a real CPU run
    of omniASR_LLM_300M (1.63B params). See DISCOVERY.md / SMOKE_RESULTS.md.

      * pipeline.model -> Wav2Vec2LlamaModel;  pipeline.tokenizer -> Tokenizer
      * tokenizer.create_encoder() takes NO lang; decode via create_decoder(skip_special_tokens=True)
      * TRAINING: loss = model(Seq2SeqBatch(...)); model builds `audio [lang] <bos> text <eos>` and
        masks the loss internally. Pad text with pad_idx (NOT -100). seq_lens must be list[int].
      * INFERENCE: pipeline.transcribe(list[dict{waveform,sample_rate}], lang=[...], batch_size=n)
      * LoRA: projections are fairseq2.nn.projection.Linear (NOT torch.nn.Linear) -> PEFT/Unsloth
        can't wrap them, so we inject LoRA manually via _LoRALinear.
    """
    loss_type = "seq2seq"
    supports_unsloth = False

    CARD = {"omnilingual-asr/omniASR_LLM_1B":   "omniASR_LLM_1B",
            "omnilingual-asr/omniASR_LLM_300M": "omniASR_LLM_300M"}

    def __init__(self, model_name, lang=LANG):
        super().__init__(model_name, lang)
        self.card = self.CARD[model_name]
        self.pipeline = None; self.tokenizer = None
        self.pad_idx = 0
        self._encoder = None; self._decoder = None
        self.derived_target_modules = None

    def load_base(self):
        local = MODEL_CACHE / self.card
        from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
        if local.exists() and any(local.iterdir()):
            print(f"[load] local -> {local}")
        else:
            print(f"[load] downloading {self.card} -> {local}")
            local.mkdir(parents=True, exist_ok=True)
        # FAIRSEQ2_CACHE_DIR controls the checkpoint cache (verified on a real 300M load).
        os.environ.setdefault("FAIRSEQ2_CACHE_DIR", str(local))

        self.pipeline  = ASRInferencePipeline(self.card, device=DEVICE)
        self.model     = self.pipeline.model
        self.tokenizer = self.pipeline.tokenizer
        self._encoder  = self.tokenizer.create_encoder()
        self._decoder  = self.tokenizer.create_decoder(skip_special_tokens=True)
        self.pad_idx   = getattr(self.tokenizer.vocab_info, "pad_idx", 0) or 0

        from omnilingual_asr.models.wav2vec2_llama.lang_ids import supported_langs
        assert self.lang in supported_langs, f"{self.lang} not in supported_langs"

        self.derived_target_modules = self._derive_target_modules()
        print(f"[load] pad_idx={self.pad_idx} | derived target_modules={self.derived_target_modules}")
        return self.model

    def _derive_target_modules(self):
        """Detect projection leaf names by DUCK TYPING (2-D `.weight`), because fairseq2's Linear
        is not a torch.nn.Linear subclass and isinstance(mod, nn.Linear) would miss all of them."""
        want = {"q_proj","k_proj","v_proj","output_proj","gate_proj","inner_proj"}
        found = set()
        for name, mod in self.model.named_modules():
            leaf = name.split(".")[-1]
            w = getattr(mod, "weight", None)
            if leaf in want and w is not None and getattr(w, "ndim", 0) == 2:
                found.add(leaf)
        return sorted(found) or sorted(want)

    def apply_lora(self, spec: LoRAConfigSpec):
        """Manual LoRA injection (PEFT/Unsloth can't wrap fairseq2.nn.projection.Linear)."""
        targets = set(self.derived_target_modules or spec.target_modules)
        replaced = 0
        for mod_name, mod in list(self.model.named_modules()):
            leaf = mod_name.split(".")[-1]
            w = getattr(mod, "weight", None)
            if (leaf in targets and w is not None and getattr(w, "ndim", 0) == 2
                    and not isinstance(mod, _LoRALinear)):
                parent = self.model.get_submodule(mod_name.rsplit(".", 1)[0]) if "." in mod_name else self.model
                setattr(parent, leaf, _LoRALinear(mod, spec.r, spec.lora_alpha, spec.lora_dropout))
                replaced += 1
        for n, p in self.model.named_parameters():
            p.requires_grad_("lora_" in n)
        n_tr = sum(p.numel() for p in self.model.parameters() if p.requires_grad)
        print(f"[lora] manual injection into {replaced} fairseq2 Linear layers | trainable params={n_tr}")
        return self.model

    def preprocess(self, ex):
        audio = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
        if sr != 16000:
            import librosa; audio = librosa.resample(np.asarray(audio, dtype=np.float32),
                                                     orig_sr=sr, target_sr=16000)
        text = normalize_ar(ex["text"])
        ids  = self._encode(text)
        return {"input_values": np.asarray(audio, dtype=np.float32),
                "labels": ids, "text": text,
                "audio_len": len(audio) / 16000.0}

    def _encode(self, text):
        if self._encoder is None: return []
        return self._encoder(text).tolist()

    def _decode(self, ids):
        if self._decoder is None: return ""
        import torch as _t
        return str(self._decoder(_t.as_tensor(ids, dtype=_t.int64)))

    def collate(self, feats):
        maxa = max(len(f["input_values"]) for f in feats)
        maxl = max(len(f["labels"]) for f in feats) or 1
        wav  = torch.zeros(len(feats), maxa, dtype=torch.float32)
        mask = torch.zeros(len(feats), maxa, dtype=torch.long)
        lab  = torch.full((len(feats), maxl), self.pad_idx, dtype=torch.long)
        lab_lens = torch.zeros(len(feats), dtype=torch.long)
        for i, f in enumerate(feats):
            a = torch.as_tensor(f["input_values"], dtype=torch.float32)
            wav[i, :len(a)] = a; mask[i, :len(a)] = 1
            if len(f["labels"]):
                lab[i, :len(f["labels"])] = torch.as_tensor(f["labels"], dtype=torch.long)
                lab_lens[i] = len(f["labels"])
        return {"input_values": wav, "attention_mask": mask, "labels": lab,
                "label_lengths": lab_lens,
                "lang": [self.lang]*len(feats), "text": [f["text"] for f in feats]}

    def train_step(self, batch) -> torch.Tensor:
        from fairseq2.datasets.batch import Seq2SeqBatch
        wav  = batch["input_values"]; mask = batch["attention_mask"]; lab = batch["labels"]
        # fairseq2 Seq2SeqBatch requires seq_lens as list[int], NOT tensors.
        src_lens = mask.sum(dim=1).to(torch.long).tolist()
        if "label_lengths" in batch:
            tgt_lens = batch["label_lengths"].to(torch.long).tolist()
        else:
            tgt_lens = (lab != self.pad_idx).sum(dim=1).to(torch.long).tolist()
        model_dtype = next(self.model.parameters()).dtype
        dev = getattr(self.model, "device", DEVICE)
        langs = batch.get("lang", [self.lang]*wav.shape[0])
        seq2seq = Seq2SeqBatch(
            source_seqs     = wav.to(dev, model_dtype),
            source_seq_lens = src_lens,
            target_seqs     = lab.to(dev).to(torch.long),
            target_seq_lens = tgt_lens,
            example         = {"lang": list(langs)},
        )
        out = self.model(seq2seq)
        return out if torch.is_tensor(out) else out[0]

    @torch.no_grad()
    def generate(self, batch):
        wav  = batch["input_values"]; mask = batch["attention_mask"]
        inp = []
        for i in range(wav.shape[0]):
            n = int(mask[i].sum().item())
            inp.append({"waveform": wav[i, :n].float().cpu().numpy(), "sample_rate": 16000})
        langs = batch.get("lang", [self.lang]*len(inp))
        out = self.pipeline.transcribe(inp, lang=list(langs), batch_size=len(inp))
        return [str(o) for o in out]

    def save_checkpoint(self, dir_):
        """Manual LoRA (no PEFT wrapper for fairseq2 Linear) -> save just the LoRA tensors."""
        dir_ = Path(dir_); dir_.mkdir(parents=True, exist_ok=True)
        sd = {k: v for k, v in self.model.state_dict().items() if "lora_" in k}
        torch.save(sd, dir_ / "adapter.pt")

    def load_checkpoint(self, dir_):
        sd = torch.load(Path(dir_) / "adapter.pt", map_location=DEVICE)
        self.model.load_state_dict(sd, strict=False)


class ConformerCTCAdapter(ModelAdapter):
    '''nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0 in pure-CTC mode.
       Verified on GPU (2026-07-18):
         * load:  ASRModel.from_pretrained -> EncDecHybridRNNTCTCBPEModel (114.6M),
                  change_decoding_strategy(decoder_type="ctc")
         * TRAIN: enc,enc_len = model(input_signal, input_signal_length)
                  log_probs   = model.ctc_decoder(encoder_output=enc)
                  loss        = model.ctc_loss(log_probs, targets, enc_len, target_lengths)
         * INFER: model.transcribe([wav_paths]) -> [Hypothesis.text]   (CTC greedy)
         * LoRA:  real PEFT on conformer attention+FF linears, injected in place. The PEFT
                  wrapper is kept in self.peft (not self.model) so the raw NeMo API stays
                  reachable on self.model for forward/ctc/transcribe.
    '''
    loss_type = "ctc"
    supports_unsloth = False

    def __init__(self, model_name, lang=LANG):
        super().__init__(model_name, lang)
        self.peft = None

    def load_base(self):
        import nemo.collections.asr as nemo_asr
        print(f"[load] {self.model_name}")
        self.model = nemo_asr.models.ASRModel.from_pretrained(self.model_name, map_location=DEVICE)
        self.model.change_decoding_strategy(decoder_type="ctc")   # drive the CTC head
        self.peft = None
        return self.model

    def apply_lora(self, spec: "LoRAConfigSpec"):
        from peft import LoraConfig, get_peft_model
        kw = dict(r=spec.r, lora_alpha=spec.lora_alpha, lora_dropout=spec.lora_dropout,
                  bias=spec.bias, target_modules=spec.target_modules)
        if spec.modules_to_save: kw["modules_to_save"] = spec.modules_to_save
        if spec.task_type:       kw["task_type"] = spec.task_type
        # get_peft_model injects lora.Linear into self.model's submodules IN PLACE and returns a
        # PeftModel wrapper. We keep self.model (the NeMo object) for forward/ctc/transcribe and
        # use self.peft only to save/print/reload the adapter.
        self.peft = get_peft_model(self.model, LoraConfig(**kw))
        self.peft.print_trainable_parameters()
        print("[lora] peft"); return self.peft

    def trainable_parameters(self):
        return [p for p in self.model.parameters() if p.requires_grad]

    def save_checkpoint(self, dir_):
        Path(dir_).mkdir(parents=True, exist_ok=True)
        self.peft.save_pretrained(str(dir_))

    def load_checkpoint(self, dir_):
        from safetensors.torch import load_file
        from peft import set_peft_model_state_dict
        sd = load_file(str(Path(dir_) / "adapter_model.safetensors"))
        set_peft_model_state_dict(self.peft, sd)

    def preprocess(self, ex):
        audio = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
        if sr != 16000:
            import librosa
            audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=16000)
        text = normalize_ar(ex["text"])
        return {"audio": np.asarray(audio, dtype=np.float32), "text": text,
                "audio_len": len(audio) / 16000.0}

    def collate(self, feats):
        sigs = [torch.from_numpy(np.asarray(f["audio"], dtype=np.float32)) for f in feats]
        lens = torch.tensor([int(s.numel()) for s in sigs], dtype=torch.long)
        maxT = int(lens.max())
        sig  = torch.zeros(len(sigs), maxT, dtype=torch.float32)
        for i, s in enumerate(sigs): sig[i, :s.numel()] = s
        toks = [self.model.tokenizer.text_to_ids(f["text"]) for f in feats]
        tlen = torch.tensor([len(t) for t in toks], dtype=torch.long)
        maxL = max(1, int(tlen.max()))
        tgt  = torch.zeros(len(toks), maxL, dtype=torch.long)
        for i, t in enumerate(toks):
            if t: tgt[i, :len(t)] = torch.tensor(t, dtype=torch.long)
        return {"input_signal": sig, "input_signal_length": lens,
                "targets": tgt, "target_lengths": tlen,
                "text": [f["text"] for f in feats], "_audio": [f["audio"] for f in feats]}

    def train_step(self, batch) -> torch.Tensor:
        enc, enc_len = self.model(input_signal=batch["input_signal"].to(DEVICE),
                                  input_signal_length=batch["input_signal_length"].to(DEVICE))
        log_probs = self.model.ctc_decoder(encoder_output=enc)
        return self.model.ctc_loss(log_probs=log_probs,
                                   targets=batch["targets"].to(DEVICE),
                                   input_lengths=enc_len,
                                   target_lengths=batch["target_lengths"].to(DEVICE))

    @torch.no_grad()
    def generate(self, batch):
        import tempfile, soundfile as sf
        was_training = self.model.training
        self.model.eval()
        tmp = tempfile.mkdtemp(prefix="ctc_infer_"); paths = []
        for i, a in enumerate(batch["_audio"]):
            p = os.path.join(tmp, f"{i}.wav")
            sf.write(p, np.asarray(a, dtype=np.float32), 16000); paths.append(p)
        hyps = self.model.transcribe(paths, batch_size=len(paths), verbose=False)
        out = [(h.text if hasattr(h, "text") else str(h)) for h in hyps]
        for p in paths:
            try: os.unlink(p)
            except Exception: pass
        try: os.rmdir(tmp)
        except Exception: pass
        if was_training: self.model.train()
        return out

class Qwen3ASRAdapter(ModelAdapter):
    """Qwen/Qwen3-ASR-0.6B-hf (transformers-native). Verified on GPU (2026-07-16):
      * load:   AutoProcessor + Qwen3ASRForConditionalGeneration (bf16)
      * TRAIN:  processor.apply_chat_template(chat, output_labels=True) -> loss = model(**in).loss
                chat = user turn holding {type:text, text:transcript} + {type:audio, audio:ndarray}
      * INFER:  processor.apply_transcription_request(audio=[...], language=[...]) -> model.generate
                -> processor.decode(gen, return_format="transcription_only")
      * inputs MUST be cast to the model dtype (BatchFeature.to(device, dtype) casts float only).
      * LoRA:   real PEFT on q/k/v/o_proj + gate/up/down_proj -- uses ModelAdapter's default
                apply_lora/save_checkpoint/load_checkpoint as-is, no override needed.
    """
    loss_type = "seq2seq"
    supports_unsloth = False

    def load_base(self):
        from transformers import AutoProcessor, Qwen3ASRForConditionalGeneration
        print(f"[load] {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = Qwen3ASRForConditionalGeneration.from_pretrained(
            self.model_name, dtype=COMPUTE_DTYPE).to(DEVICE)
        self.model.config.use_cache = False
        return self.model

    def preprocess(self, ex):
        audio = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
        if sr != 16000:
            import librosa
            audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=16000)
        text = normalize_ar(ex["text"])
        return {"audio": np.asarray(audio, dtype=np.float32), "text": text,
                "audio_len": len(audio) / 16000.0}

    def collate(self, feats):
        chat = [[{"role": "user", "content": [
                    {"type": "text",  "text":  f["text"]},
                    {"type": "audio", "audio": f["audio"]}]}]
                for f in feats]
        inputs = self.processor.apply_chat_template(
            chat, tokenize=True, return_dict=True, output_labels=True)
        batch = dict(inputs)                       # BatchFeature -> plain dict of tensors
        batch["text"]   = [f["text"] for f in feats]
        batch["_audio"] = [f["audio"] for f in feats]
        return batch

    _MODEL_KEYS = ("input_ids", "attention_mask", "input_features", "input_features_mask", "labels")

    def train_step(self, batch) -> torch.Tensor:
        inputs = {}
        for k in self._MODEL_KEYS:
            v = batch.get(k)
            if v is None: continue
            v = v.to(DEVICE)
            if v.is_floating_point(): v = v.to(COMPUTE_DTYPE)   # input_features -> bf16
            inputs[k] = v
        return self.model(**inputs).loss

    @torch.no_grad()
    def generate(self, batch):
        audios = list(batch["_audio"])
        req = self.processor.apply_transcription_request(
            audio=audios, language=[self.lang] * len(audios))
        req = req.to(DEVICE, COMPUTE_DTYPE)
        out_ids = self.model.generate(**req, max_new_tokens=256)
        gen = out_ids[:, req["input_ids"].shape[1]:]
        return [str(t) for t in self.processor.decode(gen, return_format="transcription_only")]

class CohereAsrAdapter(ModelAdapter):
    """CohereLabs/cohere-transcribe-arabic-07-2026 (transformers-native CohereAsr).
      * load:   AutoProcessor + CohereAsrForConditionalGeneration (bf16)  [GATED repo]
      * TRAIN:  explicit teacher forcing -> loss = model(**in).loss
      * INFER:  processor(audio, language, sampling_rate=16000) -> model.generate
                -> strip prompt -> tokenizer.batch_decode(skip_special_tokens=True)
                (+ processor._reassemble_chunk_texts when clips got chunked)
      * inputs MUST be cast to the model dtype (BatchFeature.to(device, dtype) casts float only).
      * LoRA:   real PEFT on discovered encoder/decoder nn.Linear projections -- target_modules
                is None in ConfigAPI for this model, discovered here at runtime.
    """
    loss_type = "seq2seq"
    supports_unsloth = False

    def load_base(self):
        from transformers import AutoProcessor, CohereAsrForConditionalGeneration
        print(f"[load] {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name)
        self.model = CohereAsrForConditionalGeneration.from_pretrained(
            self.model_name, dtype=COMPUTE_DTYPE).to(DEVICE)
        self.model.config.use_cache = False
        tok = self.processor.tokenizer
        self._prompt_ids = list(self.processor.get_decoder_prompt_ids(
            language=self.lang, punctuation=True))
        eos = self.model.generation_config.eos_token_id
        if isinstance(eos, (list, tuple)): eos = eos[0]
        self._eos_id = int(eos if eos is not None else tok.eos_token_id)
        pad = self.model.config.pad_token_id
        if pad is None: pad = tok.pad_token_id if tok.pad_token_id is not None else self._eos_id
        self._pad_id = int(pad)
        print(f"[load] prompt={tok.convert_ids_to_tokens(self._prompt_ids)} "
              f"eos={self._eos_id} pad={self._pad_id}")
        return self.model

    def _discover_lora_targets(self) -> List[str]:
        """Leaf names of nn.Linear modules that look like attention/FF projections."""
        # Attention CONTENT projections + feed-forward only -- matching the FastConformer
        # sibling adapter. Deliberately excluded: `relative_k_proj` (projects positional
        # embeddings, not content), bare `linear` (the conv-subsampling frontend), and the
        # output heads.
        pat = re.compile(r"^(q|k|v|o)_proj$|^(gate|up|down)_proj$"
                         r"|^linear_(q|k|v|out)$|^linear[12]$|^fc[12]$|^w[123]$")
        names = {n.split(".")[-1] for n, m in self.model.named_modules() if isinstance(m, nn.Linear)}
        targets = sorted(n for n in names if pat.match(n) and n not in {"lm_head", "proj_out"})
        if not targets:
            raise RuntimeError(f"No LoRA targets matched. Available Linear leaves: {sorted(names)}")
        return targets

    def apply_lora(self, spec: "LoRAConfigSpec"):
        """PEFT LoRA. CohereAsr projections are torch.nn.Linear, so this Just Works -- only
        the target_modules discovery step differs from the base class default."""
        from peft import LoraConfig, get_peft_model
        if spec.target_modules is None:
            spec = replace(spec, target_modules=self._discover_lora_targets())
            print(f"[lora] discovered target_modules: {spec.target_modules}")
        kw = dict(r=spec.r, lora_alpha=spec.lora_alpha, lora_dropout=spec.lora_dropout,
                  bias=spec.bias, target_modules=spec.target_modules)
        if spec.modules_to_save: kw["modules_to_save"] = spec.modules_to_save
        if spec.task_type:       kw["task_type"] = spec.task_type
        self.model = get_peft_model(self.model, LoraConfig(**kw))
        self.model.print_trainable_parameters()
        print("[lora] peft"); return self.model

    def preprocess(self, ex):
        audio = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
        if sr != 16000:
            import librosa
            audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=16000)
        text = normalize_ar(ex["text"])
        return {"audio": np.asarray(audio, dtype=np.float32), "text": text,
                "audio_len": len(audio) / 16000.0}

    def _features(self, audios):
        enc = self.processor(audio=[np.asarray(a, dtype=np.float32) for a in audios],
                             language=self.lang, sampling_rate=16000)
        chunk_index = enc.pop("audio_chunk_index", None)
        return enc, chunk_index

    def collate(self, feats):
        enc, chunk_index = self._features([f["audio"] for f in feats])
        if enc["input_features"].shape[0] != len(feats):
            raise RuntimeError(
                f"feature extractor chunked {len(feats)} clips into "
                f"{enc['input_features'].shape[0]} rows -- keep training clips <= max_audio_clip_s")
        tok = self.processor.tokenizer
        P = self._prompt_ids
        seqs = [P + tok(f["text"], add_special_tokens=False)["input_ids"] + [self._eos_id]
                for f in feats]
        maxT = max(len(s) for s in seqs) - 1
        dec_in  = torch.full((len(seqs), maxT), self._pad_id, dtype=torch.long)
        labels  = torch.full((len(seqs), maxT), -100, dtype=torch.long)
        dmask   = torch.zeros((len(seqs), maxT), dtype=torch.long)
        for i, s in enumerate(seqs):
            L = len(s) - 1
            dec_in[i, :L] = torch.tensor(s[:-1], dtype=torch.long)
            dmask[i, :L] = 1
            lab = torch.tensor(s[1:], dtype=torch.long)
            lab[:len(P) - 1] = -100                      # don't train on the prompt
            labels[i, :L] = lab
        batch = {"input_features": enc["input_features"],
                 "decoder_input_ids": dec_in, "decoder_attention_mask": dmask,
                 "labels": labels}
        if "attention_mask" in enc: batch["attention_mask"] = enc["attention_mask"]
        batch["text"]   = [f["text"] for f in feats]
        batch["_audio"] = [f["audio"] for f in feats]
        return batch

    _MODEL_KEYS = ("input_features", "attention_mask", "decoder_input_ids",
                   "decoder_attention_mask", "labels")

    def train_step(self, batch) -> torch.Tensor:
        inputs = {}
        for k in self._MODEL_KEYS:
            v = batch.get(k)
            if v is None: continue
            v = v.to(DEVICE)
            if v.is_floating_point(): v = v.to(COMPUTE_DTYPE)   # input_features -> bf16
            inputs[k] = v
        return self.model(**inputs).loss

    @torch.no_grad()
    def generate(self, batch):
        enc, chunk_index = self._features(list(batch["_audio"]))
        enc = enc.to(DEVICE)
        req = {k: (v.to(COMPUTE_DTYPE) if torch.is_tensor(v) and v.is_floating_point() else v)
               for k, v in enc.items()}
        out_ids = self.model.generate(**req, max_new_tokens=256)
        gen = out_ids[:, req["decoder_input_ids"].shape[1]:]
        texts = self.processor.tokenizer.batch_decode(gen, skip_special_tokens=True)
        if chunk_index is not None and any(c[1] is not None for c in chunk_index):
            texts = self.processor._reassemble_chunk_texts(texts, chunk_index, " ")
        return [str(t).strip() for t in texts]

class WhisperAdapter(ModelAdapter):
    """openai/whisper-large-v3 (transformers-native, no chat template -- log-mel features +
    tokenized labels). Verified on GPU (2026-07-31):
      * load:   AutoProcessor(language=self.lang, task="transcribe") + WhisperForConditionalGeneration
                (bf16). Setting language/task on the processor at load time makes plain
                `tokenizer(text).input_ids` automatically emit the right
                <|startoftranscript|><|lang|><|transcribe|><|notimestamps|> prefix, matching what
                generate() conditions on at inference -- verified real 128x3000 log-mel features
                and a real Arabic transcript round-trip (coherent output, finite loss).
      * TRAIN:  loss = model(input_features=..., labels=...).loss -- forward() shifts labels into
                decoder_input_ids internally via config.decoder_start_token_id, no manual shift
                needed. pad_token_id == eos_token_id here (both 50257), so label padding is masked
                via the tokenizer's attention_mask (not by comparing to a fixed id, which would
                also wipe the real trailing eos).
      * INFER:  model.generate(input_features, language=self.lang, task="transcribe") ->
                tokenizer.batch_decode(skip_special_tokens=True)
      * LoRA:   real PEFT on q/k/v/out_proj + fc1/fc2 (all real torch.nn.Linear, verified via
                named_modules()) -- uses ModelAdapter's default apply_lora/save_checkpoint/
                load_checkpoint as-is, no override needed. Deliberately excluded: proj_out (the
                final vocab projection head, not an attention/FF layer).
    """
    loss_type = "seq2seq"
    supports_unsloth = False

    def load_base(self):
        from transformers import AutoProcessor, WhisperForConditionalGeneration
        print(f"[load] {self.model_name}")
        self.processor = AutoProcessor.from_pretrained(self.model_name, language=self.lang, task="transcribe")
        self.model = WhisperForConditionalGeneration.from_pretrained(
            self.model_name, dtype=COMPUTE_DTYPE).to(DEVICE)
        self.model.config.use_cache = False
        return self.model

    def preprocess(self, ex):
        audio = ex["audio"]["array"]; sr = ex["audio"]["sampling_rate"]
        if sr != 16000:
            import librosa
            audio = librosa.resample(np.asarray(audio, dtype=np.float32), orig_sr=sr, target_sr=16000)
        text = normalize_ar(ex["text"])
        return {"audio": np.asarray(audio, dtype=np.float32), "text": text,
                "audio_len": len(audio) / 16000.0}

    def collate(self, feats):
        enc = self.processor.feature_extractor(
            [f["audio"] for f in feats], sampling_rate=16000, return_tensors="pt")
        tok = self.processor.tokenizer
        # Whisper's decoder has a HARD positional limit -- config.max_target_positions (448 for
        # every checkpoint in this family) -- and forward() raises
        #   ValueError: Labels' sequence length {n} cannot exceed the maximum allowed length
        # for anything longer. TrainConfigSpec.max_label_tokens cannot catch this for Whisper:
        # _ListDS truncates f["labels"], but this adapter's preprocess() emits raw *text* and
        # tokenization happens here, in collate(), so f["labels"] does not exist at that point.
        # Hit for real on omni's long-form segments (479 tokens; 16 of omni's 1,311 rows exceed
        # 448). Truncating at the architectural limit rather than at max_label_tokens keeps the
        # most supervision possible and leaves every other corpus in this study untouched --
        # palTrain/palVal/palTest and casa/jor top out at 168/173/165/132 tokens.
        max_label_len = getattr(getattr(self.model, "config", None), "max_target_positions", 448)
        lab = tok([f["text"] for f in feats], padding=True, truncation=True,
                  max_length=max_label_len, return_tensors="pt")
        labels = lab.input_ids.masked_fill(lab.attention_mask.ne(1), -100)
        return {"input_features": enc.input_features, "labels": labels,
                "text": [f["text"] for f in feats], "_audio": [f["audio"] for f in feats]}

    def train_step(self, batch) -> torch.Tensor:
        feats = batch["input_features"].to(COMPUTE_DTYPE)   # already .to(DEVICE)'d by the caller
        return self.model(input_features=feats, labels=batch["labels"]).loss

    @torch.no_grad()
    def generate(self, batch):
        # .to(DEVICE, COMPUTE_DTYPE) here (not relying on the caller) because PredictAPI calls
        # generate() straight off collate() with no device-move step at all.
        feats = batch["input_features"].to(DEVICE, COMPUTE_DTYPE)
        out_ids = self.model.generate(feats, language=self.lang, task="transcribe", max_new_tokens=256)
        return [t.strip() for t in self.processor.tokenizer.batch_decode(out_ids, skip_special_tokens=True)]

REGISTRY: Dict[str, Callable[..., ModelAdapter]] = {
    "omnilingual-asr/omniASR_LLM_1B":                    OmniASRAdapter,
    "omnilingual-asr/omniASR_LLM_300M":                  OmniASRAdapter,
    "nvidia/stt_ar_fastconformer_hybrid_large_pcd_v1.0": ConformerCTCAdapter,
    "nvidia/stt_ar_fastconformer_hybrid_large_pc_v1.0":  ConformerCTCAdapter,   # no-diacritics twin
    "Qwen/Qwen3-ASR-0.6B-hf":                            Qwen3ASRAdapter,
    "CohereLabs/cohere-transcribe-arabic-07-2026":       CohereAsrAdapter,
    "openai/whisper-large-v3":                            WhisperAdapter,
    "openai/whisper-medium":                             WhisperAdapter,
}

def get_adapter(name, **kw) -> ModelAdapter:
    if name not in REGISTRY: raise KeyError(f"{name} not registered. Have: {list(REGISTRY)}")
    a = REGISTRY[name](name, **kw); a.name = name; return a

from datasets import load_dataset
import soundfile as sf, io, glob as _glob, random as _random

# Real fine-tuning mix (see data_finetune_mix_v2/reports/manifest.json + harmonize_report.json):
#   - 100% of casa/pal, casa/jor, omni, layla, qasr/lev, masc/lev
#   - qasr/non_lev + masc/non_lev sampled (~2:1 hour ratio) to fill each split's target hours
#   - already split into train/val/test by the build (NOT the notebook's own 80/10/10 resplit)
#   - flattened: every shard sits directly under {split}/, one common schema:
#       uid, audio (struct<bytes: WAV PCM16 @16kHz, path>), duration, text, mix_source
REAL_DATA_DIR = Path(DATASET_DIR)  # set in Cell 2

# Force the datasets library's own Arrow cache onto LOCAL disk (Cell 2 points HF_HOME at
# /workspace/asr_env, which is the network mount -- fine for model weights, wrong for this,
# since DATASET_DIR was deliberately materialized locally for fast I/O). Arrow-backed +
# memory-mapped: rows are read from this cache on demand (reclaimable page cache), never
# fully materialized into anonymous process heap -- unlike a pyarrow-concat + to_pylist()
# approach, which silently OOM-killed this run once against the container's ~57GB cgroup
# limit (`cat /sys/fs/cgroup/memory/memory.limit_in_bytes`; `free -h`'s 124GB is the HOST).
_HF_DATASETS_CACHE = Path("/root/Palestinian-ASR/.cache_hf_datasets")
_HF_DATASETS_CACHE.mkdir(parents=True, exist_ok=True)

def _load_split_dataset(split: str, root: Path = None):
    """`root` defaults to DATASET_DIR (the pal view); the stage-1 corpora of runs 2-4 are
    loaded through this same function by passing their own directory."""
    root = Path(root or REAL_DATA_DIR)
    files = sorted(_glob.glob(str(root / split / "*.parquet")))
    if not files:
        raise FileNotFoundError(f"no parquet shards under {root / split}")
    return load_dataset("parquet", data_files=files, split="train",
                         cache_dir=str(_HF_DATASETS_CACHE))

def load_splits(smoke=SMOKE_TEST):
    train_ds = _load_split_dataset("train")
    val_ds   = _load_split_dataset("val")
    test_ds  = _load_split_dataset("test")
    # Uniform shuffle across the whole flattened split -- every mix_source (qasr/masc/casa/
    # omni/layla, lev and non_lev alike) interleaved randomly, not source-blocked. HF
    # .shuffle() only remaps indices -- it does not copy the underlying Arrow data.
    train_ds = train_ds.shuffle(seed=SEED)
    if smoke:
        max_s = ConfigAPI.train(MODEL_NAME).max_audio_seconds
        def _one_short(ds):
            durs = ds["duration"]
            idx = [i for i, d in enumerate(durs) if d <= max_s]
            _random.Random(SEED).shuffle(idx)
            return ds.select(idx[:1])
        train_ds, val_ds, test_ds = _one_short(train_ds), _one_short(val_ds), _one_short(test_ds)
    return {"train": train_ds, "validation": val_ds, "test": test_ds}

SPLITS = load_splits()          # train=palTrain, validation=palVal, test=palTest
print({k: len(v) for k, v in SPLITS.items()})
print({k: sum(v["duration"]) / 3600.0 for k, v in SPLITS.items()})

# Stage-1 corpus (runs 2-4 only): a single `train` split, shuffled the same way. Its
# validation loader during stage 1 is palVal -- with a fixed 1-epoch budget nothing is
# selected on it, it is only there so the epoch prints a comparable WER.
STAGE1_SPLITS = None
if STAGE1_DIR:
    _s1 = _load_split_dataset("train", Path(STAGE1_DIR)).shuffle(seed=SEED)
    STAGE1_SPLITS = {"train": _s1, "validation": SPLITS["validation"]}
    print(f"[stage1] {STAGE1_DIR}: {len(_s1)} rows, "
          f"{sum(_s1['duration']) / 3600.0:.3f}h")


def _fingerprint(ds) -> str:
    try: h = ds._fingerprint
    except Exception: h = str(len(ds))
    return hashlib.md5(f"{h}{len(ds)}".encode()).hexdigest()[:10]

class PredictAPI:
    @staticmethod
    def _path(model_name, split, ds, stage):
        slug = model_name.replace("/", "__")
        return PRED_DIR / f"{slug}__{split}__{_fingerprint(ds)}__{stage}.json"

    @staticmethod
    def run(adapter, ds, split="test", stage="base", batch_size=4, force=False):
        p = PredictAPI._path(adapter.name, split, ds, stage)
        if p.exists() and not force:
            print(f"[predict] CACHE HIT -> {p.name}")
            return json.loads(p.read_text(encoding="utf-8"))

        print(f"[predict] generating ({stage}, {split}, n={len(ds)})")
        feats = []
        for ex in ds:
            wav, sr = sf.read(io.BytesIO(ex["audio"]["bytes"]), dtype="float32")
            if wav.ndim > 1: wav = wav.mean(axis=1)
            feats.append(adapter.preprocess({"audio": {"array": wav, "sampling_rate": sr}, "text": ex["text"]}))
        preds, refs = [], []
        adapter.model.eval()
        for i in range(0, len(feats), batch_size):
            b = adapter.collate(feats[i:i+batch_size])
            b_dev = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
            preds.extend(adapter.generate(b_dev)); refs.extend(b["text"])
            print(f"  {min(i+batch_size,len(feats))}/{len(feats)}", end="\r")
        rec = {"model": adapter.name, "split": split, "stage": stage,
               "predictions": preds, "references": refs,
               "n": len(preds), "ts": time.time()}
        p.write_text(json.dumps(rec, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"\n[predict] saved -> {p.name}")
        return rec

class EvaluateAPI:
    @staticmethod
    def _path(model_name, split, stage, pred_record=None):
        slug = model_name.replace('/', '__')
        if pred_record is None:
            return METRIC_DIR / f"{slug}__{split}__{stage}.json"
        # Content-address the metric to the exact predictions it scores. PredictAPI already
        # keys on the dataset fingerprint; without the same discipline here a metric computed
        # on an older corpus gets silently re-served after the data changes, producing a wrong
        # base WER and a meaningless base->tuned delta.
        payload = json.dumps([pred_record["predictions"], pred_record["references"]],
                             ensure_ascii=False, sort_keys=True).encode("utf-8")
        h = hashlib.md5(payload).hexdigest()[:10]
        return METRIC_DIR / f"{slug}__{split}__{h}__{stage}.json"

    @staticmethod
    def run(model_name, pred_record, split="test", stage="base", force=False):
        p = EvaluateAPI._path(model_name, split, stage, pred_record)
        if p.exists() and not force:
            m = json.loads(p.read_text()); print(f"[eval] CACHE HIT -> {m}"); return m
        m = compute_wer_cer(pred_record["predictions"], pred_record["references"])
        m.update({"model": model_name, "split": split, "stage": stage})
        p.write_text(json.dumps(m, indent=2))
        print(f"[eval] WER={m['wer']:.4f} CER={m['cer']:.4f} (n={m['n']}) -> {p.name}")
        return m

set_seed()
adapter = get_adapter(MODEL_NAME, lang=LANG)
adapter.load_base()
n_params = sum(p.numel() for p in adapter.model.parameters())
print(f"{MODEL_NAME}: {n_params/1e6:.1f}M params | loss_type={adapter.loss_type} | unsloth={adapter.supports_unsloth}")

base_preds   = PredictAPI.run(adapter, SPLITS["test"], split="test", stage="base",
                              batch_size=ConfigAPI.train(MODEL_NAME).per_device_eval_batch_size)
base_metrics = EvaluateAPI.run(MODEL_NAME, base_preds, split="test", stage="base")

for p, r in list(zip(base_preds["predictions"], base_preds["references"]))[:3]:
    print(f"REF : {r}\nHYP : {p}\n")

val_base_preds   = PredictAPI.run(adapter, SPLITS["validation"], split="val", stage="base",
                                  batch_size=ConfigAPI.train(MODEL_NAME).per_device_eval_batch_size)
val_base_metrics = EvaluateAPI.run(MODEL_NAME, val_base_preds, split="val", stage="base")

for p, r in list(zip(val_base_preds["predictions"], val_base_preds["references"]))[:3]:
    print(f"REF : {r}\nHYP : {p}\n")


lora_spec  = ConfigAPI.lora(MODEL_NAME)
train_spec = ConfigAPI.train(MODEL_NAME)
if SMOKE_TEST:
    train_spec.num_epochs = 2
    train_spec.early_stopping_patience = 4

adapter.apply_lora(lora_spec)

import mlflow
from contextlib import nullcontext
from torch.utils.data import DataLoader, Sampler
from transformers.trainer_pt_utils import LengthGroupedSampler as HFLengthGroupedSampler

def _amp(spec):
    """bf16 autocast on CUDA; no-op elsewhere so the loop also runs on CPU."""
    if DEVICE == "cuda":
        return torch.autocast("cuda", dtype=torch.bfloat16)
    return nullcontext()

class _ListDS(torch.utils.data.Dataset):
    """Lazy: decodes audio bytes + calls adapter.preprocess() per item in __getitem__
    (parallelized across DataLoader workers), instead of eagerly decoding + preprocessing
    the WHOLE split into one Python list up front. `rows` is an Arrow-backed (memory-mapped)
    datasets.Dataset (or a .select() view of one) that already passed the cheap
    duration/text filter in TrainAPI._prep -- no audio decode happened yet."""
    def __init__(self, adapter, rows, spec):
        self.adapter = adapter; self.rows = rows; self.spec = spec
    def __len__(self): return len(self.rows)
    def __getitem__(self, i):
        row = self.rows[i]
        wav, sr = sf.read(io.BytesIO(row["audio"]["bytes"]), dtype="float32")
        if wav.ndim > 1: wav = wav.mean(axis=1)
        ex = {"audio": {"array": wav, "sampling_rate": sr}, "text": row["text"]}
        f = self.adapter.preprocess(ex)
        if len(f.get("labels") or []) > self.spec.max_label_tokens:
            f["labels"] = f["labels"][:self.spec.max_label_tokens]
        return f

# LengthGroupedSampler: transformers.trainer_pt_utils.LengthGroupedSampler (imported above),
# not a hand-rolled equivalent -- megabatches sorted by length for padding efficiency,
# megabatch order shuffled every epoch. Lengths come from the cheap `duration` column
# (scaled to ms) -- no audio decode needed just to sort by length.

def _rng_state():
    return {"python": random.getstate(), "numpy": np.random.get_state(),
            "torch": torch.get_rng_state(),
            "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None}

def _restore_rng(state):
    random.setstate(state["python"]); np.random.set_state(state["numpy"])
    torch.set_rng_state(state["torch"])
    if state.get("cuda") is not None and torch.cuda.is_available():
        torch.cuda.set_rng_state_all(state["cuda"])

class TrainAPI:
    @staticmethod
    def _prep(ds, spec):
        """Pre-flight: cheap metadata-only filtering (duration/text columns), no audio
        decode -- decode is deferred to _ListDS.__getitem__ so the full split is never
        held decoded in RAM at once (this container's cgroup memory limit is ~57GB; the
        200h train split decoded to float32 alone is ~46GB, and doing that on top of the
        Arrow table already in memory silently OOM-killed a prior run of this notebook)."""
        durations = ds["duration"]; texts = ds["text"]
        keep_idx, dropped = [], 0
        for i, (d, t) in enumerate(zip(durations, texts)):
            if d is None or d > spec.max_audio_seconds or not (t or "").strip():
                dropped += 1; continue
            keep_idx.append(i)
        print(f"[prep] kept {len(keep_idx)}, dropped {dropped}")
        kept = ds.select(keep_idx)          # index remap over the same Arrow data, no copy
        lengths_ms = [int(durations[i] * 1000) for i in keep_idx]
        return kept, lengths_ms

    @staticmethod
    @torch.no_grad()
    def _validate(adapter, loader, spec):
        adapter.model.eval(); losses, preds, refs = [], [], []
        for b in loader:
            g = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
            try:
                with _amp(spec):
                    losses.append(float(adapter.train_step(g)))
            except Exception as e:
                print(f"[val] loss skipped: {e}")
            preds.extend(adapter.generate(g)); refs.extend(b["text"])
        m = compute_wer_cer(preds, refs)
        m["val_loss"] = float(np.mean(losses)) if losses else float("nan")
        m["_preds"], m["_refs"] = preds, refs
        return m

    @staticmethod
    def run(adapter, splits, spec: TrainConfigSpec, lora_spec: LoRAConfigSpec,
            resume: bool = True, tag: str = "main"):
        """`tag` separates the checkpoint/resume namespace of successive training calls inside
        ONE process (runs 2-4 call this twice: tag="stage1" then tag="stage2"). Without it the
        second call's resume-from-latest scan would find the first call's checkpoints -- built
        against a different dataset, a different LoRA adapter, and a base model that has since
        had stage 1 merged into it -- and silently restore them."""
        slug = adapter.name.replace("/", "__")
        run_root = CKPT_DIR / slug / tag
        best_dir = run_root / "best"; best_dir.mkdir(parents=True, exist_ok=True)

        tr_rows, tr_lengths_ms = TrainAPI._prep(splits["train"], spec)
        va_rows, _ = TrainAPI._prep(splits["validation"], spec)
        tr_kw = {}
        if spec.dataloader_num_workers > 0:
            tr_kw["persistent_workers"] = spec.dataloader_persistent_workers
            tr_kw["prefetch_factor"] = spec.dataloader_prefetch_factor
        tr = DataLoader(_ListDS(adapter, tr_rows, spec), batch_size=spec.per_device_train_batch_size,
                        sampler=HFLengthGroupedSampler(
                            spec.per_device_train_batch_size, lengths=tr_lengths_ms),
                        collate_fn=adapter.collate, num_workers=spec.dataloader_num_workers,
                        pin_memory=(spec.dataloader_pin_memory and DEVICE == "cuda"),
                        drop_last=False, **tr_kw)
        va = DataLoader(_ListDS(adapter, va_rows, spec), batch_size=spec.per_device_eval_batch_size,
                        shuffle=False, collate_fn=adapter.collate,
                        num_workers=spec.dataloader_num_workers)

        params = adapter.trainable_parameters()
        opt = None
        if DEVICE == "cuda":            # bitsandbytes 8-bit optimizers are CUDA-only
            try:
                import bitsandbytes as bnb
                opt = bnb.optim.AdamW8bit(params, lr=spec.learning_rate, weight_decay=spec.weight_decay)
            except Exception as e:
                print(f"[opt] AdamW8bit unavailable ({e}); using torch.AdamW")
        if opt is None:
            opt = torch.optim.AdamW(params, lr=spec.learning_rate, weight_decay=spec.weight_decay)

        steps_pe = max(1, math.ceil(len(tr) / spec.gradient_accumulation_steps))
        total    = steps_pe * spec.num_epochs
        save_steps = spec.save_steps or max(200, steps_pe // 3)
        from transformers import get_linear_schedule_with_warmup
        sched = get_linear_schedule_with_warmup(opt, int(total * spec.warmup_ratio), total)

        best_wer, bad_epochs, gstep, history, start_epoch, mlflow_run_id = \
            float("inf"), 0, 0, [], 1, None
        last_epoch_dir = None

        ckpts = sorted(run_root.glob("ckpt_step*"))
        if resume and ckpts:
            last = ckpts[-1]
            try:
                state = json.loads((last / "trainer_state.json").read_text())
                adapter.load_checkpoint(last)
                # weights_only=False: these are our own trusted local checkpoint files, not
                # untrusted downloads. Needed because torch >=2.6 defaults weights_only=True,
                # which rejects the numpy-backed RNG state (numpy.random.get_state() pickles via
                # numpy's own _reconstruct, not in the default safe-globals allowlist) and can
                # also reject optimizer state depending on the optimizer's internals.
                opt.load_state_dict(torch.load(last / "optimizer.pt", map_location=DEVICE, weights_only=False))
                sched.load_state_dict(torch.load(last / "scheduler.pt", map_location=DEVICE, weights_only=False))
                _restore_rng(torch.load(last / "rng.pt", map_location="cpu", weights_only=False))
                best_wer, bad_epochs = state["best_wer"], state["bad_epochs"]
                gstep, start_epoch = state["gstep"], state["epoch"] + 1
                history, mlflow_run_id = state["history"], state.get("mlflow_run_id")
                print(f"[resume] epoch {start_epoch} gstep {gstep} best_wer {best_wer:.4f} <- {last}")
            except Exception as e:
                print(f"[resume] failed ({e}); starting fresh")

        if mlflow.active_run() is not None:
            mlflow.end_run()
        try:
            mlflow.start_run(run_id=mlflow_run_id, run_name=f"{slug}-{PAL_RUN}-{tag}")
        except Exception as e:
            # A resumed run_id can be unusable for reasons outside our control (belongs to
            # a different active experiment -- e.g. this same checkpoint dir was previously
            # used by a sibling notebook under a different MLflow experiment name; a
            # manually deleted run; a different MLFLOW_TRACKING_URI). Never let bookkeeping
            # block real training -- fall back to a fresh run instead of crashing.
            print(f"[mlflow] could not resume run {mlflow_run_id} ({e}); starting a new run")
            mlflow_run_id = None
            mlflow.start_run(run_id=None, run_name=f"{slug}-{PAL_RUN}-{tag}")
        mlflow_run_id = mlflow.active_run().info.run_id
        try:
            params_flat = {f"train.{k}": (str(v) if isinstance(v, (list, type(None))) else v)
                           for k, v in asdict(spec).items()}
            params_flat.update({f"lora.{k}": (str(v) if isinstance(v, (list, type(None))) else v)
                                for k, v in asdict(lora_spec).items()})
            params_flat.update({"model": adapter.name, "lang": LANG, "smoke": SMOKE_TEST,
                                "save_steps": save_steps})
            mlflow.log_params(params_flat)
        except Exception as e:
            print(f"[mlflow] log_params skipped ({e})")
        try:
            mlflow.set_tags({"dataset_version": REAL_DATA_DIR.name,
                             "pal_run": PAL_RUN, "stage": tag,
                             "train_rows": len(splits["train"]),
                             "hardware": torch.cuda.get_device_name(0) if DEVICE == "cuda" else "cpu",
                             "stage": "dev" if SMOKE_TEST else "experiment"})
        except Exception as e:
            print(f"[mlflow] set_tags skipped ({e})")

        def _save_numbered_ckpt(epoch):
            nonlocal gstep
            d = run_root / f"ckpt_step{gstep:08d}"; d.mkdir(parents=True, exist_ok=True)
            adapter.save_checkpoint(d)
            torch.save(opt.state_dict(), d / "optimizer.pt")
            torch.save(sched.state_dict(), d / "scheduler.pt")
            torch.save(_rng_state(), d / "rng.pt")
            (d / "trainer_state.json").write_text(json.dumps(
                {"epoch": epoch, "gstep": gstep, "best_wer": best_wer, "bad_epochs": bad_epochs,
                 "history": history, "mlflow_run_id": mlflow_run_id}, indent=2))
            kept = sorted(run_root.glob("ckpt_step*"))
            for old in kept[:-spec.save_total_limit] if spec.save_total_limit > 0 else kept:
                shutil.rmtree(old, ignore_errors=True)
            return d

        for epoch in range(start_epoch, spec.num_epochs + 1):
            t_epoch = time.time()
            adapter.model.train(); ep_loss, nb = 0.0, 0
            opt.zero_grad(set_to_none=True)
            epoch_audio_s = 0.0
            for i, b in enumerate(tr):
                t_step = time.time()
                g = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
                with _amp(spec):
                    loss = adapter.train_step(g) / spec.gradient_accumulation_steps
                if not torch.isfinite(loss):
                    print(f"[nan] step {i} non-finite loss, skipping batch")
                    opt.zero_grad(set_to_none=True); continue
                loss.backward()
                if (i + 1) % spec.gradient_accumulation_steps == 0 or (i + 1) == len(tr):
                    gnorm = torch.nn.utils.clip_grad_norm_(params, spec.max_grad_norm)
                    if not torch.isfinite(gnorm):
                        print(f"[nan] step {i} non-finite grad norm, skipping update")
                        opt.zero_grad(set_to_none=True); continue
                    opt.step(); sched.step(); opt.zero_grad(set_to_none=True); gstep += 1
                    mem = torch.cuda.memory_allocated() / 1e9 if DEVICE == "cuda" else 0.0
                    mlflow.log_metrics({"train/step_loss": float(loss) * spec.gradient_accumulation_steps,
                                        "train/grad_norm": float(gnorm),
                                        "train/lr": sched.get_last_lr()[0],
                                        "train/step_time_s": time.time() - t_step,
                                        "train/gpu_mem_gb": mem}, step=gstep)
                    if gstep % save_steps == 0:
                        _save_numbered_ckpt(epoch)
                ep_loss += float(loss) * spec.gradient_accumulation_steps; nb += 1

            train_loss = ep_loss / max(nb, 1)
            vm = TrainAPI._validate(adapter, va, spec)
            row = {"epoch": epoch, "train_loss": train_loss, "val_loss": vm["val_loss"],
                   "val_wer": vm["wer"], "val_cer": vm["cer"]}
            history.append(row)
            throughput = (sum(tr_lengths_ms) / 1000.0) / max(time.time() - t_epoch, 1e-6)
            mlflow.log_metrics({"epoch": epoch, "train/loss": train_loss, "val/loss": vm["val_loss"],
                                "val/wer": vm["wer"], "val/cer": vm["cer"],
                                "train/throughput_audio_s_per_s": throughput,
                                "train/epoch_time_s": time.time() - t_epoch}, step=gstep)
            try:
                import pandas as pd
                qual = pd.DataFrame({"reference": vm["_refs"][:5], "hypothesis": vm["_preds"][:5]})
                mlflow.log_table(qual, artifact_file=f"qualitative/epoch_{epoch:03d}.json")
            except Exception as e:
                print(f"[mlflow] qualitative table skipped ({e})")
            print(f"epoch {epoch:>3} | train {train_loss:.4f} | val {vm['val_loss']:.4f} "
                  f"| WER {vm['wer']:.4f} | CER {vm['cer']:.4f}")

            # ---- per-epoch adapter snapshot, kept forever ----
            # ckpt_step* dirs rotate under save_total_limit and `best` is overwritten whenever a
            # later epoch wins, so neither is a reliable home for "the adapter as of epoch N".
            # These dirs are: one small (~69MB) adapter per epoch, never deleted, which is what
            # makes an N-epoch and an (N+1)-epoch result independently reloadable afterwards.
            ep_dir = run_root / f"epoch{epoch:03d}"
            adapter.save_checkpoint(ep_dir)
            (ep_dir / "epoch.json").write_text(json.dumps({**row, "gstep": gstep}, indent=2))
            last_epoch_dir = str(ep_dir)
            print(f"  -> epoch adapter saved to {ep_dir}")

            # ---- best-WER checkpoint (protected) + early stopping ----
            if vm["wer"] < best_wer - 1e-6:
                best_wer, bad_epochs = vm["wer"], 0
                adapter.save_checkpoint(best_dir)
                (best_dir / "best.json").write_text(json.dumps({**row, "gstep": gstep}, indent=2))
                print(f"  -> new best WER {best_wer:.4f}, saved to {best_dir}")
            else:
                bad_epochs += 1
                print(f"  -> no improvement ({bad_epochs}/{spec.early_stopping_patience})")

            # ---- epoch-boundary checkpoint, regardless of step count: clean resume fallback ----
            _save_numbered_ckpt(epoch)

            if bad_epochs >= spec.early_stopping_patience:
                print(f"[early-stop] epoch {epoch}, best WER {best_wer:.4f}"); break

        mlflow.log_metric("best_val_wer", best_wer, step=gstep)
        (run_root / "history.json").write_text(json.dumps(history, indent=2))
        return {"best_wer": best_wer, "best_dir": str(best_dir), "history": history,
                "mlflow_run_id": mlflow_run_id, "last_epoch_dir": last_epoch_dir,
                "epochs_requested": spec.num_epochs, "run_root": str(run_root)}


# ============================================================================
# The pal study's actual training schedule.
#
#   run1_pal_only : LoRA(r=16) on palTrain with the shared defaults (<=50 epochs, early
#                   stopping on palVal WER, patience 3) -> eval palVal + palTest.
#   run2/3/4      : STAGE 1 -- LoRA(r=16), 1 epoch on the stage-1 corpus (Jordanian
#                   Casablanca / omni / both), then merge_and_unload() so the stage-1 delta
#                   becomes part of the base weights -> eval palVal + palTest on the MERGED
#                   model. STAGE 2 -- a brand new LoRA(r=16) on top of those merged weights,
#                   1 epoch on palTrain -> eval palVal + palTest again.
#
# Every stage's numbers land in `STAGE_METRICS` and are written out by the next cell.
# ============================================================================
STAGE_METRICS = {"base": {"val": val_base_metrics, "test": base_metrics}}
STAGE_PREDS   = {}
EVAL_BS = train_spec.per_device_eval_batch_size


def _final_adapter_dir(out, n_epochs):
    """Which adapter "after N epochs" means. For N>1 that is the LAST epoch's adapter, not
    best_dir -- best_dir holds epoch 1 whenever the later epoch did not improve val WER, and
    reporting it as the N-epoch result would silently re-report the N-1 epoch model."""
    if n_epochs and n_epochs > 1 and out.get("last_epoch_dir"):
        return Path(out["last_epoch_dir"])
    return Path(out["best_dir"])


def eval_both(stage: str, label: str = None, force: bool = True):
    """palVal + palTest under one stage label. force=True on the predict cache: the same
    (model, split, dataset fingerprint) recurs at every stage of every run, so the stage
    label is the only thing separating them -- and a stage label is exactly the kind of
    thing that gets reused by accident. Regenerating is cheap next to trusting it."""
    # `label` is the prediction-cache key. It defaults to the stage name under PAL_RUN so
    # that a resumed run re-serves an earlier run's identical stage (stage1_merged is bitwise
    # the same model on a resume -- stage 1 was skipped, not retrained) instead of spending
    # GPU time recomputing it.
    key = label or f"{PAL_RUN}__{stage}"
    vp = PredictAPI.run(adapter, SPLITS["validation"], split="val", stage=key,
                        batch_size=EVAL_BS, force=force)
    vm = EvaluateAPI.run(MODEL_NAME, vp, split="val", stage=key, force=force)
    tp = PredictAPI.run(adapter, SPLITS["test"], split="test", stage=key,
                        batch_size=EVAL_BS, force=force)
    tm = EvaluateAPI.run(MODEL_NAME, tp, split="test", stage=key, force=force)
    STAGE_METRICS[stage] = {"val": vm, "test": tm}
    STAGE_PREDS[stage]   = {"val": vp, "test": tp}
    print(f"[{stage}] palVal WER={vm['wer']:.4f} CER={vm['cer']:.4f} (n={vm['n']}) | "
          f"palTest WER={tm['wer']:.4f} CER={tm['cer']:.4f} (n={tm['n']})")
    try:
        if mlflow.active_run() is not None:
            mlflow.end_run()
        with mlflow.start_run(run_name=f"{PAL_RUN}-{stage}-eval"):
            mlflow.set_tags({"pal_run": PAL_RUN, "stage": stage, "kind": "eval"})
            mlflow.log_metrics({f"{stage}/val_wer": vm["wer"], f"{stage}/val_cer": vm["cer"],
                                f"{stage}/test_wer": tm["wer"], f"{stage}/test_cer": tm["cer"]})
    except Exception as e:
        print(f"[mlflow] stage metric logging skipped ({e})")
    return vm, tm


# Baseline prediction ran generate() over palTest+palVal just above; release its cached
# blocks before training starts allocating, so the peak is training's alone.
gc.collect()
if DEVICE == "cuda":
    torch.cuda.empty_cache()
    print(f"[gpu] pre-train reserved={torch.cuda.memory_reserved()/1e9:.2f}GB "
          f"allocated={torch.cuda.memory_allocated()/1e9:.2f}GB")

set_seed()
TRAIN_OUTS = {}

if STAGE1_SPLITS is None:
    # ---- run1: single stage ----
    r1_spec = train_spec
    if SINGLE_STAGE_EPOCHS is not None:
        # Fixed budget: patience is set past the epoch count so early stopping can never fire.
        r1_spec = replace(train_spec, num_epochs=SINGLE_STAGE_EPOCHS,
                          early_stopping_patience=SINGLE_STAGE_EPOCHS + 1)
    print(f"\n=== SINGLE STAGE: palTrain | {len(SPLITS['train'])} rows | "
          f"{r1_spec.num_epochs} epoch(s) max | patience {r1_spec.early_stopping_patience} ===")
    train_out = TrainAPI.run(adapter, SPLITS, r1_spec, lora_spec, tag="main")
    TRAIN_OUTS["main"] = train_out
    print(f"best val WER: {train_out['best_wer']:.4f} @ {train_out['best_dir']}")
    final_dir = _final_adapter_dir(train_out, SINGLE_STAGE_EPOCHS)
    try:
        adapter.load_checkpoint(final_dir)
        print(f"[ckpt] evaluating adapter <- {final_dir}")
    except Exception as e:
        print(f"[ckpt] restore failed ({e})")
    mlflow.end_run()
    eval_both("tuned", label=f"{RUN_TAG}__tuned")
else:
    if PAL_STAGE1_CKPT:
        # ---- STAGE 1: SKIPPED -- reuse an already-trained checkpoint instead of retraining ----
        # adapter.model is already LoRA-wrapped (Cell 13's apply_lora ran before this cell), so
        # load_checkpoint's set_peft_model_state_dict has a PeftModel to load onto, same as the
        # trained path just below.
        print(f"\n=== STAGE 1: SKIPPED, loading checkpoint <- {PAL_STAGE1_CKPT} ===")
        adapter.load_checkpoint(Path(PAL_STAGE1_CKPT))
        print(f"[ckpt] stage1 adapter <- {PAL_STAGE1_CKPT}")
        TRAIN_OUTS["stage1"] = {
            "best_wer": None, "best_dir": PAL_STAGE1_CKPT,
            "history": [], "last_epoch_dir": PAL_STAGE1_CKPT,
            "epochs_requested": 1, "run_root": str(Path(PAL_STAGE1_CKPT).parent),
        }
    else:
        # ---- runs 2-4, STAGE 1: 1 epoch on the stage-1 corpus ----
        s1_spec = replace(train_spec, num_epochs=STAGE_EPOCHS,
                          early_stopping_patience=STAGE_EPOCHS + 1)
        print(f"\n=== STAGE 1: {STAGE1_DIR} | {len(STAGE1_SPLITS['train'])} rows | "
              f"{STAGE_EPOCHS} epoch(s) ===")
        TRAIN_OUTS["stage1"] = TrainAPI.run(adapter, STAGE1_SPLITS, s1_spec, lora_spec, tag="stage1")
        mlflow.end_run()

        # Restore the checkpoint explicitly rather than trusting the in-memory weights: for
        # STAGE_EPOCHS==1 this is the same object either way, but it makes the merge provably
        # operate on the same weights that were written to disk. For STAGE_EPOCHS>1, "merge
        # after N epochs" means the LAST epoch's adapter, not best_dir -- best_dir would pick
        # an earlier epoch whenever a later one didn't improve palVal WER (same reasoning as
        # _final_adapter_dir, used identically for the single-stage and stage-2 budgets below).
        _s1_ckpt = _final_adapter_dir(TRAIN_OUTS["stage1"], STAGE_EPOCHS)
        try:
            adapter.load_checkpoint(_s1_ckpt)
            print(f"[ckpt] stage1 adapter <- {_s1_ckpt}")
        except Exception as e:
            print(f"[ckpt] stage1 restore failed ({e})")

    # ---- MERGE: fold the stage-1 LoRA into the base weights ----
    # merge_and_unload() returns the plain WhisperForConditionalGeneration with W <- W + BA*s
    # applied in place of every wrapped projection, so stage 2 starts from a genuinely
    # jor/omni-adapted BASE model, not from a model still carrying a frozen adapter that a
    # second get_peft_model() call would have to stack on top of.
    from peft import PeftModel
    assert isinstance(adapter.model, PeftModel), type(adapter.model)
    adapter.model = adapter.model.merge_and_unload()
    adapter.model.config.use_cache = False
    print(f"[merge] stage-1 LoRA merged -> {type(adapter.model).__name__}, "
          f"{sum(p.numel() for p in adapter.model.parameters())/1e6:.1f}M params")
    gc.collect(); torch.cuda.empty_cache() if DEVICE == "cuda" else None
    MERGED_DIR = CKPT_DIR / "stage1_merged_adapter"
    MERGED_DIR.mkdir(parents=True, exist_ok=True)
    shutil.copytree(_final_adapter_dir(TRAIN_OUTS["stage1"], STAGE_EPOCHS), MERGED_DIR, dirs_exist_ok=True)
    print(f"[merge] stage-1 adapter (the thing needed to rebuild these merged weights) "
          f"kept at {MERGED_DIR}")

    # On a resumed run (STAGE2_EPOCHS > 1) stage 1 was skipped, so these merged weights are
    # identical to the ones already scored -- force=False lets the prediction cache serve them.
    eval_both("stage1_merged", force=(STAGE2_EPOCHS == 1))

    # ---- STAGE 2: fresh LoRA on the merged model ----
    # PAL_STAGE2_EARLYSTOP=1: shared TrainConfigSpec default (<=50 epochs, early stopping on
    # palVal WER, patience 3), same as run1_pal_only's single-stage default. Otherwise: a fixed
    # STAGE2_EPOCHS budget with patience set past it so early stopping can never fire.
    _s2_n_epochs = train_spec.num_epochs if PAL_STAGE2_EARLYSTOP else STAGE2_EPOCHS
    print(f"\n=== STAGE 2: palTrain | {len(SPLITS['train'])} rows | "
          f"{'early-stopping default' if PAL_STAGE2_EARLYSTOP else f'{STAGE2_EPOCHS} epoch(s) fixed'} ===")
    set_seed()
    adapter.apply_lora(lora_spec)
    if PAL_STAGE2_EARLYSTOP:
        s2_spec = train_spec
    else:
        s2_spec = replace(train_spec, num_epochs=STAGE2_EPOCHS,
                          early_stopping_patience=STAGE2_EPOCHS + 1)
    TRAIN_OUTS["stage2"] = TrainAPI.run(adapter, SPLITS, s2_spec, lora_spec, tag="stage2")
    mlflow.end_run()
    s2_final = _final_adapter_dir(TRAIN_OUTS["stage2"], None if PAL_STAGE2_EARLYSTOP else STAGE2_EPOCHS)
    try:
        adapter.load_checkpoint(s2_final)
        print(f"[ckpt] evaluating stage2 adapter <- {s2_final}")
    except Exception as e:
        print(f"[ckpt] stage2 restore failed ({e})")
    eval_both("stage2", label=f"{RUN_TAG}__stage2")


dataset_hours  = {k: sum(v["duration"]) / 3600.0 for k, v in SPLITS.items()}
dataset_counts = {k: len(v) for k, v in SPLITS.items()}

summary = {
    "pal_run": PAL_RUN, "run_tag": RUN_TAG,
    "model": MODEL_NAME, "lang": LANG, "smoke_test": SMOKE_TEST,
    "dataset_dir": str(REAL_DATA_DIR), "stage1_dir": STAGE1_DIR,
    "stages": {s: {"val":  {"wer": m["val"]["wer"],  "cer": m["val"]["cer"],  "n": m["val"]["n"]},
                   "test": {"wer": m["test"]["wer"], "cer": m["test"]["cer"], "n": m["test"]["n"]}}
               for s, m in STAGE_METRICS.items()},
    "delta_vs_base": {
        s: {"val_wer":  STAGE_METRICS["base"]["val"]["wer"]  - m["val"]["wer"],
            "test_wer": STAGE_METRICS["base"]["test"]["wer"] - m["test"]["wer"]}
        for s, m in STAGE_METRICS.items() if s != "base"},
    "train": {t: {"best_wer": o["best_wer"], "best_dir": o["best_dir"],
                  "last_epoch_dir": o.get("last_epoch_dir"),
                  "epochs_requested": o.get("epochs_requested"),
                  "epochs_run": len(o["history"]), "history": o["history"]}
              for t, o in TRAIN_OUTS.items()},
    "dataset_counts": dataset_counts, "dataset_hours": dataset_hours,
    "stage1_rows": (len(STAGE1_SPLITS["train"]) if STAGE1_SPLITS else 0),
    "stage1_hours": (sum(STAGE1_SPLITS["train"]["duration"]) / 3600.0 if STAGE1_SPLITS else 0.0),
    "lora": asdict(lora_spec), "train_spec": asdict(train_spec),
    # train_spec above is the shared baseline; these two record the budget this run actually
    # got, and "train".<tag>.epochs_run below records what it actually used.
    "single_stage_epochs": SINGLE_STAGE_EPOCHS, "stage_epochs": STAGE_EPOCHS,
    "stage2_epochs_requested": STAGE2_EPOCHS, "stage2_earlystop": PAL_STAGE2_EARLYSTOP,
}
sp = METRIC_DIR / f"{MODEL_NAME.replace('/','__')}__{RUN_TAG}__SUMMARY.json"
sp.write_text(json.dumps(summary, indent=2, ensure_ascii=False))
# Second copy next to this run's logs/mlflow db, on local disk -- METRIC_DIR lives on the
# /workspace network mount and is shared by every model family's runs.
lp = Path(f"/root/Palestinian-ASR/Runs/whisper_medium_pal/{RUN_TAG}/SUMMARY.json")
lp.parent.mkdir(parents=True, exist_ok=True)
lp.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

print(f"\n{'stage':>16} | {'palVal WER':>10} {'palVal CER':>10} | {'palTest WER':>11} {'palTest CER':>11}")
for s, m in STAGE_METRICS.items():
    print(f"{s:>16} | {m['val']['wer']:>10.4f} {m['val']['cer']:>10.4f} | "
          f"{m['test']['wer']:>11.4f} {m['test']['cer']:>11.4f}")

try:
    if mlflow.active_run() is not None:
        mlflow.end_run()
    mlflow.start_run(run_name=f"{RUN_TAG}-summary")
    mlflow.log_metrics({f"{s}/{split}_{k}": m[split][k]
                        for s, m in STAGE_METRICS.items() for split in ("val", "test")
                        for k in ("wer", "cer")})
    mlflow.log_metrics({**{f"data/{k}_hours": v for k, v in dataset_hours.items()},
                        **{f"data/{k}_count": v for k, v in dataset_counts.items()}})
    mlflow.log_artifact(str(sp), artifact_path="summary")
    import pandas as pd
    last = list(STAGE_PREDS)[-1]
    mlflow.log_table(pd.DataFrame({"reference": STAGE_PREDS[last]["test"]["references"],
                                   "base_hyp": base_preds["predictions"],
                                   f"{last}_hyp": STAGE_PREDS[last]["test"]["predictions"]}),
                     artifact_file="qualitative/test_predictions.json")
except Exception as e:
    print(f"[mlflow] summary logging skipped ({e})")
finally:
    mlflow.end_run()

print(json.dumps({k: summary[k] for k in ("run_tag", "stages", "delta_vs_base")},
                 indent=2, ensure_ascii=False))