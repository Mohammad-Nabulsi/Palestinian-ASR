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

MODEL_NAME = "openai/whisper-medium"
LANG = "ar"
SEED = 42
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
COMPUTE_DTYPE = torch.bfloat16 if DEVICE == "cuda" else torch.float32
ROOT = Path(os.environ.get("ASR_ENV_ROOT", "/workspace/asr_env"))
MODEL_CACHE = ROOT / "models"
PRED_DIR = ROOT / "preds"
METRIC_DIR = ROOT / "metrics"
for d in (MODEL_CACHE, PRED_DIR, METRIC_DIR): d.mkdir(parents=True, exist_ok=True)
os.environ["HF_HOME"] = str(MODEL_CACHE / "hf")

def set_seed(s=SEED):
    random.seed(s); np.random.seed(s); torch.manual_seed(s); torch.cuda.manual_seed_all(s)
set_seed()

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


REGISTRY = {"openai/whisper-medium": WhisperAdapter}
def get_adapter(name, **kw):
    a = REGISTRY[name](name, **kw); a.name = name; return a

from datasets import load_dataset
import soundfile as sf, io, glob as _glob

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


# ============================================================================
# Generalization-eval driver: given a checkpoint name (from generalization_matrix.json),
# load base, sequentially merge its "merge_adapters" list into the base weights, optionally
# apply a final (unmerged) LoRA on top, then predict+eval on every listed domain.
# ============================================================================
from peft import PeftModel

def load_checkpoint_stack(adapter, lora_spec, merge_adapters, final_adapter):
    for i, path in enumerate(merge_adapters):
        adapter.apply_lora(lora_spec)
        adapter.load_checkpoint(Path(path))
        assert isinstance(adapter.model, PeftModel), type(adapter.model)
        adapter.model = adapter.model.merge_and_unload()
        adapter.model.config.use_cache = False
        print(f"[merge {i+1}/{len(merge_adapters)}] {path}")
    if final_adapter:
        adapter.apply_lora(lora_spec)
        adapter.load_checkpoint(Path(final_adapter))
        print(f"[final adapter, unmerged] {final_adapter}")


def run_checkpoint(spec, eval_sets_dir="/root/Palestinian-ASR/eval_sets"):
    name = spec["name"]
    print(f"\n{'='*70}\n=== CHECKPOINT: {name} ===\n{'='*70}")
    set_seed()
    adapter = get_adapter(MODEL_NAME, lang=LANG)
    adapter.load_base()
    lora_spec = ConfigAPI.lora(MODEL_NAME)
    load_checkpoint_stack(adapter, lora_spec, spec["merge_adapters"], spec.get("final_adapter"))

    results = {}
    for domain in spec["domains"]:
        files = [str(Path(eval_sets_dir) / f"{domain}.parquet")]
        ds = load_dataset("parquet", data_files=files, split="train")
        label = f"genexp__{name}__{domain}"
        pred = PredictAPI.run(adapter, ds, split=domain, stage=label, batch_size=16, force=True)
        metric = EvaluateAPI.run(MODEL_NAME, pred, split=domain, stage=label, force=True)
        results[domain] = {"wer": metric["wer"], "cer": metric["cer"], "n": metric["n"]}
        print(f"[{name}][{domain}] WER={metric['wer']:.4f} CER={metric['cer']:.4f} (n={metric['n']})")

    out_dir = Path("/root/Palestinian-ASR/generalization_results")
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{name}.json"
    out_path.write_text(json.dumps({"checkpoint": name, "results": results}, indent=2, ensure_ascii=False))
    print(f"[done] {name} -> {out_path}")

    del adapter
    gc.collect()
    if DEVICE == "cuda":
        torch.cuda.empty_cache()


if __name__ == "__main__":
    import sys
    matrix = json.loads(Path("/root/Palestinian-ASR/generalization_matrix.json").read_text())
    by_name = {c["name"]: c for c in matrix}
    names = sys.argv[1:]
    if not names:
        raise SystemExit("usage: run_generalization_eval.py <checkpoint_name> [<checkpoint_name> ...]")
    for n in names:
        if n not in by_name:
            raise SystemExit(f"unknown checkpoint {n!r}. Have: {sorted(by_name)}")
        run_checkpoint(by_name[n])
