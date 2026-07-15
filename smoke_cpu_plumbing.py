#!/usr/bin/env python
"""
CPU plumbing smoke test for asr_model_agnostic_finetune.ipynb.

WHY: this box has no GPU and cannot install omnilingual-asr (fairseq2 + CUDA torch, ~6 GB),
so the real-model gates (G1 LOAD / G2 CACHE / G5 BASE PRED) can only run on the RunPod GPU
box. What we CAN prove here is that the model-agnostic *harness* — the exact code in the
notebook's infra cells — runs end to end: preprocess -> collate -> PredictAPI(cache) ->
EvaluateAPI(cache) -> apply_lora(PEFT) -> TrainAPI(loop, early-stop, checkpoint) ->
load best -> tuned predict/eval.

It does so by exec-ing the *real* notebook cells (only `wandb` is stubbed) and driving them
with a FakeAdapter over a tiny nn.Module whose Linear leaves use the real fairseq2 attention
names (q_proj/k_proj/v_proj/output_proj) so PEFT LoRA actually attaches.
"""
import os, sys, json, types, time, io, contextlib, importlib.machinery
from pathlib import Path

SCRATCH = Path("/tmp/claude-0/-root-asr-Palestinian-ASR/4137da38-bb95-4b90-a13b-20eaed8d4580/scratchpad")
os.environ["ASR_ENV_ROOT"] = str(SCRATCH / "asr_env_smoke")
os.environ["WANDB_MODE"] = "disabled"
NB = Path("/root/asr/Palestinian-ASR/asr_model_agnostic_finetune.ipynb")

# ---- stub wandb (not installed; harness imports it in the Train cell) --------------------
_wandb = types.ModuleType("wandb")
class _Run:  # returned by init
    pass
_wandb.init    = lambda *a, **k: _Run()
_wandb.log     = lambda *a, **k: None
_wandb.summary = {}
_wandb.finish  = lambda *a, **k: None
_wandb.Table   = lambda *a, **k: None
_wandb.__spec__ = importlib.machinery.ModuleSpec("wandb", None)  # accelerate does find_spec("wandb")
_wandb.__version__ = "0.0-stub"
sys.modules["wandb"] = _wandb

# ---- pull specific cells out of the notebook by id --------------------------------------
nb = json.loads(NB.read_text())
cells = {c.get("id"): "".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code"}

INFRA_ORDER = [
    "5714e106",  # 1  imports (pip line stripped below)
    "628fd594",  # 2  config + DEVICE + ROOT
    "b5a6fb26",  # 3  normalize_ar + compute_wer_cer
    "267f4147",  # 4  ConfigAPI
    "312a0a67",  # 5  ModelAdapter ABC
    "12e6305e",  # 6  OmniASRAdapter (defines only; method-local imports)
    "27065793",  # 7  registry + get_adapter
    "0c7bde0c",  # 9  PredictAPI
    "6fac93e1",  # 10 EvaluateAPI
    "c0ce5c3a",  # 14 TrainAPI
]

G = {"__name__": "nb_infra"}
for cid in INFRA_ORDER:
    src = cells[cid]
    src = "\n".join(l for l in src.splitlines() if not l.lstrip().startswith("!"))  # drop !pip
    try:
        exec(compile(src, f"<cell {cid}>", "exec"), G)
    except Exception as e:
        print(f"!! infra cell {cid} failed to exec: {type(e).__name__}: {e}")
        raise
print("== infra cells exec'd OK ==")
print(f"   DEVICE={G['DEVICE']}  ROOT={G['ROOT']}")

import numpy as np, torch, torch.nn as nn
ModelAdapter = G["ModelAdapter"]; REGISTRY = G["REGISTRY"]
PredictAPI = G["PredictAPI"]; EvaluateAPI = G["EvaluateAPI"]; TrainAPI = G["TrainAPI"]
ConfigAPI = G["ConfigAPI"]; normalize_ar = G["normalize_ar"]

# ---- a tiny char tokenizer so decode(encode(x)) round-trips (G3 analog) -------------------
ALPHABET = list(" ابتثجحخدذرزسشصضطظعغفقكلمنهوياءأإآةى") + list("0123456789")
CH2ID = {c: i + 1 for i, c in enumerate(ALPHABET)}  # 0 reserved as pad
ID2CH = {i: c for c, i in CH2ID.items()}
PAD_ID = 0
VOCAB = len(ALPHABET) + 1

def enc(text): return [CH2ID[c] for c in text if c in CH2ID]
def dec(ids):  return "".join(ID2CH.get(int(i), "") for i in ids if int(i) != PAD_ID)

# ---- a fake "ASR" nn.Module with fairseq2-style attention names --------------------------
class FakeModel(nn.Module):
    def __init__(self, d=32, vocab=VOCAB):
        super().__init__()
        self.audio_proj  = nn.Linear(1, d)
        self.q_proj      = nn.Linear(d, d)
        self.k_proj      = nn.Linear(d, d)
        self.v_proj      = nn.Linear(d, d)
        self.output_proj = nn.Linear(d, d)   # real fairseq2 name (not o_proj)
        self.head        = nn.Linear(d, vocab)
    def encode_audio(self, wav):
        pooled = wav.float().mean(dim=1, keepdim=True)          # [N,1]
        h = torch.tanh(self.audio_proj(pooled))                # [N,d]
        h = self.output_proj(self.v_proj(self.k_proj(self.q_proj(h))))
        return h                                                # [N,d]

class FakeAdapter(ModelAdapter):
    """Stands in for OmniASRAdapter; exercises every harness method on CPU."""
    loss_type = "seq2seq"; supports_unsloth = False
    def __init__(self, model_name, lang="arb_Arab"):
        super().__init__(model_name, lang); self.pad_idx = PAD_ID
    def load_base(self):
        torch.manual_seed(0); self.model = FakeModel().to(G["DEVICE"]); return self.model
    def preprocess(self, ex):
        audio = np.asarray(ex["audio"]["array"], dtype=np.float32)
        text = normalize_ar(ex["text"]); ids = enc(text)
        return {"input_values": audio, "labels": ids, "text": text,
                "audio_len": len(audio) / 16000.0}
    def collate(self, feats):
        maxa = max(len(f["input_values"]) for f in feats)
        maxl = max(len(f["labels"]) for f in feats) or 1
        wav  = torch.zeros(len(feats), maxa)
        mask = torch.zeros(len(feats), maxa, dtype=torch.long)
        lab  = torch.full((len(feats), maxl), PAD_ID, dtype=torch.long)
        for i, f in enumerate(feats):
            a = torch.as_tensor(f["input_values"]); wav[i, :len(a)] = a; mask[i, :len(a)] = 1
            if f["labels"]: lab[i, :len(f["labels"])] = torch.as_tensor(f["labels"])
        return {"input_values": wav, "attention_mask": mask, "labels": lab,
                "lang": [self.lang]*len(feats), "text": [f["text"] for f in feats]}
    def train_step(self, batch):
        wav = batch["input_values"]; lab = batch["labels"]
        base = self.model.base_model.model if hasattr(self.model, "base_model") else self.model
        h = base.encode_audio(wav) if isinstance(base, FakeModel) else self.model.encode_audio(wav)
        logits = self.model.head(h) if hasattr(self.model, "head") else base.head(h)  # [N,vocab]
        L = lab.shape[1]
        logits = logits.unsqueeze(1).expand(-1, L, -1).reshape(-1, logits.shape[-1])
        loss = nn.functional.cross_entropy(logits, lab.reshape(-1), ignore_index=PAD_ID)
        return loss
    @torch.no_grad()
    def generate(self, batch):
        wav = batch["input_values"]; mask = batch["attention_mask"]
        base = self.model.base_model.model if hasattr(self.model, "base_model") else self.model
        m = base if isinstance(base, FakeModel) else self.model
        h = m.encode_audio(wav); logits = m.head(h)               # [N,vocab]
        top = logits.argmax(dim=-1)                               # [N]
        out = []
        for i in range(wav.shape[0]):
            n = max(1, int(mask[i].sum().item()) // 4000)         # a few tokens
            out.append(dec([int(top[i])] * n))
        return out

FAKE = "fake/tiny-asr"
REGISTRY[FAKE] = FakeAdapter

# ---- fake 1 / 1 / 1 samples -------------------------------------------------------------
def _sample(text, secs, seed):
    rng = np.random.default_rng(seed)
    return {"audio": {"array": rng.standard_normal(int(secs*16000)).astype(np.float32),
                      "sampling_rate": 16000}, "text": text}
SPLITS = {
    "train":      [_sample("مرحبا بالعالم", 1.0, 1)],
    "validation": [_sample("هذا اختبار", 0.8, 2)],
    "test":       [_sample("الطقس جميل اليوم", 1.2, 3)],
}
G["SPLITS"] = SPLITS

# =========================================================================================
print("\n" + "="*78 + "\nGATE-STYLE EVIDENCE (fake adapter, CPU)\n" + "="*78)
adapter = G["get_adapter"](FAKE, lang="arb_Arab"); adapter.load_base()
nparm = sum(p.numel() for p in adapter.model.parameters())
print(f"[G1'] FakeModel loaded: {nparm} params | dtype={next(adapter.model.parameters()).dtype} "
      f"| device={next(adapter.model.parameters()).device}")

# G3' preprocess round-trip
ex = SPLITS["test"][0]; f = adapter.preprocess(ex)
rt = dec(enc(f["text"]))
print(f"\n[G3'] audio shape={f['input_values'].shape} sr=16000 | text='{f['text']}'")
print(f"       label ids[:12]={f['labels'][:12]} | decoded='{dec(f['labels'])}'")
print(f"       round-trip decode(encode(x)) == normalized text ? {rt == f['text']}")
assert rt == f["text"], "round-trip failed"

# G4' collate uneven batch
b = adapter.collate([adapter.preprocess(_sample(t, s, i))
                     for i, (t, s) in enumerate([("اب", .3), ("ابت ابت", .6), ("ا", .9)])])
mask_sums = b["attention_mask"].sum(1).tolist()
true_lens = [int(.3*16000), int(.6*16000), int(.9*16000)]
print(f"\n[G4'] wav {tuple(b['input_values'].shape)} labels {tuple(b['labels'].shape)} "
      f"| mask sums={mask_sums} == true lens {true_lens} ? {mask_sums==true_lens} "
      f"| label pad id == {PAD_ID}")
assert mask_sums == true_lens

# base predict + eval, with cache-hit proof (G6')
def cap(fn, *a, **k):
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf): r = fn(*a, **k)
    return r, buf.getvalue()
bp, o1 = cap(PredictAPI.run, adapter, SPLITS["test"], "test", "base", 2)
bp2, o2 = cap(PredictAPI.run, adapter, SPLITS["test"], "test", "base", 2)
print(f"\n[G6'] 1st predict: {o1.strip().splitlines()[0]}")
print(f"       2nd predict: {o2.strip().splitlines()[-1]}")
assert "CACHE HIT" in o2, "predict cache did not hit"
bm = EvaluateAPI.run(adapter.name, bp, "test", "base")
_, oe2 = cap(EvaluateAPI.run, adapter.name, bp, "test", "base")
assert "CACHE HIT" in oe2, "eval cache did not hit"
print(f"       eval cache HIT on 2nd call ? {'CACHE HIT' in oe2}")
for r, p in zip(bp["references"], bp["predictions"]):
    print(f"       REF:{r!r}  HYP:{p!r}")

# apply real PEFT LoRA on the fake module (proves apply_lora path + target-name matching)
lora = ConfigAPI.lora(FAKE)
lora.target_modules = ["q_proj", "k_proj", "v_proj", "output_proj"]
adapter.apply_lora(lora)
tr = sum(p.numel() for p in adapter.model.parameters() if p.requires_grad)
print(f"\n[LoRA] PEFT attached to {lora.target_modules} | trainable params={tr} (>0 ? {tr>0})")
assert tr > 0

# train loop (short) + checkpoint + tuned eval
spec = ConfigAPI.train(FAKE)
spec.num_epochs = 3; spec.per_device_train_batch_size = 1; spec.gradient_accumulation_steps = 1
spec.dataloader_num_workers = 0; spec.warmup_ratio = 0.0
print("\n[TRAIN] running loop ...")
out = TrainAPI.run(adapter, SPLITS, spec, lora)
print(f"[TRAIN] best_val_wer={out['best_wer']:.4f} | ckpt={out['best_dir']}")
print(f"        loss trajectory: {[round(h['train_loss'],4) for h in out['history']]}")
assert Path(out["best_dir"]).exists()

tp, _ = cap(PredictAPI.run, adapter, SPLITS["test"], "test", "tuned", 2, True)
tm = EvaluateAPI.run(adapter.name, tp, "test", "tuned", force=True)
print(f"\n[RESULT] base WER={bm['wer']:.3f} CER={bm['cer']:.3f} | "
      f"tuned WER={tm['wer']:.3f} CER={tm['cer']:.3f}")

print("\n" + "="*78 + "\nSMOKE PASS: harness runs end-to-end on CPU with fake 1/1/1 samples.\n" + "="*78)
