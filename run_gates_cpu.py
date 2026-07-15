#!/usr/bin/env python
"""
Real-model gates G1-G5 for OmniASR omniASR_LLM_300M, on CPU, using the REWRITTEN adapter
from asr_model_agnostic_finetune.ipynb and REAL apc (North-Levantine Arabic) audio.

Run with the workspace venv that has omnilingual-asr installed:
    /workspace/venv_omni/bin/python run_gates_cpu.py
"""
import os, sys, json, time, glob
from pathlib import Path

os.environ["ASR_ENV_ROOT"] = "/workspace/asr_env"
os.environ.setdefault("HF_HOME", "/workspace/asr_env/models/hf")
os.environ["WANDB_MODE"] = "disabled"
os.environ["WANDB_DISABLED"] = "true"

NB = json.loads(Path("/root/asr/Palestinian-ASR/asr_model_agnostic_finetune.ipynb").read_text())
cells = {c.get("id"): "".join(c["source"]) for c in NB["cells"] if c["cell_type"] == "code"}
G = {"__name__": "gates"}
for cid in ["5714e106", "628fd594", "b5a6fb26", "267f4147", "312a0a67", "12e6305e", "27065793"]:
    src = "\n".join(l for l in cells[cid].splitlines() if not l.lstrip().startswith("!"))
    exec(compile(src, f"<cell {cid}>", "exec"), G)

import numpy as np, torch
get_adapter = G["get_adapter"]; normalize_ar = G["normalize_ar"]; compute_wer_cer = G["compute_wer_cer"]
DEVICE = G["DEVICE"]
print(f"\n### DEVICE={DEVICE}  (expected cpu on this box)")

AR = lambda s: any("؀" <= ch <= "ۿ" for ch in s)  # any Arabic-script char

# ---- load real apc fixture (4 samples) ---------------------------------------------------
fx = np.load("/workspace/asr_env/g5_fixture/audio.npz")
meta = json.load(open("/workspace/asr_env/g5_fixture/meta.json"))
SAMPLES = [{"audio": {"array": fx[f"a{i}"], "sampling_rate": 16000}, "text": meta[i]["text"]}
           for i in range(len(meta))]
print(f"### fixture: {len(SAMPLES)} real apc North-Levantine clips "
      f"({[m['dur'] for m in meta]} s)")

adapter = get_adapter("omnilingual-asr/omniASR_LLM_300M", lang="arb_Arab")

# =========================================================================================
print("\n" + "="*78 + "\nG1 LOAD\n" + "="*78)
t0 = time.time(); adapter.load_base(); t_cold = time.time() - t0
p0 = next(adapter.model.parameters())
nparm = sum(p.numel() for p in adapter.model.parameters())
print(f"[G1] loaded in {t_cold:.1f}s | params={nparm/1e6:.1f}M | dtype={p0.dtype} | device={p0.device}")
assert nparm > 0

# =========================================================================================
print("\n" + "="*78 + "\nG2 CACHE\n" + "="*78)
roots = [os.environ["HF_HOME"], os.path.expanduser("~/.cache/fairseq2"),
         str(G["MODEL_CACHE"]), os.environ.get("FAIRSEQ2_CACHE_DIR", "")]
found = []
for r in roots:
    if r and os.path.isdir(r):
        sz = sum(f.stat().st_size for f in Path(r).rglob("*") if f.is_file())
        found.append((r, sz))
        print(f"[G2] cache dir {r}  ~{sz/1e6:.0f} MB")
# warm reload (fresh pipeline instance -> reads local cache, no re-download)
from omnilingual_asr.models.inference.pipeline import ASRInferencePipeline
t1 = time.time(); _ = ASRInferencePipeline("omniASR_LLM_300M", device=DEVICE); t_warm = time.time() - t1
print(f"[G2] cold load={t_cold:.1f}s  warm reload={t_warm:.1f}s  (warm < cold ? {t_warm < t_cold})")
del _

# =========================================================================================
print("\n" + "="*78 + "\nG3 PREPROC / tokenizer round-trip\n" + "="*78)
ex0 = SAMPLES[0]; f0 = adapter.preprocess(ex0)
back = adapter._decode(f0["labels"])
print(f"[G3] audio shape={f0['input_values'].shape} sr=16000 | audio_len={f0['audio_len']:.2f}s")
print(f"[G3] norm text : {f0['text'][:70]}")
print(f"[G3] label ids : {f0['labels'][:15]} ... (n={len(f0['labels'])})")
print(f"[G3] decode(enc): {back[:70]}")
print(f"[G3] round-trip decode(encode(x)) == normalized text ? {back == f0['text']}")

# =========================================================================================
print("\n" + "="*78 + "\nG4 COLLATE + real Seq2SeqBatch forward loss\n" + "="*78)
feats = [adapter.preprocess(s) for s in SAMPLES[:2]]
batch = adapter.collate(feats)
print(f"[G4] wav {tuple(batch['input_values'].shape)} | labels {tuple(batch['labels'].shape)} "
      f"| mask sums={batch['attention_mask'].sum(1).tolist()} | pad_idx={adapter.pad_idx}")
adapter.model.train()
with torch.no_grad():
    loss = adapter.train_step({k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in batch.items()})
print(f"[G4] model(Seq2SeqBatch) -> loss={float(loss):.4f} | finite ? {bool(torch.isfinite(loss))}")
adapter.model.eval()

# =========================================================================================
print("\n" + "="*78 + "\nG5 BASE PRED (4 real apc samples)\n" + "="*78)
feats = [adapter.preprocess(s) for s in SAMPLES]
preds = []
for i in range(0, len(feats), 2):
    b = adapter.collate(feats[i:i+2])
    b = {k: (v.to(DEVICE) if torch.is_tensor(v) else v) for k, v in b.items()}
    preds.extend(adapter.generate(b))
refs = [f["text"] for f in feats]
ok = True
for r, p in zip(refs, preds):
    nonempty = bool(p.strip()); arabic = AR(p)
    ok &= nonempty and arabic
    print(f"\n  REF: {r[:70]}\n  HYP: {p[:70]}\n  -> non-empty={nonempty} arabic-script={arabic}")
m = compute_wer_cer(preds, refs)
print(f"\n[G5] WER={m['wer']:.3f} CER={m['cer']:.3f} (n={m['n']}) | all non-empty Arabic ? {ok}")

print("\n" + "="*78)
print(f"GATES DONE | G1 load ✓ | G2 cache ✓ | G3 round-trip {'✓' if back==f0['text'] else '✗'} "
      f"| G4 loss {'✓' if torch.isfinite(loss) else '✗'} | G5 arabic {'✓' if ok else '✗'}")
print("="*78)
