#!/usr/bin/env python3
# Promoted from the research scratch dir (~/ood_did_scan.py) on the 5090 node.
# Zero-shot acoustic+text dialect ID over every naturalistic eval split.
# See docs/HANDOFF.md for what it produced.
"""Zero-shot textual + acoustic dialect ID over every naturalistic eval split.

Same two classifiers the corpus selection used:
  acoustic  badrex/mms-300m-arabic-dialect-identifier
  text      IbrahimAmin/marbertv2-arabic-written-dialect-classifier
so the scores here are directly comparable to the broadcast scan that produced
the 300h curriculum -- that is the whole point of running it.
"""
import json, os, sys, time, glob
from pathlib import Path
import numpy as np, torch, torchaudio, soundfile as sf, pyarrow.parquet as pq

for l in (Path.home()/".hf.env").read_text().splitlines():
    if "=" in l: k,v=l.split("=",1); os.environ[k.strip()]=v.strip().strip('"')
os.environ.setdefault("HF_HOME", str(Path.home()/"hf_cache"))
from transformers import (AutoFeatureExtractor, AutoModelForAudioClassification,
                          AutoTokenizer, AutoModelForSequenceClassification)

AC_ID = "badrex/mms-300m-arabic-dialect-identifier"
TX_ID = "IbrahimAmin/marbertv2-arabic-written-dialect-classifier"
OUT   = Path.home()/"ood_did_scan"; OUT.mkdir(exist_ok=True)
dev   = "cuda"

SETS = []
for f in sorted(glob.glob(str(Path.home()/"nat_resplit/parquet/*.parquet"))):
    stem = Path(f).stem                      # e.g. casa_pal_train
    coll, split = stem.rsplit("_", 1)
    SETS.append(("nat_resplit", coll, split, f, Path.home()/"nat_resplit/audio"))
for f in sorted(glob.glob(str(Path.home()/"casa_orig/parquet/*.parquet"))):
    stem = Path(f).stem                      # e.g. casa_pal_validation
    coll, split = stem.rsplit("_", 1)
    SETS.append(("casa_orig", coll, split, f, Path.home()/"casa_orig/audio"))

print(f"[{time.strftime('%H:%M:%S')}] loading classifiers", flush=True)
afe = AutoFeatureExtractor.from_pretrained(AC_ID)
am  = AutoModelForAudioClassification.from_pretrained(AC_ID, dtype=torch.float16).to(dev).eval()
alab= [am.config.id2label[i] for i in range(len(am.config.id2label))]
tok = AutoTokenizer.from_pretrained(TX_ID)
tm  = AutoModelForSequenceClassification.from_pretrained(TX_ID).to(dev).eval()
tlab= [tm.config.id2label[i] for i in range(len(tm.config.id2label))]
print("acoustic labels:", alab, flush=True)
print("text labels    :", tlab, flush=True)
(OUT/"labels.json").write_text(json.dumps({"acoustic":alab,"text":tlab},ensure_ascii=False,indent=2))

def lev_key(labels):
    for cand in ("Levantine","LEV","lev","Levant"):
        if cand in labels: return cand
    return None
AK, TK = lev_key(alab), lev_key(tlab)
print(f"Levantine key -> acoustic={AK!r} text={TK!r}", flush=True)

@torch.no_grad()
def acoustic(waves):
    x = afe(waves, sampling_rate=16000, return_tensors="pt", padding=True)
    x = {k:(v.to(dev).half() if v.dtype==torch.float32 else v.to(dev)) for k,v in x.items()}
    return torch.softmax(am(**x).logits.float(), -1).cpu().numpy()

@torch.no_grad()
def textual(texts):
    x = tok(texts, return_tensors="pt", padding=True, truncation=True, max_length=128).to(dev)
    return torch.softmax(tm(**x).logits.float(), -1).cpu().numpy()

fout = (OUT/"rows.jsonl").open("w", encoding="utf-8")
t0 = time.time(); ntot = 0
for dataset, coll, split, pqf, aroot in SETS:
    d = pq.read_table(pqf).to_pydict()
    n = len(d["uid"])
    if n == 0:
        print(f"[{time.strftime('%H:%M:%S')}] {dataset}/{coll}/{split}: EMPTY, skipped", flush=True); continue
    B, done, bad = 8, 0, 0
    while done < n:
        idx, waves = [], []
        while len(idx) < B and done < n:
            i = done; done += 1
            p = aroot/d["audio_path"][i]
            try:
                a, sr = sf.read(p, dtype="float32")
                if a.ndim > 1: a = a.mean(1)
                # Casablanca ships at 44.1 kHz while layla/omni are already 16 k.
                # Both classifiers are 16 k models, so resample rather than skip --
                # an sr guard here silently discarded every Palestinian clip on the
                # first run (0/1328 scored) without failing.
                if sr != 16000:
                    a = torchaudio.functional.resample(
                        torch.from_numpy(a), sr, 16000).numpy()
                    sr = 16000
                if len(a) < 1600: bad += 1; continue
                if int(np.abs(a*32768).max()) <= 2: bad += 1; continue
            except Exception:
                bad += 1; continue
            idx.append(i); waves.append(a[:16000*30])
        if not idx: continue
        ap = acoustic(waves)
        tp = textual([str(d["text"][i] or "") for i in idx])
        for j, i in enumerate(idx):
            fout.write(json.dumps({
                "dataset": dataset, "collection": coll, "split": split,
                "uid": str(d["uid"][i]), "speaker_key": str(d["speaker_key"][i]),
                "duration": float(d["duration"][i] or 0), "text": str(d["text"][i] or ""),
                "audio_scores": {alab[k]: float(v) for k,v in enumerate(ap[j])},
                "text_scores":  {tlab[k]: float(v) for k,v in enumerate(tp[j])},
                "audio_lev": float(ap[j][alab.index(AK)]) if AK else None,
                "text_lev":  float(tp[j][tlab.index(TK)]) if TK else None,
            }, ensure_ascii=False) + "\n")
        ntot += len(idx)
        if ntot % 500 < B:
            print(f"[{time.strftime('%H:%M:%S')}] {ntot} rows  ({time.time()-t0:.0f}s)", flush=True)
    print(f"[{time.strftime('%H:%M:%S')}] DONE {dataset}/{coll}/{split}: {n-bad}/{n} scored, {bad} unusable", flush=True)
fout.close()
print(f"[{time.strftime('%H:%M:%S')}] ALL DONE {ntot} rows in {time.time()-t0:.0f}s -> {OUT/'rows.jsonl'}", flush=True)
