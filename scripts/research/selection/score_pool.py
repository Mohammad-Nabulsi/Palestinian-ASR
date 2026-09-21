# Promoted from the research scratch dir (~/score_pool.py) on the 5090 node.
# Score every QASR+MASC clip against the mean-centered target; per-clip + per-speaker.
# See docs/HANDOFF.md for what it produced.
"""Score every clip in the QASR+MASC pool against the centered Layla+Omni target.

Per-clip vectors are needed twice over: aggregated per real speaker (QASR
speaker_id / MASC video_id) to choose WHICH speakers enter each split, and kept
per utterance to order training ascending by similarity.
"""
import os,json,collections,time,statistics
from pathlib import Path
import numpy as np, torch, torchaudio, soundfile as sf, pyarrow.parquet as pq
for l in (Path.home()/".hf.env").read_text().splitlines():
    if "=" in l: k,v=l.split("=",1); os.environ[k.strip()]=v.strip().strip('"')
os.environ.setdefault("HF_HOME",str(Path.home()/"hf_cache"))
from transformers import AutoFeatureExtractor, WavLMForXVector
M="microsoft/wavlm-base-plus-sv"; dev="cuda"; SR=16000
fe=AutoFeatureExtractor.from_pretrained(M)
mdl=WavLMForXVector.from_pretrained(M,dtype=torch.float16).to(dev).eval()
OUT=Path.home()/"target_vec"

print("loading uid->speaker_id map",flush=True)
u2s={}
with (Path.home()/"spkmap/uid_speaker.jsonl").open(encoding="utf-8") as f:
    for line in f:
        r=json.loads(line); u2s[r["uid"]]=r["speaker_id"]
print(f"  {len(u2s):,} uids",flush=True)

d=pq.read_table(Path.home()/"v3_data/parquet/train.parquet",
                columns=["uid","audio_path","duration","source","speaker_key","text"]).to_pydict()
n=len(d["uid"]); items=[]; nomap=0
for i in range(n):
    if d["source"][i]=="qasr":
        sid=u2s.get(d["uid"][i])
        if sid is None or sid=="None": nomap+=1; sid=f"REC:{d['speaker_key'][i]}"
        key=sid
    else:
        key=f"MASC:{d['speaker_key'][i]}"
    items.append((i,key,float(d["duration"][i] or 0)))
print(f"pool {n:,} clips; qasr uids without speaker_id: {nomap:,}",flush=True)
print(f"distinct speaker keys: {len({x[1] for x in items}):,}",flush=True)

def load(p):
    a,sr=sf.read(Path.home()/"v3_data/audio"/p,dtype="float32")
    if a.ndim>1: a=a.mean(1)
    if sr!=SR: a=torchaudio.functional.resample(torch.from_numpy(a),sr,SR).numpy()
    return a[:SR*20]
@torch.no_grad()
def emb(ws):
    x=fe(ws,sampling_rate=SR,return_tensors="pt",padding=True)
    e=mdl(x.input_values.to(dev).half()).embeddings.float()
    return torch.nn.functional.normalize(e,dim=-1).cpu().numpy()

order=sorted(range(len(items)),key=lambda j:items[j][2])
E=np.zeros((len(items),512),dtype=np.float32); ok=np.zeros(len(items),bool)
buf=[];bi=[];t0=time.time();done=0
for j in order:
    i=items[j][0]
    try: a=load(d["audio_path"][i])
    except Exception: continue
    if len(a)<1600: continue
    buf.append(a); bi.append(j)
    if len(buf)==48:
        E[bi]=emb(buf); ok[bi]=True; done+=len(bi); buf=[];bi=[]
        if done%24000<48:
            el=time.time()-t0
            print(f"[{time.strftime('%H:%M:%S')}] {done:,}/{len(items):,}  {done/el:.0f}/s  eta {(len(items)-done)/max(done/el,1)/60:.1f}m",flush=True)
if buf: E[bi]=emb(buf); ok[bi]=True; done+=len(bi)
print(f"embedded {done:,} clips in {time.time()-t0:.0f}s",flush=True)

z=np.load(OUT/"target_vectors.npz",allow_pickle=True)
L,O=z["layla_vecs"],z["omni_vecs"]
glob=np.vstack([L,O,E[ok]]).mean(0)                 # shared component over everything
def ctr(X):
    Y=X-glob; return Y/np.linalg.norm(Y,axis=-1,keepdims=True)
tgt=np.vstack([ctr(L),ctr(O)]).mean(0); tgt/=np.linalg.norm(tgt)
sims=np.full(len(items),-9.0,dtype=np.float32); sims[ok]=ctr(E[ok])@tgt
np.savez_compressed(OUT/"pool_scores.npz", glob=glob, target_centered=tgt,
    row=np.array([it[0] for it in items]), key=np.array([it[1] for it in items]),
    sim=sims, dur=np.array([it[2] for it in items],dtype=np.float32), ok=ok)
print(f"centered sim: mean {sims[ok].mean():.3f} sd {sims[ok].std():.3f} range {sims[ok].min():.3f}..{sims[ok].max():.3f}",flush=True)
agg=collections.defaultdict(lambda:[0,0.0,0.0])
for j,(i,k,du) in enumerate(items):
    if not ok[j]: continue
    a=agg[k]; a[0]+=1; a[1]+=du; a[2]+=float(sims[j])
spk={k:{"clips":v[0],"hours":v[1]/3600,"mean_sim":v[2]/v[0]} for k,v in agg.items()}
json.dump(spk,(OUT/"speaker_scores.json").open("w"),indent=None)
ms=sorted(spk.items(),key=lambda x:-x[1]["mean_sim"])
qs=[x for x in ms if not x[0].startswith("MASC:")]
print(f"\nspeakers: {len(spk):,}  (qasr {len(qs):,}, masc {len(ms)-len(qs):,})")
print("top 8 speakers by mean centered similarity:")
for k,v in ms[:8]: print(f"  {k[:44]:44s} {v['mean_sim']:+.3f} {v['hours']:6.2f}h {v['clips']:5d}")
print("bottom 3:")
for k,v in ms[-3:]: print(f"  {k[:44]:44s} {v['mean_sim']:+.3f} {v['hours']:6.2f}h {v['clips']:5d}")
c=0.0
for k,v in qs:
    c+=v["hours"]
    if c>=10: break
print(f"\nQASR: top speakers reach 10h at mean_sim {v['mean_sim']:+.3f}")
print("ALL DONE",flush=True)
