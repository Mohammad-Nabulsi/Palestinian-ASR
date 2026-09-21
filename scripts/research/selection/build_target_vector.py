# Promoted from the research scratch dir (~/target_vec.py) on the 5090 node.
# Layla(5s stratified chunks)+Omni(10s/speaker) x-vector target centroid.
# See docs/HANDOFF.md for what it produced.
"""Build the Layla+Omni target vector.

Layla: every told utterance sliced into 5s chunks; the 109 speakers are dealt
round-robin into 5 groups and group k contributes only its k-th chunk, so the
target is not 109 people reciting the same sentence of the same folk tale.
Omni: 10s per speaker, spread across utterances.
Speaker vectors are averaged, then all speakers averaged equally.
"""
import os, json, collections, statistics
from pathlib import Path
import numpy as np, torch, torchaudio, soundfile as sf, pyarrow.parquet as pq
for l in (Path.home()/".hf.env").read_text().splitlines():
    if "=" in l: k,v=l.split("=",1); os.environ[k.strip()]=v.strip().strip('"')
os.environ.setdefault("HF_HOME",str(Path.home()/"hf_cache"))
from transformers import AutoFeatureExtractor, WavLMForXVector
M="microsoft/wavlm-base-plus-sv"; dev="cuda"; SR=16000; CH=5*SR
fe=AutoFeatureExtractor.from_pretrained(M)
mdl=WavLMForXVector.from_pretrained(M,dtype=torch.float16).to(dev).eval()
OUT=Path.home()/"target_vec"; OUT.mkdir(exist_ok=True)
@torch.no_grad()
def emb(ws):
    x=fe(ws,sampling_rate=SR,return_tensors="pt",padding=True)
    e=mdl(x.input_values.to(dev).half()).embeddings.float()
    return torch.nn.functional.normalize(e,dim=-1).cpu().numpy()
def load(p):
    a,sr=sf.read(p,dtype="float32")
    if a.ndim>1: a=a.mean(1)
    if sr!=SR: a=torchaudio.functional.resample(torch.from_numpy(a),sr,SR).numpy()
    return a
def chunks(a):
    return [a[i:i+CH] for i in range(0,len(a)-CH+1,CH)]

# ---- Layla (told only), stratified chunk index -----------------------------
NAT=Path.home()/"nat_resplit/audio"
rows=[]
for sp in ("train","val","test"):
    f=Path.home()/f"layla_told/layla_told_{sp}.parquet"
    d=pq.read_table(f).to_pydict()
    for i in range(len(d["uid"])):
        u=d["uid"][i]; parts=u.split("_")
        spk=parts[-5]
        rows.append({"spk":spk,"p":d["audio_path"][i],"split":sp,"uid":u})
byspk=collections.defaultdict(list)
for r in rows: byspk[r["spk"]].append(r)
speakers=sorted(byspk)
print(f"layla speakers: {len(speakers)}")
grp={s:i%5 for i,s in enumerate(speakers)}
print("group sizes:",collections.Counter(grp.values()))
lay_vec={}; used=collections.Counter(); miss=[]
for s in speakers:
    k=grp[s]; segs=[]
    for r in byspk[s]:
        cs=chunks(load(NAT/r["p"]))
        if len(cs)>k: segs.append(cs[k])
    if not segs:                       # utterance too short for that index
        for r in byspk[s]:
            cs=chunks(load(NAT/r["p"]))
            if cs: segs.append(cs[0]); break
        miss.append(s)
    V=[]
    for i in range(0,len(segs),24): V.append(emb(segs[i:i+24]))
    V=np.vstack(V); c=V.mean(0); lay_vec[s]=c/np.linalg.norm(c); used[k]+=len(segs)
print(f"layla vectors: {len(lay_vec)}  chunks used per group: {dict(used)}  fellback: {len(miss)}")

# ---- Omni: 10s total per speaker ------------------------------------------
omni=collections.defaultdict(list)
for sp in ("train","val","test"):
    d=pq.read_table(Path.home()/f"nat_resplit/parquet/omni_{sp}.parquet").to_pydict()
    for i in range(len(d["uid"])): omni[d["speaker_key"][i]].append(d["audio_path"][i])
omn_vec={}
for s in sorted(omni):
    ps=omni[s]; picks=[]
    step=max(1,len(ps)//2)
    for j in range(0,len(ps),step):
        cs=chunks(load(NAT/ps[j]))
        if cs: picks.append(cs[len(cs)//2])
        if len(picks)==2: break
    V=emb(picks); c=V.mean(0); omn_vec[s]=c/np.linalg.norm(c)
    print(f"  omni {s}: {len(picks)} chunks = {5*len(picks)}s")
print(f"omni vectors: {len(omn_vec)}")

allv=np.vstack([lay_vec[s] for s in sorted(lay_vec)]+[omn_vec[s] for s in sorted(omn_vec)])
tgt=allv.mean(0); tgt/=np.linalg.norm(tgt)
L=np.vstack([lay_vec[s] for s in sorted(lay_vec)]); O=np.vstack([omn_vec[s] for s in sorted(omn_vec)])
tl=L.mean(0); tl/=np.linalg.norm(tl); to=O.mean(0); to/=np.linalg.norm(to)
bal=(tl+to)/2; bal/=np.linalg.norm(bal)
np.savez_compressed(OUT/"target_vectors.npz",
    target=tgt, target_balanced=bal, layla_centroid=tl, omni_centroid=to,
    layla_speakers=np.array(sorted(lay_vec)), layla_vecs=L,
    omni_speakers=np.array(sorted(omn_vec)), omni_vecs=O)
print(f"\ntarget (119-speaker mean): layla weight {len(lay_vec)/119:.0%}")
print(f"  cos(target, layla_centroid) = {float(tgt@tl):.3f}")
print(f"  cos(target, omni_centroid)  = {float(tgt@to):.3f}")
print(f"  cos(layla_centroid, omni_centroid) = {float(tl@to):.3f}")
print(f"  cos(target, target_balanced) = {float(tgt@bal):.3f}")
print(f"saved -> {OUT/'target_vectors.npz'}")
