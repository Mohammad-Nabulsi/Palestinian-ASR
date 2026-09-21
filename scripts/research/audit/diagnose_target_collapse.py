# Promoted from the research scratch dir (~/diag.py) on the 5090 node.
# Shows raw cosine cannot separate domains and that centering fixes it.
# See docs/HANDOFF.md for what it produced.
import os,collections,statistics,itertools,random
from pathlib import Path
import numpy as np, torch, torchaudio, soundfile as sf, pyarrow.parquet as pq
for l in (Path.home()/".hf.env").read_text().splitlines():
    if "=" in l: k,v=l.split("=",1); os.environ[k.strip()]=v.strip().strip('"')
os.environ.setdefault("HF_HOME",str(Path.home()/"hf_cache"))
from transformers import AutoFeatureExtractor, WavLMForXVector
M="microsoft/wavlm-base-plus-sv"; dev="cuda"; SR=16000
fe=AutoFeatureExtractor.from_pretrained(M)
mdl=WavLMForXVector.from_pretrained(M,dtype=torch.float16).to(dev).eval()
z=np.load(Path.home()/"target_vec/target_vectors.npz",allow_pickle=True)
L,O=z["layla_vecs"],z["omni_vecs"]

print("=== 1. is the embedding space a narrow cone? ===")
m=np.vstack([L,O]).mean(0)
print(f"  ||mean of 119 unit speaker vectors|| = {np.linalg.norm(m):.3f}")
print(f"  (1.0 = all identical; ~0 = spread out. 0.73 'random' floor implies a big shared component)")

print("\n=== 2. embed a sample of QASR speakers and see the ranking spread ===")
d=pq.read_table(Path.home()/"v3_data/parquet/train.parquet",
                columns=["audio_path","source","speaker_key","duration"]).to_pydict()
byspk=collections.defaultdict(list)
for i in range(len(d["source"])):
    if d["source"][i]=="qasr": byspk[d["speaker_key"][i]].append(d["audio_path"][i])
random.seed(0); sample=random.sample(sorted(byspk),120)
def load(p):
    a,sr=sf.read(Path.home()/"v3_data/audio"/p,dtype="float32")
    if a.ndim>1: a=a.mean(1)
    if sr!=SR: a=torchaudio.functional.resample(torch.from_numpy(a),sr,SR).numpy()
    return a[:SR*10]
@torch.no_grad()
def emb(ws):
    x=fe(ws,sampling_rate=SR,return_tensors="pt",padding=True)
    e=mdl(x.input_values.to(dev).half()).embeddings.float()
    return torch.nn.functional.normalize(e,dim=-1).cpu().numpy()
Q=[]
for s in sample:
    ps=byspk[s][:12]; ws=[load(p) for p in ps]; ws=[w for w in ws if len(w)>=1600]
    if not ws: continue
    V=emb(ws); c=V.mean(0); Q.append(c/np.linalg.norm(c))
Q=np.vstack(Q); print(f"  {Q.shape[0]} QASR speaker vectors")

tgt=z["target"]
raw=Q@tgt
print(f"\n  RAW similarity to target: mean {raw.mean():.4f}  sd {raw.std():.4f}  range {raw.min():.3f}..{raw.max():.3f}")

# centered: remove the shared component estimated from everything we have
glob=np.vstack([L,O,Q]).mean(0)
def ctr(X):
    Y=X-glob; return Y/np.linalg.norm(Y,axis=-1,keepdims=True)
Lc,Oc,Qc=ctr(L),ctr(O),ctr(Q)
tc=np.vstack([Lc,Oc]).mean(0); tc/=np.linalg.norm(tc)
cen=Qc@tc
print(f"  CENTERED similarity     : mean {cen.mean():.4f}  sd {cen.std():.4f}  range {cen.min():.3f}..{cen.max():.3f}")
print(f"  spread gain: sd x{cen.std()/raw.std():.1f}")

print("\n=== 3. does centering restore speaker structure? ===")
def pw(X): return [float(X[i]@X[j]) for i,j in itertools.combinations(range(len(X)),2)]
print(f"  within-layla  raw {statistics.mean(pw(L)):.3f} -> centered {statistics.mean(pw(Lc)):.3f}")
print(f"  within-omni   raw {statistics.mean(pw(O)):.3f} -> centered {statistics.mean(pw(Oc)):.3f}")
lo_r=[float(L[i]@O[j]) for i in range(len(L)) for j in range(len(O))]
lo_c=[float(Lc[i]@Oc[j]) for i in range(len(Lc)) for j in range(len(Oc))]
print(f"  layla-vs-omni raw {statistics.mean(lo_r):.3f} -> centered {statistics.mean(lo_c):.3f}")
lq_r=[float(L[i]@Q[j]) for i in range(len(L)) for j in range(0,len(Q),4)]
lq_c=[float(Lc[i]@Qc[j]) for i in range(len(Lc)) for j in range(0,len(Qc),4)]
print(f"  layla-vs-QASR raw {statistics.mean(lq_r):.3f} -> centered {statistics.mean(lq_c):.3f}")
print("\n  -> if centered layla-vs-omni > layla-vs-QASR by a clear margin, the metric now separates domains")
np.savez_compressed(Path.home()/"target_vec/diag.npz", glob=glob, target_centered=tc, qasr_sample=Q)
