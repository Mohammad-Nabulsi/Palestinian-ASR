# Promoted from the research scratch dir (~/dj.py) on the 5090 node.
# Speaker-disjointness by ID and acoustically, plus length stats.
# See docs/HANDOFF.md for what it produced.
import json,os,collections,statistics,itertools
from pathlib import Path
import numpy as np, torch, torchaudio, soundfile as sf, pyarrow.parquet as pq
for l in (Path.home()/".hf.env").read_text().splitlines():
    if "=" in l: k,v=l.split("=",1); os.environ[k.strip()]=v.strip().strip('"')
os.environ.setdefault("HF_HOME",str(Path.home()/"hf_cache"))
from transformers import AutoFeatureExtractor, WavLMForXVector
M="microsoft/wavlm-base-plus-sv"; dev="cuda"
fe=AutoFeatureExtractor.from_pretrained(M)
mdl=WavLMForXVector.from_pretrained(M,dtype=torch.float16).to(dev).eval()
NAT=Path.home()/"nat_resplit"

def lay_spk(u):
    p=u.split("_"); return f"{p[1]}_{p[2]}" if len(p)>2 else "?"
def load(coll):
    rows=[]
    for sp in ("train","val","test"):
        f=NAT/f"parquet/{coll}_{sp}.parquet"
        d=pq.read_table(f).to_pydict()
        for i in range(len(d["uid"])):
            spk = lay_spk(d["uid"][i]) if coll=="layla" else d["speaker_key"][i]
            rows.append({"split":sp,"uid":d["uid"][i],"spk":spk,
                         "dur":float(d["duration"][i] or 0),"p":d["audio_path"][i]})
    return rows

def wav(p,maxsec=12):
    a,sr=sf.read(NAT/"audio"/p,dtype="float32")
    if a.ndim>1: a=a.mean(1)
    if sr!=16000: a=torchaudio.functional.resample(torch.from_numpy(a),sr,16000).numpy()
    return a[:16000*maxsec] if len(a)>=1600 else None
@torch.no_grad()
def emb(ws):
    x=fe(ws,sampling_rate=16000,return_tensors="pt",padding=True)
    e=mdl(x.input_values.to(dev).half()).embeddings.float()
    return torch.nn.functional.normalize(e,dim=-1).cpu().numpy()

print(f"{'set':22s} {'clips':>6} {'hours':>7} {'spk':>5} {'mean s':>7} {'med s':>7} {'p10':>6} {'p90':>6} {'min':>6} {'max':>6}")
SUM={}
for coll in ("layla","omni"):
    rows=load(coll)
    # per-split speakers
    bysp=collections.defaultdict(set)
    for r in rows: bysp[r["split"]].add(r["spk"])
    for sp in ("train","val","test"):
        v=[r["dur"] for r in rows if r["split"]==sp]
        q=np.percentile(v,[10,90])
        print(f"{coll+'/'+sp:22s} {len(v):6d} {sum(v)/3600:7.2f} {len(bysp[sp]):5d} "
              f"{statistics.mean(v):7.2f} {statistics.median(v):7.2f} {q[0]:6.2f} {q[1]:6.2f} {min(v):6.2f} {max(v):6.2f}")
    inter = (bysp["train"]&bysp["val"]) | (bysp["train"]&bysp["test"]) | (bysp["val"]&bysp["test"])
    print(f"   -> speakers: train {len(bysp['train'])}, val {len(bysp['val'])}, test {len(bysp['test'])}; "
          f"ID overlap: {sorted(inter) if inter else 'NONE'}")
    # acoustic cross-split check: centroid per speaker, then max cross-split speaker pair
    E={}; 
    for r in rows:
        a=wav(r["p"])
        if a is not None: E.setdefault(r["spk"],[]).append((r["split"],a))
    cents={}
    for s,items in E.items():
        vecs=[]
        for i in range(0,len(items),16): vecs.append(emb([x[1] for x in items[i:i+16]]))
        V=np.vstack(vecs); c=V.mean(0); cents[s]=(c/np.linalg.norm(c), items[0][0], len(items))
    names=list(cents)
    worst=[]
    for a,b in itertools.combinations(names,2):
        ca,sa,_=cents[a]; cb,sb,_=cents[b]
        if sa!=sb: worst.append((float(ca@cb),a,sa,b,sb))
    worst.sort(reverse=True)
    same=[float(cents[a][0]@cents[b][0]) for a,b in itertools.combinations(names,2) if cents[a][1]==cents[b][1]]
    print(f"   -> acoustic: highest cross-split speaker pair {worst[0][0]:.3f} "
          f"({worst[0][1]}[{worst[0][2]}] vs {worst[0][3]}[{worst[0][4]}])")
    print(f"      top5 cross-split: {[round(w[0],3) for w in worst[:5]]}")
    print(f"      same-split speaker pairs for scale: mean {statistics.mean(same):.3f} max {max(same):.3f}")
    print(f"      same-voice reference 0.94 | random 0.73")
# casa + broadcast lengths
for coll in ("casa_pal","casa_jor"):
    for sp in ("train","val","test"):
        f=NAT/f"parquet/{coll}_{sp}.parquet"
        d=pq.read_table(f,columns=["duration"]).to_pydict()["duration"]
        if not d: continue
        v=[float(x or 0) for x in d]; q=np.percentile(v,[10,90])
        print(f"{coll+'/'+sp:22s} {len(v):6d} {sum(v)/3600:7.2f} {'-':>5} {statistics.mean(v):7.2f} {statistics.median(v):7.2f} {q[0]:6.2f} {q[1]:6.2f} {min(v):6.2f} {max(v):6.2f}")
for sp in ("train","val","test"):
    d=pq.read_table(Path.home()/f"v3_data/parquet/{sp}.parquet",columns=["duration","source"]).to_pydict()
    for src in ("qasr","masc_c"):
        v=[float(x or 0) for x,s in zip(d["duration"],d["source"]) if s==src]
        q=np.percentile(v,[10,90])
        print(f"{'v3_'+sp+'/'+src:22s} {len(v):6d} {sum(v)/3600:7.2f} {'-':>5} {statistics.mean(v):7.2f} {statistics.median(v):7.2f} {q[0]:6.2f} {q[1]:6.2f} {min(v):6.2f} {max(v):6.2f}")
