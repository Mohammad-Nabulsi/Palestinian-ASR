# Promoted from the research scratch dir (~/sv_check.py) on the 5090 node.
# x-vector check that a QASR speaker_id is acoustically one voice.
# See docs/HANDOFF.md for what it produced.
"""Is a QASR speaker_id acoustically one voice? Verify with a speaker-verification model."""
import json,os,statistics,collections,importlib.util,itertools,random
from pathlib import Path
import numpy as np, torch, pyarrow.parquet as pq
for l in (Path.home()/".hf.env").read_text().splitlines():
    if "=" in l: k,v=l.split("=",1); os.environ[k.strip()]=v.strip().strip('"')
from transformers import AutoFeatureExtractor, WavLMForXVector
spec=importlib.util.spec_from_file_location("ex", str(Path.home()/"Palestinian-ASR/scripts/extract_full_acoustic_speaker_split.py"))
ex=importlib.util.module_from_spec(spec); spec.loader.exec_module(ex)
import soundfile as sf, io

M="microsoft/wavlm-base-plus-sv"
fe=AutoFeatureExtractor.from_pretrained(M); mdl=WavLMForXVector.from_pretrained(M).cuda().eval()

per_utt={}
for sub in ("qasr_lev","qasr_non_lev"):
    with (Path.home()/f"scans/audio_v2/acoustic_full_scan/{sub}/row_probabilities.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            if r.get("status")!="ok": continue
            u=str(r.get("uid","")); i=u.find("qasr:")
            if i>=0: per_utt[u[i:]]=float((r.get("label_scores") or {}).get("Levantine",0.0))

TARGETS=["0C41E87E_D85C_4E40_84EF_689171FD5211","03039DC5_6EDC_4BDA_A97B_B3","0778623F_DA15_4965_866F_06",
         "01B2FCDA_4AE7_4C58_8544_38","07F3CE1A_910C_4F87_B55D_0B"]
rows=collections.defaultdict(list)  # (recording, speaker_id) -> [(lev, si, i)]
tabs=[]
for si,f in enumerate((Path.home()/"lbwork/s.parquet", Path.home()/"lbwork/nl.parquet")):
    d=pq.read_table(f,columns=["uid","speaker_id","recording_id","duration","audio","sampling_rate"]).to_pydict(); tabs.append(d)
    for i in range(len(d["uid"])):
        u=str(d["uid"][i]); j=u.find("qasr:"); nu=u[j:] if j>=0 else u
        if nu not in per_utt: continue
        if not (2.0<=float(d["duration"][i] or 0)<=12): continue
        sid=str(d["speaker_id"][i]); rec=str(d["recording_id"][i])
        if not any(sid.startswith(t) for t in TARGETS): continue
        rows[(rec,sid)].append((per_utt[nu],si,i))

def emb(si,i):
    d=tabs[si]
    wav,_,_=ex.to_wav_bytes(d["audio"][i], d["sampling_rate"][i])
    if wav is None: return None
    a,sr=sf.read(io.BytesIO(wav),dtype="float32")
    if sr!=16000: return None
    with torch.no_grad():
        x=fe(a,sampling_rate=16000,return_tensors="pt").input_values.cuda()
        e=mdl(x).embeddings[0]
    return torch.nn.functional.normalize(e,dim=-1).cpu().numpy()

random.seed(0)
E={}
for k,v in rows.items():
    v=sorted(v); pick = v[:6]+v[-6:] if len(v)>12 else v
    out=[]
    for lev,si,i in pick:
        e=emb(si,i)
        if e is not None: out.append((lev,e))
    if len(out)>=4: E[k]=out
def cos(a,b): return float(np.dot(a,b))

print(f"{'speaker_id':>30} {'n':>3} {'within':>7} {'lo-lo':>7} {'hi-hi':>7} {'lo-hi':>7}")
withins=[]; crosses=[]
for k,v in sorted(E.items()):
    lo=[e for l,e in v if l<=0.2]; hi=[e for l,e in v if l>=0.8]
    w=[cos(a,b) for (_,a),(_,b) in itertools.combinations(v,2)]
    ll=[cos(a,b) for a,b in itertools.combinations(lo,2)] or [float('nan')]
    hh=[cos(a,b) for a,b in itertools.combinations(hi,2)] or [float('nan')]
    lh=[cos(a,b) for a in lo for b in hi] or [float('nan')]
    withins+=w
    print(f"{k[1][-26:]:>30} {len(v):>3} {statistics.mean(w):7.3f} {statistics.mean(ll):7.3f} "
          f"{statistics.mean(hh):7.3f} {statistics.mean(lh):7.3f}")
# cross-cluster, same recording
byrec=collections.defaultdict(list)
for (rec,sid),v in E.items(): byrec[rec].append((sid,v))
for rec,lst in byrec.items():
    for (s1,v1),(s2,v2) in itertools.combinations(lst,2):
        crosses+=[cos(a,b) for _,a in v1 for _,b in v2]
print(f"\nwithin-cluster  mean cos = {statistics.mean(withins):.3f}  (n={len(withins)})")
if crosses: print(f"cross-cluster,  same recording  = {statistics.mean(crosses):.3f}  (n={len(crosses)})")
allv=[e for v in E.values() for _,e in v]
rnd=[cos(a,b) for a,b in random.sample(list(itertools.combinations(allv,2)),min(4000,len(allv)*(len(allv)-1)//2))]
print(f"random pair across everything   = {statistics.mean(rnd):.3f}")
