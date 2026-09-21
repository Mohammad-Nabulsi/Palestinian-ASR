# Promoted from the research scratch dir (~/carve.py) on the 5090 node.
# Carve test/val/train from the similarity ranking, speaker- AND recording-disjoint.
# See docs/HANDOFF.md for what it produced.
"""Carve test / val / train from the similarity ranking, speaker- and recording-disjoint."""
import json,collections,random,statistics
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq
OUT=Path.home()/"sim_sets"; OUT.mkdir(exist_ok=True)
V=Path.home()/"target_vec"
z=np.load(V/"pool_scores.npz",allow_pickle=True)
row,key,sim,dur,ok=z["row"],z["key"],z["sim"],z["dur"],z["ok"]
src=pq.read_table(Path.home()/"v3_data/parquet/train.parquet").to_pydict()
u2s={}
with (Path.home()/"spkmap/uid_speaker.jsonl").open(encoding="utf-8") as f:
    for line in f:
        r=json.loads(line); u2s[r["uid"]]=r["recording_id"]

rows=[]
for j in range(len(row)):
    if not ok[j]: continue
    i=int(row[j]); k=str(key[j])
    rec = f"MASC:{src['speaker_key'][i]}" if k.startswith("MASC:") else u2s.get(src["uid"][i],"?")
    rows.append({"i":i,"spk":k,"rec":rec,"sim":float(sim[j]),"dur":float(dur[j]),
                 "src":src["source"][i]})
print(f"{len(rows):,} scored clips")

agg=collections.defaultdict(lambda:{"n":0,"h":0.0,"s":0.0,"recs":set(),"src":None})
for r in rows:
    a=agg[r["spk"]]; a["n"]+=1; a["h"]+=r["dur"]/3600; a["s"]+=r["sim"]
    a["recs"].add(r["rec"]); a["src"]=r["src"]
for k,a in agg.items(): a["mean"]=a["s"]/a["n"]

MIN_CLIPS=10   # a 1-clip "speaker" tops any mean-similarity ranking by noise alone
elig=[(k,a) for k,a in agg.items() if a["n"]>=MIN_CLIPS]
print(f"speakers {len(agg):,} -> eligible (>={MIN_CLIPS} clips) {len(elig):,}")
qs=sorted([x for x in elig if x[1]["src"]=="qasr"],key=lambda x:-x[1]["mean"])
alls=sorted(elig,key=lambda x:-x[1]["mean"])

def take(cands,target_h,banned_spk,banned_rec):
    picked=[];h=0.0
    for k,a in cands:
        if k in banned_spk or (a["recs"] & banned_rec): continue
        picked.append(k); h+=a["h"]
        if h>=target_h: break
    return picked,h

bs,br=set(),set()
test_spk,th=take(qs,5.0,bs,br)
for k in test_spk: bs.add(k); br|=agg[k]["recs"]
val_spk,vh=take(qs,5.0,bs,br)
for k in val_spk: bs.add(k); br|=agg[k]["recs"]
train_spk,trh=take(alls,50.0,bs,br)
print(f"\ntest : {len(test_spk):4d} spk {th:6.2f}h  sim {agg[test_spk[0]]['mean']:+.3f}..{agg[test_spk[-1]]['mean']:+.3f}")
print(f"val  : {len(val_spk):4d} spk {vh:6.2f}h  sim {agg[val_spk[0]]['mean']:+.3f}..{agg[val_spk[-1]]['mean']:+.3f}")
print(f"train: {len(train_spk):4d} spk {trh:6.2f}h  sim {agg[train_spk[0]]['mean']:+.3f}..{agg[train_spk[-1]]['mean']:+.3f}")
tr_src=collections.Counter(agg[k]["src"] for k in train_spk)
tr_h=collections.Counter(); 
for k in train_spk: tr_h[agg[k]["src"]]+=agg[k]["h"]
print(f"       train mix: "+", ".join(f"{s}: {tr_src[s]} spk / {tr_h[s]:.1f}h" for s in tr_h))

assign={}
for k in test_spk: assign[k]="test"
for k in val_spk: assign[k]="val"
for k in train_spk: assign[k]="train"
schema=pq.read_table(Path.home()/"v3_data/parquet/train.parquet").schema
def write(name,sel,order):
    sel=sorted(sel,key=order)
    cols={n:[src[n][r["i"]] for r in sel] for n in schema.names}
    pq.write_table(pa.table(cols,schema=schema),OUT/f"{name}.parquet")
    h=sum(r["dur"] for r in sel)/3600
    print(f"  wrote {name}.parquet {len(sel):6d} clips {h:6.2f}h")
    json.dump([{"uid":src["uid"][r["i"]],"sim":r["sim"],"spk":r["spk"]} for r in sel],
              (OUT/f"{name}_index.json").open("w"))
    return h
bysplit=collections.defaultdict(list)
for r in rows:
    s=assign.get(r["spk"])
    if s: bysplit[s].append(r)
print("\nwriting:")
write("test",bysplit["test"],lambda r:-r["sim"])
write("val",bysplit["val"],lambda r:-r["sim"])
write("train_ascending",bysplit["train"],lambda r:(r["sim"],r["dur"]))
rnd=list(bysplit["train"]); random.seed(1337); random.shuffle(rnd)
write("train_shuffled",rnd,lambda r:0)

# ---- run 3: random 50h train + random 5h val, disjoint from the shared test ----
banned_spk=set(test_spk); banned_rec=set()
for k in test_spk: banned_rec|=agg[k]["recs"]
pool=[(k,a) for k,a in agg.items() if k not in banned_spk and not (a["recs"]&banned_rec)]
random.seed(7); random.shuffle(pool)
rv,h=[],0.0
for k,a in pool:
    rv.append(k); h+=a["h"]
    if h>=5.0: break
bs2=set(rv); br2=set()
for k in rv: br2|=agg[k]["recs"]
rt,h2=[],0.0
for k,a in pool:
    if k in bs2 or (a["recs"]&br2): continue
    rt.append(k); h2+=a["h"]
    if h2>=50.0: break
print(f"\nrandom run: val {len(rv)} spk {h:.2f}h | train {len(rt)} spk {h2:.2f}h")
ra={}; 
for k in rv: ra[k]="rval"
for k in rt: ra[k]="rtrain"
bs3=collections.defaultdict(list)
for r in rows:
    s=ra.get(r["spk"])
    if s: bs3[s].append(r)
rr=list(bs3["rtrain"]); random.seed(99); random.shuffle(rr)
write("random_train",rr,lambda r:0)
write("random_val",bs3["rval"],lambda r:0)
json.dump({"test":test_spk,"val":val_spk,"train":train_spk,"rtrain":rt,"rval":rv},
          (OUT/"speaker_assignment.json").open("w"))
# disjointness proof
S={n:{r["spk"] for r in v} for n,v in list(bysplit.items())+list(bs3.items())}
R={n:{r["rec"] for r in v} for n,v in list(bysplit.items())+list(bs3.items())}
print("\ndisjointness (speaker / recording overlaps, all must be 0):")
import itertools
for a,b in itertools.combinations(sorted(S),2):
    print(f"  {a:7s} vs {b:7s}: spk {len(S[a]&S[b])}  rec {len(R[a]&R[b])}")
