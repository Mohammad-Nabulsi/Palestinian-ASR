"""6 x 50h ascending-score bands for the random-matched corpus.

Identical procedure to the curated 300h run: speakers ordered ascending by their
mean composite score, cumulative hours cut every 50h. Only the speaker SELECTION
differs between the two runs, not the ordering or the banding.
"""
import json, collections
from pathlib import Path
import pyarrow as pa, pyarrow.parquet as pq
H=Path.home(); D=H/"rnd300_data"
w=json.load((H/"sample_weights_composite.json").open())
t=pq.read_table(D/"parquet/train.parquet")
d=t.to_pydict(); n=len(d["uid"])
def score(u,txt):
    v=w.get(f"{u}\t{txt or ''}")
    return v if v is not None else w.get(str(u))
sc=[score(d["uid"][i], d["text"][i]) for i in range(n)]
miss=sum(1 for x in sc if x is None)
sc=[0.0 if x is None else x for x in sc]
print(f"rows {n:,}  hours {sum(d['duration'])/3600:.2f}  uids without a score {miss:,} ({100*miss/n:.1f}%)")

agg=collections.defaultdict(lambda:[0.0,0.0,0])
for i in range(n):
    k=(d["source"][i], str(d["speaker_key"][i]))
    a=agg[k]; a[0]+=float(d["duration"][i] or 0); a[1]+=sc[i]; a[2]+=1
spk=[{"source":s,"speaker_key":k,"hours":v[0]/3600,"score":v[1]/v[2]} for (s,k),v in agg.items()]
spk.sort(key=lambda e:(e["score"], e["hours"], e["source"], e["speaker_key"]))   # ascending, length tiebreak
cum=0.0
for e in spk:
    cum+=e["hours"]; e["cum_hours"]=cum
(D/"rank_meta.json").write_text(json.dumps(spk))
print(f"{len(spk)} speakers, {cum:.2f}h, score {spk[0]['score']:.3f} .. {spk[-1]['score']:.3f}")

# reorder the parquet so rows sit in band order, ascending within band (matches the curated run)
order={(e["source"],e["speaker_key"]):i for i,e in enumerate(spk)}
idx=sorted(range(n), key=lambda i:(order[(d["source"][i],str(d["speaker_key"][i]))], sc[i], d["duration"][i]))
pq.write_table(t.take(idx), D/"parquet/train_ordered.parquet")
print("wrote train_ordered.parquet")
bands=collections.Counter(); bh=collections.Counter()
for e in spk:
    b=min(5,int((e["cum_hours"]-1e-9)//50)); bands[b]+=1; bh[b]+=e["hours"]
print(f"{'band':>5} {'speakers':>9} {'hours':>7}  score range")
lo=0.0
for b in range(6):
    es=[e for e in spk if min(5,int((e['cum_hours']-1e-9)//50))==b]
    print(f"{b+1:5d} {bands[b]:9d} {bh[b]:7.2f}  {es[0]['score']:.3f} .. {es[-1]['score']:.3f}")
