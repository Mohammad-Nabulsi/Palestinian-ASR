# Promoted from the research scratch dir (~/mkrank.py) on the 5090 node.
# rank_meta JSON so a whole set trains as one chunk.
# See docs/HANDOFF.md for what it produced.
import json, collections
from pathlib import Path
import pyarrow.parquet as pq
S=Path.home()/"sim_sets"
for name in ("train_ascending","train_shuffled","random_train"):
    d=pq.read_table(S/f"{name}.parquet",columns=["source","speaker_key","duration"]).to_pydict()
    seen=[]; h=collections.OrderedDict()
    for s,k,du in zip(d["source"],d["speaker_key"],d["duration"]):
        key=(s,str(k))
        if key not in h: h[key]=0.0
        h[key]+=float(du or 0)
    # every speaker inside the single 50h chunk: cum_hours must stay under chunk_hours
    meta=[]; cum=0.0
    for (s,k),sec in h.items():
        cum+=sec/3600
        meta.append({"source":s,"speaker_key":k,"hours":sec/3600,"cum_hours":min(cum,49.99)})
    (S/f"{name}_rank.json").write_text(json.dumps(meta))
    print(f"{name}: {len(meta)} speakers, {cum:.2f}h -> {name}_rank.json")
