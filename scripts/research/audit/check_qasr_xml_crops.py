# Promoted from the research scratch dir (~/xml_check.py) on the 5090 node.
# Whether QASR's XML-derived clip boundaries are sound.
# See docs/HANDOFF.md for what it produced.
"""Is QASR's XML-based segmentation sound? Test within-speaker vs within-recording spread,
plus boundary sanity (duration vs start/end) and score-vs-duration behaviour."""
import json, statistics, collections
from pathlib import Path
import pyarrow.parquet as pq

# per-utterance scores from the scan
per_utt={}
for sub in ("qasr_lev","qasr_non_lev"):
    fp=Path.home()/f"scans/audio_v2/acoustic_full_scan/{sub}/row_probabilities.jsonl"
    with fp.open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            if r.get("status")!="ok": continue
            u=str(r.get("uid","")); i=u.find("qasr:")
            if i<0: continue
            per_utt[u[i:]]=float((r.get("label_scores") or {}).get("Levantine",0.0))

rows=[]
for f in (Path.home()/"lbwork/s.parquet", Path.home()/"lbwork/nl.parquet"):
    if not f.exists(): continue
    sch=pq.ParquetFile(f).schema_arrow
    want=[c for c in ("uid","speaker_id","recording_id","duration","start_time","end_time") if c in sch.names]
    d=pq.read_table(f,columns=want).to_pydict()
    for i in range(len(d["uid"])):
        u=str(d["uid"][i]); j=u.find("qasr:"); nu=u[j:] if j>=0 else u
        if nu not in per_utt: continue
        rows.append({k:d[k][i] for k in want} | {"lev":per_utt[nu]})
print("joined rows:",len(rows))

# 1. boundary sanity
bad=0; diffs=[]
for r in rows:
    if r.get("start_time") is None or r.get("end_time") is None: continue
    span=float(r["end_time"])-float(r["start_time"]); dur=float(r["duration"] or 0)
    diffs.append(abs(span-dur))
    if span<=0 or abs(span-dur)>0.25: bad+=1
if diffs:
    print(f"\n=== boundary sanity (|(end-start) - duration|) ===")
    print(f"  median {statistics.median(diffs):.4f}s   p95 {sorted(diffs)[int(.95*len(diffs))]:.4f}s   "
          f"segments off by >0.25s or non-positive: {bad} of {len(diffs)} ({100*bad/len(diffs):.2f}%)")

# 2. within-speaker vs within-recording spread
by_rec=collections.defaultdict(list); by_spk=collections.defaultdict(list)
for r in rows:
    if r.get("recording_id"): by_rec[str(r["recording_id"])].append(r["lev"])
    if r.get("speaker_id"): by_spk[str(r["speaker_id"])].append(r["lev"])
rec_sd=[statistics.pstdev(v) for v in by_rec.values() if len(v)>=15]
spk_sd=[statistics.pstdev(v) for v in by_spk.values() if len(v)>=15]
print(f"\n=== spread within a RECORDING vs within one real SPEAKER ===")
print(f"  by recording_id: n={len(rec_sd):4d}  median sd {statistics.median(rec_sd):.3f}")
print(f"  by speaker_id  : n={len(spk_sd):4d}  median sd {statistics.median(spk_sd):.3f}")
ext_spk=[sum(1 for x in v if x<=0.05 or x>=0.95)/len(v) for v in by_spk.values() if len(v)>=15]
print(f"  clips at extremes within one speaker_id: {statistics.mean(ext_spk):.1%}")

# 3. score vs duration
buckets=collections.defaultdict(list)
for r in rows:
    d=float(r["duration"] or 0)
    b = "<2s" if d<2 else ("2-4s" if d<4 else ("4-8s" if d<8 else ">=8s"))
    buckets[b].append(r["lev"])
print(f"\n=== does the classifier behave differently on short clips? ===")
for b in ("<2s","2-4s","4-8s",">=8s"):
    v=buckets.get(b) or []
    if not v: continue
    ext=sum(1 for x in v if x<=0.05 or x>=0.95)/len(v)
    print(f"  {b:6s} n={len(v):6d}  mean {statistics.mean(v):.3f}  at extremes {ext:.1%}")
