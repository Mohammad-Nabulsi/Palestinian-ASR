# Promoted from the research scratch dir (~/random300c.py) on the 5090 node.
# Size-matched random 300h ablation corpus (v3's units/hours, random score).
# See docs/HANDOFF.md for what it produced.
"""Size-matched random 300h: same units, same count, same hours -- only the
selection criterion differs.

A uniform draw gives ~2.8 min per QASR speaker while v3's average 22 min, so
1468 random units yield 82h, not 300h. Matching each v3 unit to a random unit of
similar duration holds count AND hours fixed and leaves the Levantine score as
the only thing that varies -- which is the ablation.
"""
import json,random,collections,statistics,bisect
from pathlib import Path
import pyarrow.parquet as pq
H=Path.home(); OUT=H/"random300_matched"; OUT.mkdir(exist_ok=True)
random.seed(4242)
BAN_V3_EVAL=True  # keep the draw clear of the 300h run's val/test units
clips=[]
for sub in ("qasr_lev","qasr_non_lev","masc_lev","masc_non_lev"):
    with (H/f"scans/audio_v2/acoustic_full_scan/{sub}/row_probabilities.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            if r.get("status")!="ok": continue
            clips.append((r["source"],str(r["uid"]),str(r.get("speaker_key")),
                          float(r.get("duration_sec") or 0),
                          float((r.get("label_scores") or {}).get("Levantine",0.0))))
u2r={}
with (H/"spkmap/uid_speaker.jsonl").open(encoding="utf-8") as f:
    for line in f:
        r=json.loads(line); u2r[r["uid"]]=r["recording_id"]
# v3's own unit: qasr recording_id, masc video_id
rows=[]
for src,uid,sk,dur,lev in clips:
    if dur<=0: continue
    key=(u2r.get(uid) or sk) if src=="qasr" else f"MASC:{sk}"
    rows.append({"src":src,"uid":uid,"unit":str(key),"dur":dur,"lev":lev})
by=collections.defaultdict(list)
for r in rows: by[r["unit"]].append(r)
h_of={k:sum(x["dur"] for x in v)/3600 for k,v in by.items()}
src_of={k:v[0]["src"] for k,v in by.items()}
lev_of={k:statistics.mean(x["lev"] for x in v) for k,v in by.items()}
print(f"pool units: qasr {sum(1 for k in by if src_of[k]=='qasr'):,} / "
      f"{sum(h_of[k] for k in by if src_of[k]=='qasr'):.1f}h | "
      f"masc {sum(1 for k in by if src_of[k]=='masc_c'):,} / "
      f"{sum(h_of[k] for k in by if src_of[k]=='masc_c'):.1f}h")

# v3's units and their sizes
v3units=collections.defaultdict(lambda: collections.defaultdict(float))
v3src={}
for s in ("train","val","test"):
    d=pq.read_table(H/f"v3_data/parquet/{s}.parquet",columns=["speaker_key","source","duration"]).to_pydict()
    for k,src,du in zip(d["speaker_key"],d["source"],d["duration"]):
        key=str(k) if src=="qasr" else f"MASC:{k}"
        v3units[s][key]+=float(du or 0)/3600; v3src[key]=src
# Exclude every unit the 300h run evaluates on, so this corpus can be scored
# against that same val/test and compared with 30.85 directly. QASR speaker_ids
# are recording-scoped (<recording_id>_speakerN), so banning the recording bans
# every speaker inside it.
banned=set()
if BAN_V3_EVAL:
    for s_ in ("val","test"):
        banned |= set(v3units[s_].keys())
    print(f"banned {len(banned)} v3 val/test units from the draw")
used=set(banned)
def match(targets,src):
    """for each target size, take a random unused unit of similar duration"""
    cands=sorted((h_of[k],k) for k in by if src_of[k]==src and k not in used)
    sizes=[c[0] for c in cands]; picked=[]
    for t in sorted(targets,reverse=True):
        if not cands: break
        i=bisect.bisect_left(sizes,t)
        lo=max(0,i-25); hi=min(len(cands),i+25)
        win=[c for c in cands[lo:hi] if c[1] not in used]
        if not win: win=[c for c in cands if c[1] not in used][-25:]
        if not win: break
        pick=random.choice(win); used.add(pick[1]); picked.append(pick[1])
    return picked
plan={}
for s in ("train",):
    plan[s]={}
    for src in ("qasr","masc_c"):
        tg=[v for k,v in v3units[s].items() if v3src[k]==src]
        plan[s][src]=match(tg,src)
final=collections.defaultdict(list)
for s in ("train",):
    for src in ("qasr","masc_c"):
        for k in plan[s][src]: final[s]+=by[k]
print(f"\n{'split':7s} {'clips':>8} {'hours':>8} {'units':>7} {'qasr u':>7} {'masc u':>7} {'h/unit':>7} {'mean lev':>9} {'med lev':>8} {'>=0.8':>7}")
rs={}
for s in ("train",):
    v=final[s]; h=sum(r["dur"] for r in v)/3600; un={r["unit"] for r in v}
    q=len([k for k in un if not k.startswith("MASC:")]); lv=[r["lev"] for r in v]
    rs[s]={"clips":len(v),"hours":round(h,2),"units":len(un),"qasr_units":q,"masc_units":len(un)-q,
           "mean_lev":round(statistics.mean(lv),4),"median_lev":round(statistics.median(lv),4),
           "pct_ge_0.8":round(100*sum(1 for x in lv if x>=0.8)/len(lv),1)}
    a=rs[s]
    print(f"{s:7s} {a['clips']:8d} {a['hours']:8.2f} {a['units']:7d} {a['qasr_units']:7d} {a['masc_units']:7d} "
          f"{60*h/len(un):7.1f} {a['mean_lev']:9.3f} {a['median_lev']:8.3f} {a['pct_ge_0.8']:6.1f}%")
import itertools
S={k:{r["unit"] for r in v} for k,v in final.items()}
print("\ndisjointness (unit = qasr recording_id / masc video_id):")
tr=S["train"]
for s_ in ("val","test"):
    ov=tr & set(v3units[s_].keys())
    print(f"  random train vs v3 {s_}: {len(ov)} shared units")
uids_tr={r["uid"] for r in final["train"]}
import pyarrow.parquet as _pq
for s_ in ("val","test"):
    du=_pq.read_table(H/f"v3_data/parquet/{s_}.parquet",columns=["uid"]).to_pydict()["uid"]
    print(f"  random train vs v3 {s_}: {len(uids_tr & set(map(str,du)))} shared CLIPS")
print("\nv3 for reference:")
for s in ("train","val","test"):
    tot=sum(v3units[s].values()); n=len(v3units[s])
    print(f"  {s:6s} {n:5d} units {tot:7.2f}h  {60*tot/n:5.1f} min/unit")
for s in ("train",):
    json.dump([{"uid":r["uid"],"unit":r["unit"],"src":r["src"],"dur":r["dur"],"lev":r["lev"]} for r in final[s]],
              (OUT/f"{s}_manifest.json").open("w"))
json.dump({"matched":rs,"seed":4242,"unit":"qasr=recording_id, masc=video_id",
           "method":"duration-matched random draw against v3's per-unit size distribution"},
          (OUT/"config.json").open("w"),indent=2)
print(f"\n-> {OUT}")
