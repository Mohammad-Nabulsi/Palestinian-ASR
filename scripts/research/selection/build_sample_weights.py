# Promoted from the research scratch dir (~/mkweights.py) on the 5090 node.
# uid -> 0.3*text + 0.7*acoustic composite score map for --sample-weights.
# See docs/HANDOFF.md for what it produced.
"""uid -> 0.3*text + 0.7*acoustic Levantine score, for --sample-weights."""
import json,re,statistics
from pathlib import Path
H=Path.home(); T=H/"scans/text_v1/dialect_id_scans"
def lev(r):
    ls=r.get("label_scores") or {}
    for k in ("Levantine","LEV","lev"):
        if k in ls: return float(ls[k])
    return None
txt={}
for sub,kind in (("text_dialect_scan_marbertv2_written_clean_qasr_only","qasr"),
                 ("text_dialect_scan_marbertv2_written_clean_masc_c_only","masc")):
    with (T/sub/"row_probabilities.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line); v=lev(r)
            if v is None: continue
            u=str(r.get("uid",""))
            if kind=="qasr":
                i=u.find("qasr:")
                if i>=0: txt[u[i:]]=v
            else:
                m=re.search(r"video_id=([A-Za-z0-9_-]+)",u)
                if m: txt["masc_c:"+m.group(1)]=v
out={}; mascvid={}
for sub in ("qasr_lev","qasr_non_lev","masc_lev","masc_non_lev"):
    with (H/f"scans/audio_v2/acoustic_full_scan/{sub}/row_probabilities.jsonl").open(encoding="utf-8") as f:
        for line in f:
            if not line.strip(): continue
            r=json.loads(line)
            if r.get("status")!="ok": continue
            u=str(r["uid"]); a=lev(r) or 0.0
            if r["source"]=="qasr":
                i=u.find("qasr:"); k=u[i:] if i>=0 else u
                sc=round(0.3*txt.get(k,0.0)+0.7*a,4)
                out[u]=sc; out[k]=sc
            else:
                vid=str(r.get("speaker_key")); k="masc_c:"+vid
                sc=round(0.3*txt.get(k,0.0)+0.7*a,4)
                out[vid+"	"+str(r.get("text") or "")]=sc
                mascvid.setdefault(vid,[]).append(sc)
import statistics as _st
for _v,_l in mascvid.items(): out[_v]=round(_st.mean(_l),4)
p=H/"sample_weights_composite.json"; p.write_text(json.dumps(out))
v=list(out.values())
print(f"{len(out):,} uids -> {p}")
print(f"score: mean {statistics.mean(v):.3f} median {statistics.median(v):.3f} "
      f"min {min(v):.3f} max {max(v):.3f}")
for f_,g in ((0.25,1.0),(0.25,2.0),(0.10,1.0)):
    w=[f_+(1-f_)*x**g for x in v]; m=statistics.mean(w)
    ws=sorted(w)
    print(f"  floor={f_} gamma={g}: weight mean {m:.3f} -> normalized p10 {ws[len(ws)//10]/m:.2f} "
          f"p50 {ws[len(ws)//2]/m:.2f} p90 {ws[9*len(ws)//10]/m:.2f} max {ws[-1]/m:.2f}")
