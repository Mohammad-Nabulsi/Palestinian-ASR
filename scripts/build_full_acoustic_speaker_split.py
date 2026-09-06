#!/usr/bin/env python3
"""Create full-corpus speaker-disjoint test/val/train blocks from two scores."""
from __future__ import annotations
import argparse, json
from collections import defaultdict
from pathlib import Path

def rows(paths):
    for path in paths:
        with path.open(encoding="utf-8") as f:
            for line in f:
                if line.strip(): yield json.loads(line)

def key(r):
    source = r.get("source")
    if source == "masc": source = "masc_c"
    if r.get("speaker_key"): return source, str(r["speaker_key"])
    uid = str(r.get("uid", ""))
    if source == "qasr" and uid.startswith("qasr:"): return source, uid.split(":")[1]
    if source == "masc_c": return source, uid.removeprefix("masc_c:").split(":")[0]
    return None

def main():
    p=argparse.ArgumentParser(); p.add_argument("--text-scan", type=Path, nargs="+", required=True); p.add_argument("--audio-root", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True); a=p.parse_args()
    text=defaultdict(lambda:[0.,0]); audio=defaultdict(lambda:[0.,0,0.])
    for r in rows(a.text_scan):
        k=key(r)
        if k and r.get("status", "ok") == "ok": text[k][0]+=(r.get("label_scores") or {}).get("LEV", 0.); text[k][1]+=1
    for fp in sorted(a.audio_root.glob("*/row_probabilities.jsonl")):
        for r in rows([fp]):
            k=key(r)
            if k and r.get("status") == "ok": audio[k][0]+=(r.get("label_scores") or {}).get("Levantine", 0.); audio[k][1]+=1; audio[k][2]+=float(r.get("duration_sec") or 0)/3600
    speakers=[]
    for k in set(text)|set(audio):
        t, au=text[k], audio[k]
        # No acoustic result means no usable audio/training duration, never selected.
        if not au[1] or not au[2]: continue
        speakers.append({"source":k[0],"speaker_key":k[1],"text_mean":t[0]/t[1] if t[1] else 0.,"audio_mean":au[0]/au[1],"hours":au[2],"score":.3*(t[0]/t[1] if t[1] else 0.)+.7*au[0]/au[1]})
    speakers.sort(key=lambda s:(-s["score"],-s["hours"],s["source"],s["speaker_key"]))
    limits=(("test",8.),("val",8.),("train",200.)); chosen=[]; start=0
    for split, limit in limits:
        hours=0.
        while start<len(speakers) and hours<limit:
            s=speakers[start]; start+=1; hours+=s["hours"]; chosen.append({**s,"split":split})
    a.out_dir.mkdir(parents=True,exist_ok=True)
    (a.out_dir/"speaker_assignments.json").write_text(json.dumps({"meta":{"text_weight":.3,"audio_weight":.7,"block_order":[x[0] for x in limits],"all_samples":True},"assignments":chosen},indent=2,ensure_ascii=False)+"\n")
    cumulative=0.; rank=[]
    for s in chosen:
        if s["split"]=="train": cumulative+=s["hours"]; rank.append({"source":s["source"],"speaker_key":s["speaker_key"],"cum_hours":cumulative})
    (a.out_dir/"train_full_rank.json").write_text(json.dumps(rank,indent=2,ensure_ascii=False)+"\n")
    print(json.dumps({x:round(sum(s['hours'] for s in chosen if s['split']==x),3) for x,_ in limits},indent=2))
if __name__ == "__main__": main()
