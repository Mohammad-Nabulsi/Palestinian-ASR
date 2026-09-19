#!/usr/bin/env python3
"""Create full-corpus speaker-disjoint test/val/train blocks from two scores."""
from __future__ import annotations
import argparse, json, re
from collections import defaultdict
from pathlib import Path

# The audio scan writes a `speaker_key` field and short uids; the text scan predates
# both and writes only a LONG uid that embeds the id, e.g.
#   "...__clean.parquet:uid=qasr:0010E262-0FE5-...:..._utt_1_align"
#   "...__clean.parquet:video_id=OGqz9G-JO0E"
# Matching only the short form silently dropped every QASR text row and turned every
# MASC text row into a filename-keyed speaker that joined to nothing -- zeroing the
# text half of the score with no error. Same patterns as pipeline/speaker_scoring.py.
QASR_LONG = re.compile(r"uid=qasr:([0-9A-Fa-f-]+):")
QASR_SHORT = re.compile(r"^qasr:([0-9A-Fa-f-]+):")
MASC_LONG = re.compile(r"video_id=([^:]+)$")

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
    if source == "qasr":
        m = QASR_LONG.search(uid) or QASR_SHORT.match(uid)
        return (source, m.group(1)) if m else None
    if source == "masc_c":
        m = MASC_LONG.search(uid)
        if m: return source, m.group(1)
        return source, uid.removeprefix("masc_c:").split(":")[0]
    return None

def lev(r, label):
    """Levantine probability, from either scan's format.

    The audio scan writes {"label_scores": {"Levantine": p, ...}}; the text scan
    writes {"predictions": [{"label": "LEV", "score": p}, ...]} and carries no
    label_scores at all, so reading only label_scores scored every text row 0.0.
    """
    scores = r.get("label_scores")
    if isinstance(scores, dict): return scores.get(label, 0.)
    for p in r.get("predictions") or []:
        if p.get("label") == label: return float(p.get("score") or 0.)
    return 0.

def main():
    p=argparse.ArgumentParser(); p.add_argument("--text-scan", type=Path, nargs="+", required=True); p.add_argument("--audio-root", type=Path, required=True); p.add_argument("--out-dir", type=Path, required=True); a=p.parse_args()
    text=defaultdict(lambda:[0.,0]); audio=defaultdict(lambda:[0.,0,0.])
    for r in rows(a.text_scan):
        k=key(r)
        if k and r.get("status", "ok") == "ok": text[k][0]+=lev(r, "LEV"); text[k][1]+=1
    for fp in sorted(a.audio_root.glob("*/row_probabilities.jsonl")):
        for r in rows([fp]):
            k=key(r)
            if k and r.get("status") == "ok": audio[k][0]+=lev(r, "Levantine"); audio[k][1]+=1; audio[k][2]+=float(r.get("duration_sec") or 0)/3600
    # A join failure here is invisible in the output -- the score just silently
    # collapses to 0.7*audio and still produces a plausible-looking ranking. That
    # shipped once. Fail loudly instead: the two scans cover the same corpus, so
    # nearly every audio speaker must carry a text score too.
    overlap = len(set(text) & set(audio))
    if overlap < 0.5 * len(audio):
        raise SystemExit(
            f"text/audio speaker join failed: only {overlap} of {len(audio)} audio speakers "
            f"matched a text speaker ({len(text)} text speakers seen). Check uid formats "
            f"in --text-scan against key()."
        )
    print(f"joined {overlap} of {len(audio)} audio speakers to text scores", flush=True)
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
