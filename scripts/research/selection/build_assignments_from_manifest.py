"""Assignments for the random-matched 300h train block. val/test reuse v3's own sets."""
import json, collections
from pathlib import Path
H=Path.home()
m=json.load((H/"random300_matched/train_manifest.json").open())
units=collections.OrderedDict()
for r in m:
    u=r["unit"]
    if u.startswith("MASC:"): units[("masc_c",u[5:])]=1
    else: units[("qasr",u)]=1
asg=[{"source":s,"speaker_key":k,"split":"train"} for s,k in units]
out=H/"random300_matched/assignments.json"
out.write_text(json.dumps({"assignments":asg}))
c=collections.Counter(s for s,_ in units)
print(f"{len(asg)} units -> {out}   qasr {c['qasr']}  masc_c {c['masc_c']}")
print(f"manifest: {len(m):,} clips, {sum(r['dur'] for r in m)/3600:.2f}h, "
      f"mean composite {sum(r['lev'] for r in m)/len(m):.3f}")
