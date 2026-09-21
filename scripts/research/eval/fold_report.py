# Promoted from the research scratch dir (~/fold_report.py) on the 5090 node.
# WER/CER matrix, as-scored and with ta-marbuta folded.
# See docs/HANDOFF.md for what it produced.
"""Rebuild the WER matrix from the sweep, as-scored and with ta-marbuta folded.

MASC/QASR references carry zero ta marbuta, so a model that writes it correctly
is penalised for orthography rather than recognition; folding ة->ه and ى->ي
separates the two.
"""
import json, re
from pathlib import Path
import jiwer
D=Path.home()/"eval_sweep"
MODELS=["base","cohere","v3_300h","sim_ascending","sim_shuffled","random_50h"]
SETS=["newtest","casa_pal","casa_jor","layla_told","omni_all"]
def fold(s): return s.replace("ة","ه").replace("ى","ي")
def wer(rs,hs):
    rs2,hs2=zip(*[(r,h) for r,h in zip(rs,hs) if r.strip()])
    return 100*jiwer.wer(list(rs2),list(hs2)), 100*jiwer.cer(list(rs2),list(hs2))
res={}
for m in MODELS:
    for s in SETS:
        f=D/f"{m}__{s}.json"
        if not f.exists(): continue
        d=json.loads(f.read_text(encoding="utf-8"))
        pu=d.get("per_utterance") or []
        rs=[x.get("reference",x.get("ref","")) for x in pu]
        hs=[x.get("hypothesis",x.get("hyp","")) for x in pu]
        if rs and hs:
            raw=wer(rs,hs); fol=wer([fold(r) for r in rs],[fold(h) for h in hs])
        else:
            raw=(100*d["wer"],100*d["cer"]); fol=(float("nan"),float("nan"))
        res[(m,s)]=(raw,fol,len(rs) or d.get("n_scored"))
hdr=f"{'test set':14s} " + " ".join(f"{m[:13]:>15s}" for m in MODELS)
print("=== WER %, as-scored / ta-marbuta folded ===")
print(hdr); print("-"*len(hdr))
for s in SETS:
    cells=[]
    for m in MODELS:
        if (m,s) not in res: cells.append(f"{'-':>15s}"); continue
        (rw,_),(fw,_),_=res[(m,s)]
        cells.append(f"{rw:6.2f}/{fw:6.2f}".rjust(15))
    n=next((res[(m,s)][2] for m in MODELS if (m,s) in res),0)
    print(f"{s+f' ({n})':14s} "+" ".join(cells))
print("\n=== CER %, as-scored / folded ===")
print(hdr); print("-"*len(hdr))
for s in SETS:
    cells=[]
    for m in MODELS:
        if (m,s) not in res: cells.append(f"{'-':>15s}"); continue
        (_,rc),(_,fc),_=res[(m,s)]
        cells.append(f"{rc:6.2f}/{fc:6.2f}".rjust(15))
    print(f"{s:14s} "+" ".join(cells))
print("\n=== deltas vs base (folded WER, negative = better) ===")
for s in SETS:
    if ("base",s) not in res: continue
    b=res[("base",s)][1][0]
    row=[f"{m}: {res[(m,s)][1][0]-b:+6.2f}" for m in MODELS[1:] if (m,s) in res]
    print(f"  {s:12s} base {b:6.2f} | "+"  ".join(row))
json.dump({f"{m}|{s}":{"wer":res[(m,s)][0][0],"cer":res[(m,s)][0][1],
                       "wer_folded":res[(m,s)][1][0],"cer_folded":res[(m,s)][1][1],
                       "n":res[(m,s)][2]} for m,s in res},
          (D/"matrix.json").open("w"),indent=2)
print(f"\nsaved {D/'matrix.json'}")
