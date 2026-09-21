# Promoted from the research scratch dir (~/registry.py) on the 5090 node.
# Rebuild RUN_REGISTRY.json: config + data + results for every run.
# See docs/HANDOFF.md for what it produced.
"""One file tracking every run: config, data, results, artifact locations."""
import json,re,collections
from pathlib import Path
import pyarrow.parquet as pq
H=Path.home(); REG={}
def pqstat(f):
    f=Path(f)
    if not f.exists(): return None
    d=pq.read_table(f,columns=["source","duration","speaker_key"]).to_pydict()
    h=collections.Counter(); sp=collections.defaultdict(set)
    for a,b,c in zip(d["source"],d["duration"],d["speaker_key"]): h[a]+=b or 0; sp[a].add(str(c))
    return {"path":str(f),"clips":len(d["source"]),"hours":round(sum(h.values())/3600,2),
            "by_source":{k:{"hours":round(h[k]/3600,2),"speaker_keys":len(sp[k])} for k in h}}
def summaries(log):
    log=Path(log); out=[]
    if not log.exists(): return out
    for line in log.read_text(encoding="utf-8",errors="ignore").splitlines():
        m=re.search(r"SUMMARY (\S+): (\{.*\})",line)
        if not m: continue
        d=json.loads(m.group(2)); d.pop("train_loss",None)
        out.append({"tag":m.group(1),"hours":d.get("cumulative_hours_trained"),
                    "val_wer":round(100*(d.get("val") or {}).get("wer",0),2) or None,
                    "test_wer":round(100*(d.get("test") or {}).get("wer",0),2) or None})
    return out
RUNS={
 "v3_300h_step_guided":{"dir":"v3_run","train":"v3_data/parquet/train.parquet","val":"v3_data/parquet/val.parquet",
   "test":"v3_data/parquet/test.parquet","notes":"300h ascending by 0.3*text+0.7*audio DID score, 6x50h chunks x2 epochs, lr 1e-4"},
 "v3_best50h_c6":{"dir":"v3_best50h_run","train":"v3_data/parquet/train.parquet","val":"v3_data/parquet/val.parquet",
   "test":"v3_data/parquet/test.parquet","notes":"highest-scoring 50h only, 2 epochs"},
 "v3_continuation":{"dir":"v3_cont_run","train":"v3_data/parquet/train.parquet","val":"v3_data/parquet/val.parquet",
   "test":"v3_data/parquet/test.parquet","notes":"continued from 300h best ckpt on c5+c6 then c6; CONTAMINATED (concurrent run, skipped stage)"},
 "sim_ascending":{"dir":"runs/sim_ascending","train":"sim_sets/train_ascending.parquet","val":"sim_sets/val.parquet",
   "test":"sim_sets/test.parquet","notes":"50h nearest Layla+Omni centered-embedding centroid, ascending per-utterance sim, shuffle OFF, lr 2e-5"},
 "sim_shuffled":{"dir":"runs/sim_shuffled","train":"sim_sets/train_shuffled.parquet","val":"sim_sets/val.parquet",
   "test":"sim_sets/test.parquet","notes":"same 50h, randomly shuffled, lr 2e-5"},
 "random_50h":{"dir":"runs/random_50h","train":"sim_sets/random_train.parquet","val":"sim_sets/random_val.parquet",
   "test":"sim_sets/test.parquet","notes":"random 50h + random 5h val, disjoint from shared test, lr 2e-5"},
}
for name,r in RUNS.items():
    d=H/r["dir"]
    REG[name]={"notes":r["notes"],"run_dir":str(d),"exists":d.exists(),
        "checkpoints":sorted(p.name for p in (d/"checkpoints").iterdir()) if (d/"checkpoints").exists() else [],
        "data":{k:pqstat(H/r[k]) for k in ("train","val","test")},
        "stages":summaries(d/"train.log")}
mx=H/"eval_sweep/matrix.json"
REG["_eval_matrix"]=json.loads(mx.read_text()) if mx.exists() else None
REG["_lora_config"]={"base":"openai/whisper-medium","r":32,"alpha":32,"dropout":0.05,
    "targets":"q,k,v,o,fc1,fc2","batch":8,"scheduler":"OneCycleLR","warmup_ratio":0.1,
    "precision":"bf16","grad_checkpointing":True}
REG["_test_sets"]={n:pqstat(H/p) for n,p in {
    "sim_newtest":"sim_sets/test.parquet","v3_old_test":"v3_data/parquet/test.parquet",
    "casa_pal":"nat_resplit/parquet/casa_pal_test.parquet","casa_jor":"nat_resplit/parquet/casa_jor_test.parquet",
    "layla_told":"layla_told/layla_told_test.parquet","omni_all_speakers":"sim_sets/omni_test_all_speakers.parquet"}.items()}
REG["_r2"]="backup/transfer/levantine_similarity_v1/"
out=H/"RUN_REGISTRY.json"; out.write_text(json.dumps(REG,ensure_ascii=False,indent=2))
print(f"wrote {out}")
for n,v in REG.items():
    if n.startswith("_"): continue
    st=v["stages"][-1] if v["stages"] else {}
    tr=(v["data"]["train"] or {}).get("hours")
    print(f"  {n:22s} exists={v['exists']!s:5s} train={tr}h  ckpts={len(v['checkpoints'])}  last val={st.get('val_wer')} test={st.get('test_wer')}")
