# Promoted from the research scratch dir (~/spkmap.py) on the 5090 node.
# Stream every QASR shard on R2 to extract uid -> real speaker_id (audio discarded).
# See docs/HANDOFF.md for what it produced.
"""Stream every QASR shard, keep only uid->speaker_id, discard the audio.

speaker_id (the diarised person) lives only in the R2 shards; our corpus
parquet carries recording_id under `speaker_key`. 116 GiB of audio for three
string columns, so each shard is fetched, read, and deleted in turn.
"""
import subprocess, os, json, time
from pathlib import Path
import pyarrow.parquet as pq
W=Path.home()/"spkmap"; W.mkdir(exist_ok=True)
RC=str(Path.home()/"bin/rclone"); BASE="R2:backup/transfer/curated_corpus/train/qasr"
def rc(a): return subprocess.run([RC,*a],check=True,text=True,capture_output=True).stdout
out=(W/"uid_speaker.jsonl").open("w",encoding="utf-8")
n=0; t0=time.time()
for leaf in ("lev","non_lev"):
    names=sorted(x for x in rc(["lsf",f"{BASE}/{leaf}"]).splitlines() if x.endswith(".parquet.zst"))
    print(f"{leaf}: {len(names)} shards",flush=True)
    for name in names:
        z=W/name; p=W/"s.parquet"
        if not z.exists(): rc(["copy",f"{BASE}/{leaf}/{name}",str(W)])
        subprocess.run(["unzstd","-f","-q","-o",str(p),str(z)],check=True); z.unlink()
        sch=pq.ParquetFile(p).schema_arrow
        cols=[c for c in ("uid","speaker_id","recording_id","duration") if c in sch.names]
        d=pq.read_table(p,columns=cols).to_pydict(); p.unlink()
        for i in range(len(d["uid"])):
            out.write(json.dumps({"uid":str(d["uid"][i]),
                                  "speaker_id":str(d.get("speaker_id",[None]*len(d["uid"]))[i]),
                                  "recording_id":str(d.get("recording_id",[None]*len(d["uid"]))[i]),
                                  "leaf":leaf})+"\n")
        n+=len(d["uid"])
        print(f"[{time.strftime('%H:%M:%S')}] {leaf}/{name} +{len(d['uid'])} rows (total {n:,}, {time.time()-t0:.0f}s)",flush=True)
out.close(); print(f"DONE {n:,} rows -> {W/'uid_speaker.jsonl'}",flush=True)
