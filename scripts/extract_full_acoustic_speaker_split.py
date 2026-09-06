#!/usr/bin/env python3
"""Extract every sample of selected speakers into train/val/test parquet files.

QASR's "audio" struct holds raw headerless PCM16 (real rate in a sibling
`sampling_rate` column), not self-describing WAV/FLAC/OGG like MASC's -- the
same gotcha that broke full_acoustic_scan.py's decode() (see that file's
comment). The training loader downstream (ParquetAudioTextDataset in
scripts/train_whisper_medium_lora_progressive.py) calls decode_audio_cell()
with NO sampling-rate hint, so it needs self-describing bytes -- QASR's raw
PCM16 is re-encoded to WAV here so both sources land in a uniform "audio"
column. A cheap decode+validate pass also drops the rare genuinely-corrupt
row (observed empirically in curated_corpus) rather than writing garbage into
the training set.
"""
from __future__ import annotations
import argparse,io,json,subprocess
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, soundfile as sf

SCHEMA=pa.schema([("uid",pa.string()),("audio",pa.binary()),("text",pa.string()),("source",pa.string()),("speaker_key",pa.string()),("duration",pa.float64())])
def rc(args): return subprocess.run(["rclone",*args],check=True,text=True,capture_output=True).stdout

def to_wav_bytes(cell, sampling_rate=None):
    """Return (wav_bytes, decoded_seconds) or (None, None) if unusable."""
    raw = cell["bytes"] if isinstance(cell, dict) else bytes(cell)
    try:
        if sampling_rate is not None and raw[:4] not in (b"RIFF", b"fLaC", b"OggS"):
            data = np.frombuffer(raw, dtype="<i2")
            sr = int(sampling_rate)
        else:
            data, sr = sf.read(io.BytesIO(raw), dtype="int16", always_2d=False)
            if data.ndim > 1: data = data.mean(axis=1).astype("int16")
        if len(data) == 0: return None, None
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue(), len(data) / sr
    except Exception:
        return None, None

def main():
 p=argparse.ArgumentParser();p.add_argument('--assignments',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--work-dir',type=Path,required=True);p.add_argument('--remote',default='R2:backup/transfer/curated_corpus/train');a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True);a.work_dir.mkdir(parents=True,exist_ok=True)
 assignments=json.loads(a.assignments.read_text())["assignments"]; route={(x['source'],str(x['speaker_key'])):x['split'] for x in assignments}; writers={}
 def writer(split):
  if split not in writers: writers[split]=pq.ParquetWriter(a.out_dir/f'{split}.parquet',SCHEMA)
  return writers[split]
 n_corrupt = 0
 for source in ('masc','qasr'):
  outsource='masc_c' if source=='masc' else 'qasr'; ident='video_id' if source=='masc' else 'recording_id'
  for leaf in ('lev','non_lev'):
   remote=f'{a.remote}/{source}/{leaf}'
   for name in sorted(x for x in rc(['lsf',remote]).splitlines() if x.endswith('.parquet.zst')):
    z,pqfile=a.work_dir/'shard.parquet.zst',a.work_dir/'shard.parquet';rc(['copy',f'{remote}/{name}',str(a.work_dir)]);(a.work_dir/name).replace(z);subprocess.run(['unzstd','-f','-o',str(pqfile),str(z)],check=True);z.unlink()
    schema=pq.ParquetFile(pqfile).schema_arrow; text=next(x for x in ('text','normalized_transcript','transcript','manual_normalized_transcript') if x in schema.names); uid='uid' if 'uid' in schema.names else ident
    has_sr = 'sampling_rate' in schema.names
    cols=[uid,ident,text,'duration','audio']+(['sampling_rate'] if has_sr else [])
    d=pq.read_table(pqfile,columns=cols).to_pydict(); buckets={}
    for i,k in enumerate(d[ident]):
     split=route.get((outsource,str(k)))
     if not split: continue
     wav_bytes, decoded_sec = to_wav_bytes(d['audio'][i], d['sampling_rate'][i] if has_sr else None)
     if wav_bytes is None:
      n_corrupt += 1; continue
     buckets.setdefault(split,{n:[] for n in SCHEMA.names}); b=buckets[split]
     b['uid'].append(str(d[uid][i])); b['audio'].append(wav_bytes); b['text'].append(d[text][i] or '')
     b['source'].append(outsource); b['speaker_key'].append(str(k)); b['duration'].append(float(d['duration'][i] or decoded_sec or 0))
    for split,b in buckets.items(): writer(split).write_table(pa.table(b,schema=SCHEMA))
    pqfile.unlink();print(f'{source}/{leaf}/{name} (corrupt so far: {n_corrupt})',flush=True)
 for w in writers.values():w.close()
 print(f'DONE. total corrupt/unusable rows skipped: {n_corrupt}', flush=True)
if __name__=='__main__':main()
