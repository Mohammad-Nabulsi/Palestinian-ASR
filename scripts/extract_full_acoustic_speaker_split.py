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
from collections import Counter
from pathlib import Path
import numpy as np, pyarrow as pa, pyarrow.parquet as pq, soundfile as sf

SCHEMA=pa.schema([("uid",pa.string()),("audio",pa.binary()),("text",pa.string()),("source",pa.string()),("speaker_key",pa.string()),("duration",pa.float64())])
# int16 peak at or below this counts as silence, not quiet speech. Real clips in
# this corpus peak in the thousands; the broken ones peaked at 0 or 1.
SILENCE_PEAK=2
def rc(args): return subprocess.run(["rclone",*args],check=True,text=True,capture_output=True).stdout

def to_wav_bytes(cell, sampling_rate=None):
    """Return (wav_bytes, decoded_seconds, None), or (None, None, reason) if unusable."""
    try:
        raw = cell["bytes"] if isinstance(cell, dict) else bytes(cell)
        if not raw: return None, None, "empty"
        if sampling_rate is not None and raw[:4] not in (b"RIFF", b"fLaC", b"OggS"):
            data = (np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0)
            sr = int(sampling_rate)
        else:
            # MASC's WAV is 32-bit IEEE float (audio_format=3, with a `fact`
            # chunk) -- asking soundfile to decode straight to dtype="int16"
            # on this layout silently returns near-all-zero samples instead
            # of scaling floats into int16 range (confirmed empirically: the
            # same bytes read at their native float32 dtype peak at ~0.93,
            # i.e. real speech, but come back as all-0 when dtype="int16" is
            # requested directly). Always decode at the file's native float
            # dtype and scale to int16 ourselves, exactly like this repo's
            # other working decoder (full_acoustic_scan.py's decode()).
            data, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
            if data.ndim > 1: data = data.mean(axis=1)
        if len(data) == 0: return None, None, "empty"
        data = np.clip(data * 32768.0, -32768, 32767).astype("int16")
        # Silence guard. Every MASC-C row of full_acoustic_split_v2 was written as
        # digital silence because the decode above requested dtype="int16" on
        # MASC's float WAVs and got zeros back, and nothing downstream noticed:
        # the schema, the row counts and the durations were all still correct, so
        # 38% of a 200h training set and 66% of its test set were silence paired
        # with real transcripts. Peak amplitude is the one cheap check that would
        # have caught it, so it runs on every row now and the row is dropped
        # rather than written.
        if int(np.abs(data).max()) <= SILENCE_PEAK: return None, None, "silent"
        buf = io.BytesIO()
        sf.write(buf, data, sr, format="WAV", subtype="PCM_16")
        return buf.getvalue(), len(data) / sr, None
    except Exception:
        return None, None, "decode"

def main():
 p=argparse.ArgumentParser();p.add_argument('--assignments',type=Path,required=True);p.add_argument('--out-dir',type=Path,required=True);p.add_argument('--work-dir',type=Path,required=True);p.add_argument('--remote',default='R2:backup/transfer/curated_corpus/train');a=p.parse_args();a.out_dir.mkdir(parents=True,exist_ok=True);a.work_dir.mkdir(parents=True,exist_ok=True)
 assignments=json.loads(a.assignments.read_text())["assignments"]; route={(x['source'],str(x['speaker_key'])):x['split'] for x in assignments}; writers={}
 def writer(split):
  if split not in writers: writers[split]=pq.ParquetWriter(a.out_dir/f'{split}.parquet',SCHEMA)
  return writers[split]
 dropped=Counter(); kept_per_source=Counter()
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
     wav_bytes, decoded_sec, why = to_wav_bytes(d['audio'][i], d['sampling_rate'][i] if has_sr else None)
     if wav_bytes is None:
      dropped[f'{outsource}:{why}'] += 1; continue
     kept_per_source[outsource] += 1
     buckets.setdefault(split,{n:[] for n in SCHEMA.names}); b=buckets[split]
     b['uid'].append(str(d[uid][i])); b['audio'].append(wav_bytes); b['text'].append(d[text][i] or '')
     b['source'].append(outsource); b['speaker_key'].append(str(k)); b['duration'].append(float(d['duration'][i] or decoded_sec or 0))
    for split,b in buckets.items(): writer(split).write_table(pa.table(b,schema=SCHEMA))
    pqfile.unlink();print(f'{source}/{leaf}/{name} kept={dict(kept_per_source)} dropped={dict(dropped)}',flush=True)
 for w in writers.values():w.close()
 print(f'DONE. kept={dict(kept_per_source)} dropped={dict(dropped)}', flush=True)
 # A whole source coming back silent is the v2 failure mode. It is not a warning,
 # it is a corrupt dataset, so refuse to leave one behind.
 for src in ('masc_c','qasr'):
  silent, kept = dropped.get(f'{src}:silent',0), kept_per_source.get(src,0)
  if silent and silent > kept:
   raise SystemExit(f'ABORT: {src} produced {silent} silent rows vs {kept} usable ones -- decode is broken for this source, not the data')
if __name__=='__main__':main()
