#!/usr/bin/env python
"""Build a ~10h stage-1 pretraining view sampled from QASR Levantine (data_curated_levant_binary_v1
train/qasr/lev), harmonized to the same schema unified.whisper.run.ipynb / pal_s1e1of2_run.py
expect (matches data_stage1_v1/{jor,omni,omni_jor} from build_pal_stage_datasets.py):

    uid: string
    audio: struct<bytes: binary (WAV PCM16 @16kHz mono), path: string>
    duration: double
    text: string
    mix_source: string  # "qasr_lev"

qasr/lev's curated rows already store raw PCM16LE @16kHz mono audio (a bare `audio: binary`
column + a `sampling_rate` column, not a WAV container and not the struct schema) with
manual_normalized_transcript 100% populated -- no resample needed, just a WAV-container
wrap + a schema rename, unlike casa/omni which needed resample + struct wrapping.

Row-level seeded shuffle (seed=42, same seed as build_pal_stage_datasets.py) over the full
21,824-row / 26.47h train split, taking rows in shuffled order until cumulative duration
reaches the 10h target.
"""
import io, glob
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

CURATED_LEAF = Path("/workspace/asr/Palestinian-ASR/data_curated_levant_binary_v1/train/qasr/lev")
OUT = Path("/root/Palestinian-ASR/data_stage1_v1/qasr_lev_10h/train")
SEED = 42
TARGET_HOURS = 10.0
TARGET_SR = 16000

SCHEMA = pa.schema([
    ("uid", pa.string()),
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("duration", pa.float64()),
    ("text", pa.string()),
    ("mix_source", pa.string()),
])


def _pcm16_to_wav(raw: bytes) -> bytes:
    arr = np.frombuffer(raw, dtype="<i2")
    buf = io.BytesIO()
    sf.write(buf, arr, TARGET_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def main():
    files = sorted(glob.glob(str(CURATED_LEAF / "*.parquet")))
    print(f"[qasr_lev_10h] {len(files)} source shards under {CURATED_LEAF}")

    rows = []
    for f in files:
        stem = Path(f).stem
        tbl = pq.read_table(f, columns=["audio", "sampling_rate", "duration",
                                        "manual_normalized_transcript"])
        srs = tbl.column("sampling_rate").to_pylist()
        assert all(s == TARGET_SR for s in srs), f"{f} has non-16kHz rows"
        audio_col = tbl.column("audio").to_pylist()
        dur_col = tbl.column("duration").to_pylist()
        text_col = tbl.column("manual_normalized_transcript").to_pylist()
        for i in range(tbl.num_rows):
            text = (text_col[i] or "").strip()
            if not text or dur_col[i] is None:
                continue
            rows.append({
                "uid": f"qasr_lev__train__{stem}__{i:06d}",
                "_raw_audio": audio_col[i],
                "duration": float(dur_col[i]),
                "text": text,
                "mix_source": "qasr_lev",
            })
    print(f"[qasr_lev_10h] {len(rows)} candidate rows, "
          f"{sum(r['duration'] for r in rows) / 3600.0:.2f}h available")

    idx = np.arange(len(rows))
    np.random.default_rng(SEED).shuffle(idx)
    target_s = TARGET_HOURS * 3600.0
    picked, cum = [], 0.0
    for i in idx:
        if cum >= target_s:
            break
        picked.append(rows[i])
        cum += rows[i]["duration"]
    print(f"[qasr_lev_10h] picked {len(picked)} rows, {cum / 3600.0:.4f}h "
          f"(target {TARGET_HOURS}h)")

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "qasr_lev__10h.parquet"
    writer = pq.ParquetWriter(path, SCHEMA, compression="snappy")
    batch = 128
    try:
        for s in range(0, len(picked), batch):
            chunk = picked[s:s + batch]
            audio = [{"bytes": _pcm16_to_wav(r["_raw_audio"]), "path": None} for r in chunk]
            writer.write_table(pa.Table.from_pydict({
                "uid": [r["uid"] for r in chunk],
                "audio": audio,
                "duration": [r["duration"] for r in chunk],
                "text": [r["text"] for r in chunk],
                "mix_source": [r["mix_source"] for r in chunk],
            }, schema=SCHEMA))
            print(f"  {min(s + batch, len(picked))}/{len(picked)}", end="\r", flush=True)
    finally:
        writer.close()
    print(f"\n[qasr_lev_10h] wrote {len(picked)} rows, {cum/3600.0:.4f}h -> {path}")


if __name__ == "__main__":
    main()
