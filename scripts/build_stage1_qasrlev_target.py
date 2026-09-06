#!/usr/bin/env python
"""Build the data views for the qasr_lev-as-target study:

  data_qasrlev_v1/{train,val,test}   -- qasr_lev's own native splits, harmonized to the
      same schema as data_pal_v1 (this becomes the new SPLITS: qasrTrain/qasrVal/qasrTest).
      train=21,824 rows/26.47h, val=4,676 rows/5.71h, test=4,678 rows/5.65h.

  data_stage1_v1/casa_pal_all/train  -- casa/pal ALL native splits (train+val+test) combined
      into one pretraining shard, 1,328 rows/1.97h -- mirrors how casa_jor/omni/layla were
      already built as "all splits combined" stage-1 corpora.

  data_stage1_v1/qasrlev_target_mix/train -- the STAGE-1 mix: qasr_lev train (only) +
      casa_pal_all + casa_jor (copied from data_stage1_v1/jor) + omni (copied) + layla
      (copied). qasr_lev's own val/test are held out of this mix entirely.

qasr/lev curated rows are raw PCM16LE @16kHz mono (audio: binary + sampling_rate, no WAV
container) -- same harmonization as build_stage1_qasr_lev_10h.py. casa/pal rows are already
audio: struct<bytes: WAV, path> -- same harmonization as build_pal_stage_datasets.py.
"""
import io, glob, shutil
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import soxr

CURATED = Path("/workspace/asr/Palestinian-ASR/data_curated_levant_binary_v1")
QASRLEV_OUT = Path("/root/Palestinian-ASR/data_qasrlev_v1")
S1_OUT = Path("/root/Palestinian-ASR/data_stage1_v1")
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


def _wav_to_wav16k_mono(raw: bytes) -> bytes:
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        wav = soxr.resample(wav, sr, TARGET_SR)
    buf = io.BytesIO()
    sf.write(buf, wav, TARGET_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def load_qasrlev_split(split: str):
    files = sorted(glob.glob(str(CURATED / split / "qasr" / "lev" / "*.parquet")))
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
                "uid": f"qasr_lev__{split}__{stem}__{i:06d}",
                "_raw": audio_col[i], "_kind": "pcm",
                "duration": float(dur_col[i]), "text": text, "mix_source": "qasr_lev",
            })
    return rows


def load_casapal_split(split: str):
    files = sorted(glob.glob(str(CURATED / split / "casa" / "pal" / "*.parquet")))
    rows = []
    for f in files:
        stem = Path(f).stem
        tbl = pq.read_table(f, columns=["audio", "duration", "manual_normalized_transcript"])
        audio_col = tbl.column("audio").to_pylist()
        dur_col = tbl.column("duration").to_pylist()
        text_col = tbl.column("manual_normalized_transcript").to_pylist()
        for i in range(tbl.num_rows):
            text = (text_col[i] or "").strip()
            if not text or dur_col[i] is None:
                continue
            rows.append({
                "uid": f"casa_pal__{split}__{stem}__{i:06d}",
                "_raw": audio_col[i]["bytes"], "_kind": "wav",
                "duration": float(dur_col[i]), "text": text, "mix_source": "casa_pal",
            })
    return rows


def write_shard(rows, out_dir: Path, name: str, batch_rows: int = 128):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    writer = pq.ParquetWriter(path, SCHEMA, compression="snappy")
    try:
        for s in range(0, len(rows), batch_rows):
            chunk = rows[s:s + batch_rows]
            audio = []
            for r in chunk:
                b = _pcm16_to_wav(r["_raw"]) if r["_kind"] == "pcm" else _wav_to_wav16k_mono(r["_raw"])
                audio.append({"bytes": b, "path": None})
            writer.write_table(pa.Table.from_pydict({
                "uid": [r["uid"] for r in chunk],
                "audio": audio,
                "duration": [r["duration"] for r in chunk],
                "text": [r["text"] for r in chunk],
                "mix_source": [r["mix_source"] for r in chunk],
            }, schema=SCHEMA))
            print(f"  {name}: {min(s + batch_rows, len(rows))}/{len(rows)}", end="\r", flush=True)
    finally:
        writer.close()
    hours = sum(r["duration"] for r in rows) / 3600.0
    print(f"\n  {name}: {len(rows)} rows, {hours:.4f}h -> {path}")
    return len(rows), hours


def main():
    # ---- qasr_lev native splits -> data_qasrlev_v1 ----
    for split, outname in (("train", "train"), ("val", "val"), ("test", "test")):
        rows = load_qasrlev_split(split)
        print(f"[qasrlev_v1/{outname}] {len(rows)} candidate rows")
        write_shard(rows, QASRLEV_OUT / outname, f"qasr_lev__{outname}")

    # ---- casa_pal all splits combined -> stage1 pretraining shard ----
    casapal_rows = sum((load_casapal_split(s) for s in ("train", "val", "test")), [])
    print(f"[casa_pal_all] {len(casapal_rows)} candidate rows")
    write_shard(casapal_rows, S1_OUT / "casa_pal_all" / "train", "casa_pal__all")

    # ---- assemble the stage-1 mix: qasr_lev(train only) + casa_pal_all + casa_jor + omni + layla ----
    mix_dir = S1_OUT / "qasrlev_target_mix" / "train"
    mix_dir.mkdir(parents=True, exist_ok=True)
    sources = [
        QASRLEV_OUT / "train" / "qasr_lev__train.parquet",
        S1_OUT / "casa_pal_all" / "train" / "casa_pal__all.parquet",
        S1_OUT / "jor" / "train" / "casa_jor__all.parquet",
        S1_OUT / "omni" / "train" / "omni__all.parquet",
        S1_OUT / "layla" / "train" / "layla__all.parquet",
    ]
    for src in sources:
        assert src.exists(), f"missing source shard: {src}"
        dst = mix_dir / src.name
        shutil.copy2(src, dst)
        print(f"[mix] copied {src} -> {dst}")

    total_rows, total_hours = 0, 0.0
    for f in mix_dir.glob("*.parquet"):
        t = pq.read_table(f, columns=["duration"])
        total_rows += t.num_rows
        total_hours += sum(x for x in t.column("duration").to_pylist() if x)
    print(f"[mix] TOTAL: {total_rows} rows, {total_hours:.2f}h -> {mix_dir}")


if __name__ == "__main__":
    main()
