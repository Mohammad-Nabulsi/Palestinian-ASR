#!/usr/bin/env python
"""Build data_layla_v1/{train,val,test} -- Layla's own native curated splits, harmonized to
the same schema as data_pal_v1/data_qasrlev_v1 (uid, audio: struct<bytes,path>, duration, text,
mix_source). Used as the fine-tuning TARGET for the "layla as target" generalization-transfer
experiment: reusing existing stage-1 pretrained checkpoints from the pal study, doing a fresh
LoRA stage fine-tuned on Layla train, early-stopping on Layla's own native val split.

Layla curated rows are already audio: struct<bytes: WAV, path> (like casa_pal), so this uses the
same resample-to-16k-mono harmonization as build_pal_stage_datasets.py / build_stage1_qasrlev_target.py,
not the raw-PCM path QASR needs.

train=715 rows/4.68h, val=153 rows/1.02h, test=154 rows/1.00h (native curated split sizes).
"""
import io, glob
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import soxr

CURATED = Path("/workspace/asr/Palestinian-ASR/data_curated_levant_binary_v1")
OUT = Path("/root/Palestinian-ASR/data_layla_v1")
TARGET_SR = 16000

SCHEMA = pa.schema([
    ("uid", pa.string()),
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("duration", pa.float64()),
    ("text", pa.string()),
    ("mix_source", pa.string()),
])


def _wav_to_wav16k_mono(raw: bytes) -> bytes:
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        wav = soxr.resample(wav, sr, TARGET_SR)
    buf = io.BytesIO()
    sf.write(buf, wav, TARGET_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def load_split(split: str):
    files = sorted(glob.glob(str(CURATED / split / "layla" / "*.parquet")))
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
                "uid": f"layla__{split}__{stem}__{i:06d}",
                "_raw": audio_col[i]["bytes"],
                "duration": float(dur_col[i]), "text": text, "mix_source": "layla",
            })
    return rows


def write_shard(rows, out_dir: Path, name: str, batch_rows: int = 128):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    writer = pq.ParquetWriter(path, SCHEMA, compression="snappy")
    try:
        for s in range(0, len(rows), batch_rows):
            chunk = rows[s:s + batch_rows]
            audio = [{"bytes": _wav_to_wav16k_mono(r["_raw"]), "path": None} for r in chunk]
            writer.write_table(pa.Table.from_pydict({
                "uid": [r["uid"] for r in chunk],
                "audio": audio,
                "duration": [r["duration"] for r in chunk],
                "text": [r["text"] for r in chunk],
                "mix_source": [r["mix_source"] for r in chunk],
            }, schema=SCHEMA))
    finally:
        writer.close()
    hours = sum(r["duration"] for r in rows) / 3600.0
    print(f"  {name}: {len(rows)} rows, {hours:.4f}h -> {path}")


def main():
    for split, outname in (("train", "train"), ("val", "val"), ("test", "test")):
        rows = load_split(split)
        print(f"[layla_v1/{outname}] {len(rows)} candidate rows")
        write_shard(rows, OUT / outname, f"layla__{outname}")


if __name__ == "__main__":
    main()
