#!/usr/bin/env python
"""Build the qasr_lev + masc_lev stage-1 pretraining views (full train+val+test of each,
no sampling), harmonized to the same schema as the other data_stage1_v1/* views:

    uid, audio: struct<bytes: binary (WAV PCM16 @16kHz mono), path: string>, duration, text,
    mix_source

Two output views:
  data_stage1_v1/qasr_lev_masc_lev/train/
      qasr_lev__all.parquet   (qasr/lev train+val+test, raw-PCM16LE source -> WAV wrap)
      masc_lev__all.parquet   (masc/lev train+val+test, source audio already a WAV-bytes
                                struct -> re-encode via soundfile passthrough for schema
                                consistency, no resample needed since already 16kHz mono)

  data_stage1_v1/qasr_lev_masc_lev_plus_nonlev50h/train/
      symlinks to the two files above + the existing qasr_non_lev_50h/train shard, so the
      three corpora are visible under one directory without duplicating ~11GB on disk.
"""
import io, glob
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

CURATED = Path("/workspace/asr/Palestinian-ASR/data_curated_levant_binary_v1")
OUT1 = Path("/root/Palestinian-ASR/data_stage1_v1/qasr_lev_masc_lev/train")
OUT2 = Path("/root/Palestinian-ASR/data_stage1_v1/qasr_lev_masc_lev_plus_nonlev50h/train")
NONLEV50H = Path("/root/Palestinian-ASR/data_stage1_v1/qasr_non_lev_50h/train/qasr_non_lev__50h.parquet")
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


def build_qasr_lev(out_path: Path):
    """qasr/lev: raw PCM16LE @16kHz, sampling_rate column, manual_normalized_transcript text."""
    writer = pq.ParquetWriter(out_path, SCHEMA, compression="snappy")
    total_rows = 0
    total_hours = 0.0
    try:
        for split in ("train", "val", "test"):
            files = sorted(glob.glob(str(CURATED / split / "qasr" / "lev" / "*.parquet")))
            for f in files:
                stem = Path(f).stem
                t = pq.read_table(f, columns=["audio", "sampling_rate", "duration",
                                              "manual_normalized_transcript"])
                srs = t.column("sampling_rate").to_pylist()
                assert all(s == TARGET_SR for s in srs), f"{f} has non-16kHz rows"
                audio_col = t.column("audio").to_pylist()
                dur_col = t.column("duration").to_pylist()
                text_col = t.column("manual_normalized_transcript").to_pylist()
                batch = 128
                for s in range(0, t.num_rows, batch):
                    e = min(s + batch, t.num_rows)
                    uids, audio, durs, texts = [], [], [], []
                    for i in range(s, e):
                        txt = (text_col[i] or "").strip()
                        d = dur_col[i]
                        if not txt or d is None:
                            continue
                        uids.append(f"qasr_lev__{split}__{stem}__{i:06d}")
                        audio.append({"bytes": _pcm16_to_wav(audio_col[i]), "path": None})
                        durs.append(float(d))
                        texts.append(txt)
                    if not uids:
                        continue
                    writer.write_table(pa.Table.from_pydict({
                        "uid": uids, "audio": audio, "duration": durs, "text": texts,
                        "mix_source": ["qasr_lev"] * len(uids),
                    }, schema=SCHEMA))
                    total_rows += len(uids)
                    total_hours += sum(durs) / 3600.0
                print(f"  [qasr_lev] {split}/{stem}: running total {total_rows} rows, "
                      f"{total_hours:.2f}h", end="\r", flush=True)
    finally:
        writer.close()
    print(f"\n[qasr_lev] wrote {total_rows} rows, {total_hours:.4f}h -> {out_path}")
    return total_rows, total_hours


def build_masc_lev(out_path: Path):
    """masc/lev: audio already struct<bytes: WAV, path>, manual_normalized_transcript text."""
    writer = pq.ParquetWriter(out_path, SCHEMA, compression="snappy")
    total_rows = 0
    total_hours = 0.0
    try:
        for split in ("train", "val", "test"):
            files = sorted(glob.glob(str(CURATED / split / "masc" / "lev" / "*.parquet")))
            for f in files:
                stem = Path(f).stem
                t = pq.read_table(f, columns=["audio", "duration", "manual_normalized_transcript"])
                audio_col = t.column("audio").to_pylist()
                dur_col = t.column("duration").to_pylist()
                text_col = t.column("manual_normalized_transcript").to_pylist()
                batch = 128
                for s in range(0, t.num_rows, batch):
                    e = min(s + batch, t.num_rows)
                    uids, audio, durs, texts = [], [], [], []
                    for i in range(s, e):
                        txt = (text_col[i] or "").strip()
                        d = dur_col[i]
                        if not txt or d is None:
                            continue
                        uids.append(f"masc_lev__{split}__{stem}__{i:06d}")
                        audio.append({"bytes": audio_col[i]["bytes"], "path": None})
                        durs.append(float(d))
                        texts.append(txt)
                    if not uids:
                        continue
                    writer.write_table(pa.Table.from_pydict({
                        "uid": uids, "audio": audio, "duration": durs, "text": texts,
                        "mix_source": ["masc_lev"] * len(uids),
                    }, schema=SCHEMA))
                    total_rows += len(uids)
                    total_hours += sum(durs) / 3600.0
                print(f"  [masc_lev] {split}/{stem}: running total {total_rows} rows, "
                      f"{total_hours:.2f}h", end="\r", flush=True)
    finally:
        writer.close()
    print(f"\n[masc_lev] wrote {total_rows} rows, {total_hours:.4f}h -> {out_path}")
    return total_rows, total_hours


def main():
    OUT1.mkdir(parents=True, exist_ok=True)
    OUT2.mkdir(parents=True, exist_ok=True)

    qasr_path = OUT1 / "qasr_lev__all.parquet"
    masc_path = OUT1 / "masc_lev__all.parquet"
    qr, qh = build_qasr_lev(qasr_path)
    mr, mh = build_masc_lev(masc_path)
    print(f"\n[combined qasr_lev_masc_lev] {qr+mr} rows, {qh+mh:.4f}h")

    assert NONLEV50H.exists(), f"missing {NONLEV50H} -- build_stage1_qasr_nonlev_50h.py first"
    for src, name in [(qasr_path, "qasr_lev__all.parquet"),
                       (masc_path, "masc_lev__all.parquet"),
                       (NONLEV50H, "qasr_non_lev__50h.parquet")]:
        dst = OUT2 / name
        if dst.exists() or dst.is_symlink():
            dst.unlink()
        dst.symlink_to(src)
    print(f"[combined qasr_lev_masc_lev_plus_nonlev50h] symlinked 3 shards -> {OUT2}")


if __name__ == "__main__":
    main()
