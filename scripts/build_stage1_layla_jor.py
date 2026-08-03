#!/usr/bin/env python
"""Build the layla stage-1 pretraining view (full train+val+test, no sampling -- only 6.7h
total), harmonized to the same schema as the other data_stage1_v1/* views. layla's curated
audio is already struct<bytes: WAV, path> @16kHz mono (like masc), so this is a straight
concat + rename, no PCM/resample work (unlike qasr).

Output: data_stage1_v1/layla/train/layla__all.parquet
"""
import glob
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

CURATED = Path("/workspace/asr/Palestinian-ASR/data_curated_levant_binary_v1")
OUT = Path("/root/Palestinian-ASR/data_stage1_v1/layla/train")

SCHEMA = pa.schema([
    ("uid", pa.string()),
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("duration", pa.float64()),
    ("text", pa.string()),
    ("mix_source", pa.string()),
])


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    out_path = OUT / "layla__all.parquet"
    writer = pq.ParquetWriter(out_path, SCHEMA, compression="snappy")
    total_rows, total_hours = 0, 0.0
    try:
        for split in ("train", "val", "test"):
            files = sorted(glob.glob(str(CURATED / split / "layla" / "*.parquet")))
            for f in files:
                stem = Path(f).stem
                t = pq.read_table(f, columns=["audio", "duration", "manual_normalized_transcript"])
                audio_col = t.column("audio").to_pylist()
                dur_col = t.column("duration").to_pylist()
                text_col = t.column("manual_normalized_transcript").to_pylist()
                uids, audio, durs, texts = [], [], [], []
                for i in range(t.num_rows):
                    txt = (text_col[i] or "").strip()
                    d = dur_col[i]
                    if not txt or d is None:
                        continue
                    uids.append(f"layla__{split}__{stem}__{i:06d}")
                    audio.append({"bytes": audio_col[i]["bytes"], "path": None})
                    durs.append(float(d))
                    texts.append(txt)
                if uids:
                    writer.write_table(pa.Table.from_pydict({
                        "uid": uids, "audio": audio, "duration": durs, "text": texts,
                        "mix_source": ["layla"] * len(uids),
                    }, schema=SCHEMA))
                    total_rows += len(uids)
                    total_hours += sum(durs) / 3600.0
    finally:
        writer.close()
    print(f"[layla] wrote {total_rows} rows, {total_hours:.4f}h -> {out_path}")


if __name__ == "__main__":
    main()
