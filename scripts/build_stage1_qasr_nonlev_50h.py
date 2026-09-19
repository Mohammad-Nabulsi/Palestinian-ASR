#!/usr/bin/env python
"""Build a ~50h stage-1 pretraining view sampled from QASR non-Levantine
(data_curated_levant_binary_v1 train/qasr/non_lev), harmonized to the same schema
unified.whisper.run.ipynb / pal_s1e1of2_run.py expect (matches data_stage1_v1/{jor,omni,
omni_jor,qasr_lev_10h}):

    uid: string
    audio: struct<bytes: binary (WAV PCM16 @16kHz mono), path: string>
    duration: double
    text: string
    mix_source: string  # "qasr_non_lev"

Same raw-PCM16LE@16kHz-mono format as qasr/lev (see build_stage1_qasr_lev_10h.py), but the
non_lev leaf is ~1,034,147 rows / 1218h across 21 shards -- loading every row's audio into
memory before sampling (as the lev/10h script did, fine at 21.8k rows) is not viable at this
scale (~135GB of decoded audio). Two passes instead:
  1. metadata-only scan (duration) across all shards -> seeded shuffle -> take rows until the
     50h target is reached, recording (shard_index, row_index_within_shard) for each pick.
  2. group picks by shard, re-open only those shards, pull audio+text only for the picked row
     indices, transcode, write.
"""
import io, glob
from collections import defaultdict
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

CURATED_LEAF = Path("/workspace/asr/Palestinian-ASR/data_curated_levant_binary_v1/train/qasr/non_lev")
OUT = Path("/root/Palestinian-ASR/data_stage1_v1/qasr_non_lev_50h/train")
SEED = 42
TARGET_HOURS = 50.0
TARGET_SR = 16000
MIX_SOURCE = "qasr_non_lev"

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
    print(f"[qasr_non_lev_50h] {len(files)} source shards under {CURATED_LEAF}")

    # ---- pass 1: metadata-only scan ----
    candidates = []  # (shard_idx, row_idx, duration)
    total_hours = 0.0
    for si, f in enumerate(files):
        t = pq.read_table(f, columns=["duration", "sampling_rate", "manual_normalized_transcript"])
        srs = t.column("sampling_rate").to_pylist()
        assert all(s == TARGET_SR for s in srs), f"{f} has non-16kHz rows"
        dur_col = t.column("duration").to_pylist()
        text_col = t.column("manual_normalized_transcript").to_pylist()
        for ri in range(t.num_rows):
            d = dur_col[ri]
            txt = text_col[ri]
            if d is None or not (txt and str(txt).strip()):
                continue
            candidates.append((si, ri, float(d)))
            total_hours += float(d) / 3600.0
        print(f"  scanned shard {si+1}/{len(files)}: {Path(f).name} "
              f"({len(candidates)} candidates so far)", end="\r", flush=True)
    print(f"\n[qasr_non_lev_50h] {len(candidates)} candidate rows, {total_hours:.1f}h available")

    idx = np.arange(len(candidates))
    np.random.default_rng(SEED).shuffle(idx)
    target_s = TARGET_HOURS * 3600.0
    picked, cum = [], 0.0
    for i in idx:
        if cum >= target_s:
            break
        picked.append(candidates[i])
        cum += candidates[i][2]
    print(f"[qasr_non_lev_50h] picked {len(picked)} rows, {cum / 3600.0:.4f}h "
          f"(target {TARGET_HOURS}h)")

    # ---- pass 2: re-open only the shards that have a pick, pull audio+text for those rows ----
    by_shard = defaultdict(list)
    for si, ri, dur in picked:
        by_shard[si].append((ri, dur))

    OUT.mkdir(parents=True, exist_ok=True)
    path = OUT / "qasr_non_lev__50h.parquet"
    writer = pq.ParquetWriter(path, SCHEMA, compression="snappy")
    written = 0
    try:
        for si, row_list in sorted(by_shard.items()):
            f = files[si]
            stem = Path(f).stem
            t = pq.read_table(f, columns=["audio", "duration", "manual_normalized_transcript"])
            audio_col = t.column("audio")
            text_col = t.column("manual_normalized_transcript")
            row_list.sort()
            batch = 128
            for s in range(0, len(row_list), batch):
                chunk = row_list[s:s + batch]
                uids, audio, durs, texts = [], [], [], []
                for ri, dur in chunk:
                    uids.append(f"qasr_non_lev__train__{stem}__{ri:06d}")
                    audio.append({"bytes": _pcm16_to_wav(audio_col[ri].as_py()), "path": None})
                    durs.append(dur)
                    texts.append(str(text_col[ri].as_py()).strip())
                writer.write_table(pa.Table.from_pydict({
                    "uid": uids, "audio": audio, "duration": durs, "text": texts,
                    "mix_source": [MIX_SOURCE] * len(chunk),
                }, schema=SCHEMA))
                written += len(chunk)
                print(f"  wrote {written}/{len(picked)}", end="\r", flush=True)
    finally:
        writer.close()
    print(f"\n[qasr_non_lev_50h] wrote {written} rows, {cum/3600.0:.4f}h -> {path}")


if __name__ == "__main__":
    main()
