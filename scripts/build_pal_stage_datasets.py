#!/usr/bin/env python
"""Build the `pal` fine-tuning view + the three stage-1 pretraining views out of
`data_curated_levant_binary_v1` (the full 2,209h curated corpus), harmonized into exactly
the schema `unified.*.run.ipynb` expects (the same one `data_finetune_mix_v2` uses):

    uid: string
    audio: struct<bytes: binary (WAV PCM16 @16kHz mono), path: string>
    duration: double            # carried over from the curated corpus, not re-measured
    text: string                # manual_normalized_transcript (v3 where the source has one)
    mix_source: string          # casa_pal | casa_jor | omni

Views built (all written to LOCAL disk -- the curated corpus lives on the /workspace network
mount, and training I/O off that mount is slow):

  data_pal_v1/
    train = casa/pal {train + val}          664 rows   (~0.99h)
    val   = 30% of casa/pal test            199 rows
    test  = remaining 70% of casa/pal test  465 rows
    -- the 30/70 cut is a seeded shuffle of the native Casablanca test shard, so palVal and
       palTest are disjoint and neither overlaps palTrain.

  data_stage1_v1/jor/train      = casa/jor {train + val + test}   1,695 rows (~1.98h)
  data_stage1_v1/omni/train     = omni     {train + val + test}   1,311 rows (~7.91h)
  data_stage1_v1/omni_jor/train = both of the above               3,006 rows (~9.89h)
"""
import io, json, glob, hashlib
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import soxr

CURATED = Path("/workspace/asr/Palestinian-ASR/data_curated_levant_binary_v1")
PAL_OUT = Path("/root/Palestinian-ASR/data_pal_v1")
S1_OUT = Path("/root/Palestinian-ASR/data_stage1_v1")
SEED = 42
VAL_FRACTION_OF_TEST = 0.30
TARGET_SR = 16000

# audio.path is declared `string` explicitly (never inferred): an all-null field whose type
# pandas re-infers per shard is what produced mismatched struct types + ArrowNotImplementedError
# when load_dataset concatenated shards in the first build of data_finetune_mix_v2.
SCHEMA = pa.schema([
    ("uid", pa.string()),
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("duration", pa.float64()),
    ("text", pa.string()),
    ("mix_source", pa.string()),
])

# Curated leaves per logical source. Text column preference is per-source: omni went through a
# v3 re-clean pass (recover_omnilingual_token_span_rows_v3.py) that left the v2 column in place,
# so v3 wins where it is non-null; Casablanca only ever had the one normalized column.
SOURCES = {
    "casa_pal": {"leaf": "casa/pal", "text_cols": ["manual_normalized_transcript"]},
    "casa_jor": {"leaf": "casa/jor", "text_cols": ["manual_normalized_transcript"]},
    "omni": {"leaf": "omni", "text_cols": ["manual_normalized_transcript_v3",
                                           "manual_normalized_transcript"]},
}


def _leaf_files(source: str, split: str):
    return sorted(glob.glob(str(CURATED / split / SOURCES[source]["leaf"] / "*.parquet")))


def _to_wav16k_mono(raw: bytes) -> bytes:
    """Decode whatever the curated corpus stored (44.1kHz stereo WAV for Casablanca, 48kHz
    FLAC or WAV for omni, depending on whether the row came out of the segmentation pass) and
    re-encode to the one format every adapter's preprocess() expects: mono PCM16 @16kHz."""
    wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
    wav = wav.mean(axis=1)
    if sr != TARGET_SR:
        wav = soxr.resample(wav, sr, TARGET_SR)
    buf = io.BytesIO()
    sf.write(buf, wav, TARGET_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def load_rows(source: str, split: str):
    """One curated leaf -> list of harmonized dicts (audio still un-transcoded: transcoding is
    deferred to write_shard so a row selected into two views is only decoded once per write)."""
    files = _leaf_files(source, split)
    if not files:
        raise FileNotFoundError(f"no parquet under {CURATED / split / SOURCES[source]['leaf']}")
    text_cols = SOURCES[source]["text_cols"]
    out = []
    for f in files:
        stem = Path(f).stem
        tbl = pq.read_table(f)
        cols = {c: tbl.column(c).to_pylist() for c in
                ["audio", "duration"] + [c for c in text_cols if c in tbl.column_names]}
        for i in range(tbl.num_rows):
            text = ""
            for c in text_cols:
                v = cols.get(c, [None] * tbl.num_rows)[i]
                if v and str(v).strip():
                    text = str(v).strip(); break
            out.append({
                "uid": f"{source}__{split}__{stem}__{i:06d}",
                "_raw_audio": cols["audio"][i]["bytes"],
                "duration": float(cols["duration"][i]) if cols["duration"][i] is not None else None,
                "text": text,
                "mix_source": source,
            })
    return out


def write_shard(rows, out_dir: Path, name: str, batch_rows: int = 128):
    """Stream to parquet in row batches -- a full decode+re-encode of every row held in one
    Python list before writing would spike RSS for no reason."""
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    writer = pq.ParquetWriter(path, SCHEMA, compression="snappy")
    n_bytes = 0
    try:
        for s in range(0, len(rows), batch_rows):
            chunk = rows[s:s + batch_rows]
            audio = []
            for r in chunk:
                b = _to_wav16k_mono(r["_raw_audio"])
                n_bytes += len(b)
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
    hours = sum((r["duration"] or 0.0) for r in rows) / 3600.0
    print(f"  {name}: {len(rows)} rows, {hours:.4f}h, {n_bytes / 1e6:.1f}MB -> {path}")
    return {"rows": len(rows), "hours": round(hours, 4), "bytes": n_bytes, "path": str(path)}


def main():
    report = {"seed": SEED, "source": str(CURATED), "views": {}}

    # ---------------- pal view ----------------
    pal_train = load_rows("casa_pal", "train") + load_rows("casa_pal", "val")
    pal_eval = load_rows("casa_pal", "test")

    idx = np.arange(len(pal_eval))
    np.random.default_rng(SEED).shuffle(idx)
    n_val = int(round(len(pal_eval) * VAL_FRACTION_OF_TEST))
    val_rows = [pal_eval[i] for i in idx[:n_val]]
    test_rows = [pal_eval[i] for i in idx[n_val:]]
    assert not ({r["uid"] for r in val_rows} & {r["uid"] for r in test_rows})
    assert not ({r["uid"] for r in pal_train} & {r["uid"] for r in pal_eval})

    print(f"[pal] train={len(pal_train)} val={len(val_rows)} test={len(test_rows)}")
    report["views"]["data_pal_v1"] = {
        "train": write_shard(pal_train, PAL_OUT / "train", "casa_pal__train"),
        "val": write_shard(val_rows, PAL_OUT / "val", "casa_pal__val"),
        "test": write_shard(test_rows, PAL_OUT / "test", "casa_pal__test"),
        "composition": {
            "train": "casa/pal train + casa/pal val (curated)",
            "val": f"{VAL_FRACTION_OF_TEST:.0%} of casa/pal test, seeded shuffle (seed={SEED})",
            "test": f"remaining {1 - VAL_FRACTION_OF_TEST:.0%} of casa/pal test",
        },
    }

    # ---------------- stage-1 views ----------------
    jor = sum((load_rows("casa_jor", s) for s in ("train", "val", "test")), [])
    omni = sum((load_rows("omni", s) for s in ("train", "val", "test")), [])
    print(f"[stage1] jor={len(jor)} omni={len(omni)} omni_jor={len(jor) + len(omni)}")

    report["views"]["data_stage1_v1/jor"] = {
        "train": write_shard(jor, S1_OUT / "jor" / "train", "casa_jor__all")}
    report["views"]["data_stage1_v1/omni"] = {
        "train": write_shard(omni, S1_OUT / "omni" / "train", "omni__all")}
    # omni_jor is written as two shards (one per source) rather than one concatenated shard --
    # load_dataset globs the whole split dir, so the shard boundary is invisible to training,
    # and keeping them separate makes the per-source row counts checkable on disk.
    report["views"]["data_stage1_v1/omni_jor"] = {
        "train_omni": write_shard(omni, S1_OUT / "omni_jor" / "train", "omni__all"),
        "train_jor": write_shard(jor, S1_OUT / "omni_jor" / "train", "casa_jor__all")}

    for d in (PAL_OUT, S1_OUT):
        (d / "reports").mkdir(parents=True, exist_ok=True)
    (PAL_OUT / "reports" / "manifest.json").write_text(json.dumps(report, indent=2))
    (S1_OUT / "reports" / "manifest.json").write_text(json.dumps(report, indent=2))
    print(json.dumps({k: {kk: (vv if not isinstance(vv, dict) else
                              {"rows": vv.get("rows"), "hours": vv.get("hours")})
                          for kk, vv in v.items()} for k, v in report["views"].items()}, indent=2))


if __name__ == "__main__":
    main()
