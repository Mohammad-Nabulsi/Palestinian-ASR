#!/usr/bin/env python
"""Build the 7 held-out generalization-eval datasets used to probe each pal-study run's
adapters on domains they did/didn't train on.

Test-set domains (curated corpus's own held-out test split, used whole):
    eval_sets/omni_test       (198 rows, 1.17h)
    eval_sets/layla_test      (154 rows, 1.00h)
    eval_sets/casa_jor_test   (848 rows, 0.98h)

Fresh-2h domains (never touched by ANY training run in this study):
    eval_sets/qasr_lev_2h     -- from qasr/lev's curated *test* split (raw PCM16 source)
    eval_sets/masc_lev_2h     -- from masc/lev's curated *val+test* (test alone is only 0.3h)
    eval_sets/qasr_non_lev_2h -- from qasr/non_lev's *val* split (all runs only ever touched
                                 the *train* split's first 50h; val/test are untouched)
    eval_sets/masc_non_lev_2h -- from masc/non_lev's *test* split (nobody has touched masc/non_lev
                                 at all yet, any split is fine)

All views harmonized to: uid, audio: struct<bytes: WAV PCM16 @16kHz mono, path>, duration,
text, mix_source -- same schema as data_stage1_v1/*, so the existing WhisperAdapter/
PredictAPI/EvaluateAPI pipeline needs zero changes to consume these.
"""
import io, glob
from pathlib import Path

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

CURATED = Path("/workspace/asr/Palestinian-ASR/data_curated_levant_binary_v1")
OUT = Path("/root/Palestinian-ASR/eval_sets")
TARGET_SR = 16000
TARGET_SECONDS = 2 * 3600.0

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


def _write(rows, out_path, name):
    OUT.mkdir(parents=True, exist_ok=True)
    writer = pq.ParquetWriter(out_path, SCHEMA, compression="snappy")
    n, hours = 0, 0.0
    try:
        for s in range(0, len(rows), 128):
            chunk = rows[s:s + 128]
            writer.write_table(pa.Table.from_pydict({
                "uid": [r["uid"] for r in chunk],
                "audio": [{"bytes": r["audio_bytes"], "path": None} for r in chunk],
                "duration": [r["duration"] for r in chunk],
                "text": [r["text"] for r in chunk],
                "mix_source": [name] * len(chunk),
            }, schema=SCHEMA))
            n += len(chunk)
            hours += sum(r["duration"] for r in chunk) / 3600.0
    finally:
        writer.close()
    print(f"[{name}] wrote {n} rows, {hours:.4f}h -> {out_path}")


def _harmonized_struct_source(leaf_glob, text_cols, name_prefix):
    """masc/layla/omni/casa_jor style: audio already struct<bytes: WAV, path>. text_cols is a
    preference list (first non-null wins) -- omni's v3 re-clean left most rows null, matching
    build_pal_stage_datasets.py's original ["manual_normalized_transcript_v3",
    "manual_normalized_transcript"] fallback."""
    if isinstance(text_cols, str):
        text_cols = [text_cols]
    rows = []
    for f in sorted(glob.glob(leaf_glob)):
        stem = Path(f).stem
        t = pq.read_table(f, columns=["audio", "duration"] + text_cols)
        audio_col, dur_col = t.column("audio").to_pylist(), t.column("duration").to_pylist()
        text_col_lists = [t.column(c).to_pylist() for c in text_cols]
        for i in range(t.num_rows):
            txt = ""
            for col_vals in text_col_lists:
                v = col_vals[i]
                if v and str(v).strip():
                    txt = str(v).strip(); break
            d = dur_col[i]
            if not txt or d is None:
                continue
            rows.append({"uid": f"{name_prefix}__{stem}__{i:06d}",
                         "audio_bytes": audio_col[i]["bytes"], "duration": float(d), "text": txt})
    return rows


def _raw_pcm_source(leaf_glob, text_col, name_prefix, cap_seconds=None, seed=42):
    """qasr style: audio is bare PCM16LE binary + sampling_rate column."""
    rows = []
    for f in sorted(glob.glob(leaf_glob)):
        stem = Path(f).stem
        t = pq.read_table(f, columns=["audio", "sampling_rate", "duration", text_col])
        srs = t.column("sampling_rate").to_pylist()
        assert all(s == TARGET_SR for s in srs), f"{f} has non-16kHz rows"
        audio_col, dur_col, text_col_vals = (t.column("audio").to_pylist(),
                                              t.column("duration").to_pylist(),
                                              t.column(text_col).to_pylist())
        for i in range(t.num_rows):
            txt = (text_col_vals[i] or "").strip()
            d = dur_col[i]
            if not txt or d is None:
                continue
            rows.append({"uid": f"{name_prefix}__{stem}__{i:06d}",
                         "audio_bytes": _pcm16_to_wav(audio_col[i]), "duration": float(d), "text": txt})
    if cap_seconds is not None:
        idx = np.arange(len(rows))
        np.random.default_rng(seed).shuffle(idx)
        picked, cum = [], 0.0
        for i in idx:
            if cum >= cap_seconds:
                break
            picked.append(rows[i]); cum += rows[i]["duration"]
        rows = picked
    return rows


def main():
    # ---- test-set domains (whole split, no capping) ----
    _write(_harmonized_struct_source(
        str(CURATED / "test" / "omni" / "*.parquet"),
        ["manual_normalized_transcript_v3", "manual_normalized_transcript"], "omni_test"),
        OUT / "omni_test.parquet", "omni_test")
    _write(_harmonized_struct_source(
        str(CURATED / "test" / "layla" / "*.parquet"), "manual_normalized_transcript", "layla_test"),
        OUT / "layla_test.parquet", "layla_test")
    _write(_harmonized_struct_source(
        str(CURATED / "test" / "casa" / "jor" / "*.parquet"), "manual_normalized_transcript", "casa_jor_test"),
        OUT / "casa_jor_test.parquet", "casa_jor_test")

    # ---- fresh-2h domains ----
    _write(_raw_pcm_source(
        str(CURATED / "test" / "qasr" / "lev" / "*.parquet"), "manual_normalized_transcript",
        "qasr_lev_2h", cap_seconds=TARGET_SECONDS),
        OUT / "qasr_lev_2h.parquet", "qasr_lev_2h")

    masc_lev_rows = (
        _harmonized_struct_source(str(CURATED / "val" / "masc" / "lev" / "*.parquet"),
                                   "manual_normalized_transcript", "masc_lev_2h")
        + _harmonized_struct_source(str(CURATED / "test" / "masc" / "lev" / "*.parquet"),
                                     "manual_normalized_transcript", "masc_lev_2h"))
    idx = np.arange(len(masc_lev_rows)); np.random.default_rng(42).shuffle(idx)
    picked, cum = [], 0.0
    for i in idx:
        if cum >= TARGET_SECONDS:
            break
        picked.append(masc_lev_rows[i]); cum += masc_lev_rows[i]["duration"]
    _write(picked, OUT / "masc_lev_2h.parquet", "masc_lev_2h")

    _write(_raw_pcm_source(
        str(CURATED / "val" / "qasr" / "non_lev" / "*.parquet"), "manual_normalized_transcript",
        "qasr_non_lev_2h", cap_seconds=TARGET_SECONDS),
        OUT / "qasr_non_lev_2h.parquet", "qasr_non_lev_2h")

    masc_nonlev_rows = _harmonized_struct_source(
        str(CURATED / "test" / "masc" / "non_lev" / "*.parquet"), "manual_normalized_transcript",
        "masc_non_lev_2h")
    idx = np.arange(len(masc_nonlev_rows)); np.random.default_rng(42).shuffle(idx)
    picked, cum = [], 0.0
    for i in idx:
        if cum >= TARGET_SECONDS:
            break
        picked.append(masc_nonlev_rows[i]); cum += masc_nonlev_rows[i]["duration"]
    _write(picked, OUT / "masc_non_lev_2h.parquet", "masc_non_lev_2h")


if __name__ == "__main__":
    main()
