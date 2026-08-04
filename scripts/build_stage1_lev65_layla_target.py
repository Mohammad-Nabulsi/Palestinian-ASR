#!/usr/bin/env python
"""Build data_stage1_v1/lev65_combined/train -- a broader-threshold "lev" pretraining corpus:
every QASR/MASC row whose BADREX/MMS-300M audio-stage Levantine probability exceeds 0.65
(looser than the original pipeline's 0.80 cutoff used for data_curated_levant_binary_v1's own
lev/non_lev split), re-extracted from the curated corpus by joining on uid.

Source of the continuous scores: Runs/dialect_scan_badrex_mms300m_lev08_text_candidates_{qasr,masc_c}_only/
row_probabilities.jsonl (the completed, PCM-fixed audio-stage classifier pass -- 122,348 QASR /
48,455 MASC candidate rows already pre-filtered by the >=0.80 TEXT-stage classifier). Their own
"uid" field embeds the exact same uid format the curated corpus uses
(qasr:{recording_id}:{segment_id}), after the "uid=" prefix -- e.g.
".../shard_000000...clean.parquet:uid=qasr:0010E262-...:0010E262_..._utt_12_align" joins to
curated row uid "qasr:0010E262-...:0010E262_..._utt_12_align". This lets us skip the missing
intermediate "data/clean/*.parquet" files entirely and re-pull audio+text straight from
data_curated_levant_binary_v1 (which already has everything, just split into lev/non_lev at the
0.80 cutoff -- rows scoring 0.65-0.80 currently sit in non_lev and need pulling from there too).

At >0.65: QASR 34,806 rows/41.92h (vs 31,178/37.83h at >=0.80), MASC 9,036 rows/11.50h (vs
8,214/10.51h) -- roughly 4-5h more combined audio than the existing lev subset.

Two-pass: pass 1 scans uid-only across ALL qasr/masc shards (both lev and non_lev dirs, all
splits) to find which shard+row indices match the target uid set; pass 2 re-opens only the
shards with a hit to pull audio+text for those specific rows.
"""
import json
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq
import numpy as np
import soundfile as sf
import io

CURATED = Path("/workspace/asr/Palestinian-ASR/data_curated_levant_binary_v1")
DIALECT_RUNS = Path("/workspace/asr/Palestinian-ASR/Runs")
OUT = Path("/root/Palestinian-ASR/data_stage1_v1/lev65_combined/train")
TARGET_SR = 16000
THRESHOLD = 0.65

SCHEMA = pa.schema([
    ("uid", pa.string()),
    ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
    ("duration", pa.float64()),
    ("text", pa.string()),
    ("mix_source", pa.string()),
])


def target_uids(row_probs_path, source_prefix):
    uids = set()
    with open(row_probs_path) as f:
        for line in f:
            try:
                r = json.loads(line)
            except Exception:
                continue
            if r.get("status") != "ok":
                continue
            lev = r.get("label_scores", {}).get("Levantine")
            if lev is None or lev <= THRESHOLD:
                continue
            raw_uid = r.get("uid", "")
            if ":uid=" not in raw_uid:
                continue
            joined = raw_uid.split(":uid=", 1)[1]
            if joined.startswith(source_prefix):
                uids.add(joined)
    return uids


def scan_and_extract(source_leaf, want_uids, mix_source, is_pcm):
    """source_leaf e.g. 'qasr' -- scans both {source_leaf}/lev and {source_leaf}/non_lev under
    every split, since 0.65-0.80-scoring rows currently live in non_lev. True two-pass: pass 1
    reads ONLY the uid column (cheap) to find which row indices match per shard; pass 2 re-opens
    just those shards and uses .take() to pull audio/text for exactly the matching rows -- never
    materializes the audio column for a whole 1M+-row source in memory."""
    remaining = set(want_uids)
    hits_by_shard = []  # (path, [row_indices])
    all_leaves = [CURATED / split / source_leaf / dialect_dir
                  for split in ("train", "val", "test") for dialect_dir in ("lev", "non_lev")]
    for leaf in all_leaves:
        if not leaf.exists() or not remaining:
            continue
        for f in sorted(leaf.glob("*.parquet")):
            if not remaining:
                break
            uid_col = pq.read_table(f, columns=["uid"]).column("uid").to_pylist()
            hit_idx = [i for i, u in enumerate(uid_col) if u in remaining]
            if hit_idx:
                hits_by_shard.append((f, hit_idx))
                for i in hit_idx:
                    remaining.discard(uid_col[i])
                print(f"  [{source_leaf}] pass1 {f.name}: +{len(hit_idx)} hits, {len(remaining)} remaining", flush=True)
    if remaining:
        print(f"  [{source_leaf}] WARNING: {len(remaining)} target uids never found")

    found = []
    cols = ["uid", "duration", "manual_normalized_transcript", "audio"]
    for f, hit_idx in hits_by_shard:
        # NOTE: .take() on this table can hit "offset overflow" on shards whose binary audio
        # column exceeds the int32 offset limit -- reading full python lists and indexing
        # avoids the Arrow-level concat entirely (these are the already-narrowed hit shards
        # from pass 1, not the full 1.78M-row corpus, so this is cheap).
        tbl = pq.read_table(f, columns=cols)
        hit_set = set(hit_idx)
        uid_col = tbl.column("uid").to_pylist()
        dur_col = tbl.column("duration").to_pylist()
        text_col = tbl.column("manual_normalized_transcript").to_pylist()
        audio_col = tbl.column("audio").to_pylist()
        n = 0
        for i in hit_idx:
            text = (text_col[i] or "").strip()
            if not text or dur_col[i] is None:
                continue
            raw = audio_col[i] if is_pcm else audio_col[i]["bytes"]
            found.append({
                "uid": uid_col[i], "_raw": raw, "_kind": "pcm" if is_pcm else "wav",
                "duration": float(dur_col[i]), "text": text, "mix_source": mix_source,
            })
            n += 1
        print(f"  [{source_leaf}] pass2 {f.name}: extracted {n} rows", flush=True)
    return found


def _pcm16_to_wav(raw: bytes) -> bytes:
    arr = np.frombuffer(raw, dtype="<i2")
    buf = io.BytesIO()
    sf.write(buf, arr, TARGET_SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def write_shard(rows, out_dir: Path, name: str, batch_rows: int = 128):
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / f"{name}.parquet"
    writer = pq.ParquetWriter(path, SCHEMA, compression="snappy")
    try:
        for s in range(0, len(rows), batch_rows):
            chunk = rows[s:s + batch_rows]
            audio = [{"bytes": _pcm16_to_wav(r["_raw"]) if r["_kind"] == "pcm" else r["_raw"],
                      "path": None} for r in chunk]
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
    # MASC is deliberately excluded: its curated rows have no per-segment uid (only a
    # non-unique video_id -- one video maps to dozens of segments), so a video_id join would
    # silently pull in unverified segments never actually scored by the classifier. QASR's
    # row_probabilities uid maps 1:1 and exactly onto the curated corpus's own uid column, so
    # only QASR is used here (34,806 rows/41.92h of the ~43,842/53.4h combined >0.65 pool).
    qasr_uids = target_uids(
        DIALECT_RUNS / "dialect_scan_badrex_mms300m_lev08_text_candidates_qasr_only" / "row_probabilities.jsonl",
        "qasr:")
    print(f"[targets] qasr={len(qasr_uids)}")

    qasr_rows = scan_and_extract("qasr", qasr_uids, "qasr_lev65", is_pcm=True)
    print(f"[found] qasr={len(qasr_rows)}")

    write_shard(qasr_rows, OUT, "qasr_lev65")


if __name__ == "__main__":
    main()
