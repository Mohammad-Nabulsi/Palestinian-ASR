#!/usr/bin/env python3
"""Resumable full-corpus Badrex acoustic dialect scan.

Scans every QASR and MASC row (both old lev/non_lev leaves), not only rows that
previously passed a text classifier.  Output is durable, shard-resumable JSONL.
R2 authentication is deliberately supplied through standard rclone environment
variables/configuration; credentials never belong in this source file.
"""
from __future__ import annotations
import argparse, io, json, subprocess, time
from pathlib import Path
import librosa, numpy as np, pyarrow.parquet as pq, soundfile as sf, torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

MODEL_ID = "badrex/mms-300m-arabic-dialect-identifier"
TARGET_SR = 16000

def rclone(args: list[str]) -> str:
    return subprocess.run(["rclone", *args], check=True, text=True, capture_output=True).stdout

def decode(cell, sampling_rate=None):
    raw = cell["bytes"] if isinstance(cell, dict) else bytes(cell)
    # QASR's "audio" struct holds raw headerless PCM16 (with the real rate in a
    # sibling `sampling_rate` column) rather than self-describing WAV/FLAC/OGG --
    # the same gotcha documented elsewhere in this repo for QASR's eval-set audio
    # (scripts/zero_shot_eval/audio_io.py). Calling sf.read() on it unconditionally
    # (as the original version of this function did) makes libsndfile try to
    # resync through what looks like garbage MPEG-header noise, which either
    # raises "Format not recognised" (~100% of QASR rows, confirmed empirically)
    # or occasionally mis-decodes to an implausible length. MASC's audio genuinely
    # is self-describing, so it must keep going through sf.read().
    if sampling_rate is not None and raw[:4] not in (b"RIFF", b"fLaC", b"OggS"):
        wave = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
        sr = int(sampling_rate)
    else:
        wave, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=False)
    if wave.ndim > 1: wave = wave.mean(axis=1)
    if sr != TARGET_SR: wave = librosa.resample(wave, orig_sr=sr, target_sr=TARGET_SR)
    return np.ascontiguousarray(wave, dtype=np.float32)

def get_columns(table_columns: list[str], source: str):
    text = next((c for c in ("text", "normalized_transcript", "transcript", "manual_normalized_transcript") if c in table_columns), None)
    ident = "video_id" if source == "masc" else "uid"
    if text is None or ident not in table_columns:
        raise ValueError(f"{source}: required text/id field absent; columns={table_columns}")
    return text, ident

@torch.no_grad()
def classify(waves, processor, model, device, batch_size):
    out, i, bs = [], 0, batch_size
    n_since_clear = 0
    while i < len(waves):
        try:
            x = processor(waves[i:i+bs], sampling_rate=TARGET_SR, padding=True, return_tensors="pt")
            x = {k: v.to(device) for k, v in x.items()}
            with torch.autocast("cuda", dtype=torch.float16, enabled=device == "cuda"):
                logits = model(**x).logits
            out.extend(torch.softmax(logits.float(), -1).cpu().numpy())
            i += bs
            n_since_clear += 1
            # Large shards (QASR non_lev: ~49k rows/shard vs MASC's ~14k) run many more
            # variable-length-padded batches per shard than MASC did; the caching
            # allocator's reserved-but-unallocated pool grows across iterations and can
            # starve a later, larger allocation even though total free memory looks fine
            # in aggregate (observed: 17GB reserved, needed 6.5GB more, only 2.6GB free).
            # Periodic clearing plus PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
            # (set by the supervisor) avoids this without slowing down the common case.
            if n_since_clear >= 20:
                torch.cuda.empty_cache()
                n_since_clear = 0
        except torch.cuda.OutOfMemoryError:
            if bs == 1: raise
            torch.cuda.empty_cache(); bs //= 2
            print(f"OOM: retrying at batch_size={bs}", flush=True)
    return out, bs

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--remote", default="R2:backup/transfer/curated_corpus/train")
    p.add_argument("--output-root", type=Path, required=True)
    p.add_argument("--work-dir", type=Path, required=True)
    p.add_argument("--batch-size", type=int, default=96)
    a = p.parse_args(); a.output_root.mkdir(parents=True, exist_ok=True); a.work_dir.mkdir(parents=True, exist_ok=True)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    if device != "cuda": raise SystemExit("A CUDA GPU is required for the full acoustic scan")
    processor = AutoFeatureExtractor.from_pretrained(MODEL_ID)
    model = AutoModelForAudioClassification.from_pretrained(MODEL_ID, dtype=torch.float16).to(device).eval()
    labels = [model.config.id2label[i] for i in range(len(model.config.id2label))]
    bs, started = a.batch_size, time.time()
    for source in ("masc", "qasr"):
        for leaf in ("lev", "non_lev"):
            remote_dir = f"{a.remote}/{source}/{leaf}"
            names = sorted(x for x in rclone(["lsf", remote_dir]).splitlines() if x.endswith(".parquet.zst"))
            out_dir = a.output_root / f"{source}_{leaf}"; out_dir.mkdir(exist_ok=True)
            ledger_path, output = out_dir / "done_shards.json", out_dir / "row_probabilities.jsonl"
            done = set(json.loads(ledger_path.read_text())) if ledger_path.exists() else set()
            print(f"{source}/{leaf}: {len(done)}/{len(names)} shards complete", flush=True)
            # The ledger commits a whole shard at a time, but rows are appended as
            # they are classified -- so a kill/OOM part-way through a shard leaves
            # that shard's rows in the JSONL while the ledger still calls it
            # pending. Resuming would then re-append every one of them, silently
            # double-counting the speaker hours that drive the split (and possibly
            # leaving a half-written final line that json.loads chokes on
            # downstream). Drop anything not attributable to a committed shard
            # before appending; committed shards are never rewritten.
            # Deliberately conservative: a row is dropped only when it NAMES a shard
            # that is not committed. A row carrying no source_file at all predates
            # that field and cannot be attributed to any shard, so it is kept --
            # dropping those instead cost a full masc re-scan once, because an older
            # version of this script emitted source_file on skipped/error rows but
            # not on ok ones, and "unattributable" silently meant "delete".
            if output.exists():
                kept_lines, dropped = [], 0
                with output.open(encoding="utf-8") as f:
                    for line in f:
                        if not line.strip(): continue
                        try: src = json.loads(line).get("source_file")
                        except Exception: src = None; keep = False
                        else: keep = src is None or src in done
                        if keep: kept_lines.append(line if line.endswith("\n") else line + "\n")
                        else: dropped += 1
                if dropped:
                    output.write_text("".join(kept_lines), encoding="utf-8")
                    print(f"{source}/{leaf}: dropped {dropped} rows from uncommitted shards", flush=True)
            with output.open("a", encoding="utf-8") as f:
                for ordinal, name in enumerate(names, 1):
                    if name in done: continue
                    zst, parquet = a.work_dir / "shard.parquet.zst", a.work_dir / "shard.parquet"
                    rclone(["copy", f"{remote_dir}/{name}", str(a.work_dir)])
                    (a.work_dir / name).replace(zst)
                    subprocess.run(["unzstd", "-f", "-o", str(parquet), str(zst)], check=True); zst.unlink()
                    schema = pq.ParquetFile(parquet).schema_arrow
                    text_col, id_col = get_columns(schema.names, source)
                    has_sr = "sampling_rate" in schema.names
                    cols = [text_col, "duration", "audio", id_col] + (["sampling_rate"] if has_sr else [])
                    d = pq.read_table(parquet, columns=cols).to_pydict()
                    waves, kept = [], []
                    for row in range(len(d[id_col])):
                        uid = d[id_col][row]
                        try: wave = decode(d["audio"][row], d["sampling_rate"][row] if has_sr else None)
                        except Exception as exc:
                            f.write(json.dumps({"source": source, "source_file": name, "row_idx": row, "uid": uid, "status": "error", "error": str(exc)}, ensure_ascii=False)+"\n"); continue
                        # Some audio cells are corrupted (bytes that misparse as a
                        # compressed format libsndfile then tries to resync through,
                        # e.g. "Illegal Audio-MPEG-Header" noise) and decode to a
                        # wildly wrong length -- far too short (crashes the model's
                        # conv kernel) or far too long (a multi-minute garbage decode
                        # OOMs the GPU regardless of batch size, since a single such
                        # sample alone can exceed capacity -- this is what crashed
                        # qasr/non_lev shard 5 repeatedly). Cross-check the decoded
                        # length against the row's own duration metadata (generous
                        # 3x tolerance for legitimate resampling rounding) and hard-cap
                        # at 60s regardless of metadata, which never happens for real
                        # short-utterance ASR segments in this corpus.
                        actual_dur = len(wave) / TARGET_SR
                        meta_dur = float(d["duration"][row] or 0.0)
                        if actual_dur > 60.0 or (meta_dur > 0 and (actual_dur > meta_dur * 3 + 2 or actual_dur < meta_dur / 3 - 0.5)):
                            f.write(json.dumps({"source": source, "source_file": name, "row_idx": row, "uid": uid, "status": "skipped_corrupt", "duration_sec": meta_dur, "decoded_sec": actual_dur}, ensure_ascii=False)+"\n"); continue
                        if len(wave) < TARGET_SR:
                            f.write(json.dumps({"source": source, "source_file": name, "row_idx": row, "uid": uid, "status": "skipped_short", "duration_sec": d["duration"][row]}, ensure_ascii=False)+"\n"); continue
                        waves.append(wave); kept.append(row)
                    probs, bs = classify(waves, processor, model, device, bs)
                    for prob, row in zip(probs, kept):
                        top = int(np.argmax(prob)); uid = str(d[id_col][row])
                        f.write(json.dumps({"source": "masc_c" if source == "masc" else "qasr", "source_file": name, "row_idx": row, "uid": f"masc_c:{uid}" if source == "masc" else uid, "speaker_key": uid if source == "masc" else uid.split(":")[1], "text": d[text_col][row] or "", "duration_sec": d["duration"][row], "status": "ok", "top_label": labels[top], "top_score": float(prob[top]), "label_scores": {labels[i]: float(x) for i,x in enumerate(prob)}}, ensure_ascii=False)+"\n")
                    f.flush(); parquet.unlink(); done.add(name); ledger_path.write_text(json.dumps(sorted(done)))
                    print(f"{source}/{leaf} {ordinal}/{len(names)}: {len(kept)} classified, bs={bs}, elapsed={(time.time()-started)/60:.1f}m", flush=True)
    print("ALL DONE", flush=True)

if __name__ == "__main__": main()
