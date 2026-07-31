#!/usr/bin/env python
"""Segment audio samples longer than 30s into <=28s segments.

Pipeline (offline, one-time preprocessing):
  1. Silero VAD -> speech regions / silence gaps
  2. WhisperX forced alignment of the existing transcript (wav2vec2 Arabic)
  3. Cut-point selection at VAD silence gaps, accumulating ~<=28s of audio
  4. Transcript slicing by word-midpoint assignment
  5. Quality gate on alignment scores (drop, don't down-weight)
  6. Output: new parquet rows under segmented/v1/ with full provenance.
     Raw corpus under data/ is never touched.

Input: segmented/flagged_over30s.json (produced by the duration scan).
Resumable: input files already listed in segmented/v1/checkpoint.txt are skipped.
"""
import argparse
import gc
import io
import json
import os
import sys
import tempfile
import time
import traceback
from collections import defaultdict

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf
import torch

SR = 16000
TARGET_MAX_S = 28.0          # headroom under Whisper's 30s ceiling
MIN_SEG_S = 0.5              # drop degenerate slivers
SCORE_DROP_THRESHOLD = 0.45  # mean word alignment score below this -> drop segment
EDGE_PAD_S = 0.05            # pad cut points into silence, away from speech

DATA_DIR = "/root/Palestinian-ASR/data"
OUT_DIR = "/root/Palestinian-ASR/segmented/v1"

TEXT_COLS = {  # group -> (primary normalized col, raw fallback col)
    "layla": ("manual_normalized_transcript", "transcription"),
    "omnilingual_apc": ("manual_normalized_transcript", "raw_text"),
    "masc_c_only": ("manual_normalized_transcript", "text"),
}
ID_COLS = {"layla": "seg_id", "omnilingual_apc": "segment_id", "masc_c_only": "video_id"}


def log(msg):
    print(f"[{time.strftime('%H:%M:%S')}] {msg}", flush=True)


def decode_audio(audio_struct):
    """Decode HF audio struct {bytes, path} -> mono float32 @16k."""
    raw = audio_struct["bytes"]
    try:
        wav, sr = sf.read(io.BytesIO(raw), dtype="float32", always_2d=True)
        wav = wav.mean(axis=1)
    except Exception:
        # fall back to ffmpeg via whisperx for exotic codecs
        import whisperx
        suffix = os.path.splitext(audio_struct.get("path") or "x.bin")[1] or ".bin"
        with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
            f.write(raw)
            tmp = f.name
        try:
            wav = whisperx.load_audio(tmp, SR)
            sr = SR
        finally:
            os.unlink(tmp)
    if sr != SR:
        import torchaudio.functional as AF
        wav = AF.resample(torch.from_numpy(wav), sr, SR).numpy()
    return np.ascontiguousarray(wav, dtype=np.float32)


def encode_wav(wav):
    buf = io.BytesIO()
    sf.write(buf, wav, SR, format="WAV", subtype="PCM_16")
    return buf.getvalue()


def pick_cut_points(speech_regions, total_dur, word_ends):
    """Greedy forward walk: cut at the silence gap nearest to (but not past)
    start+TARGET_MAX_S. Never cut inside a VAD speech region; fall back to the
    largest inter-word pause in the window if VAD offers no gap."""
    gaps = []  # candidate cut instants (midpoints of silence gaps)
    prev_end = 0.0
    for s, e in speech_regions:
        if s - prev_end > 0.02:
            gaps.append((prev_end + s) / 2.0)
        prev_end = max(prev_end, e)
    cuts = []
    start = 0.0
    while total_dur - start > TARGET_MAX_S:
        limit = start + TARGET_MAX_S
        candidates = [g for g in gaps if start + MIN_SEG_S < g <= limit]
        if candidates:
            cut = max(candidates)
        else:
            # no VAD gap in window: cut at the largest pause between aligned words
            wcands = [t for t in word_ends if start + MIN_SEG_S < t <= limit]
            cut = (max(wcands) + EDGE_PAD_S) if wcands else limit
        if cut <= start + MIN_SEG_S:
            cut = limit
        cuts.append(min(cut, total_dur))
        start = cuts[-1]
    return cuts


def align_words(model, metadata, wav, text, device):
    import whisperx
    seg = [{"start": 0.0, "end": len(wav) / SR, "text": text}]
    res = whisperx.align(seg, model, metadata, wav, device, return_char_alignments=False)
    words = []
    for s in res["segments"]:
        words.extend(s.get("words", []))
    return words


def slice_sample(wav, text, words, speech_regions):
    """Return list of segment dicts for one flagged sample."""
    total_dur = len(wav) / SR
    word_ends = [w["end"] for w in words if "end" in w]
    cuts = pick_cut_points(speech_regions, total_dur, word_ends)
    bounds = [0.0] + cuts + [total_dur]

    # assign each transcript word to a segment by its timestamp midpoint;
    # words the aligner couldn't time follow their previous word
    text_words = text.split()
    n_seg = len(bounds) - 1
    seg_words = [[] for _ in range(n_seg)]
    seg_scores = [[] for _ in range(n_seg)]
    cur = 0
    for i, tw in enumerate(text_words):
        w = words[i] if i < len(words) else {}
        if "start" in w and "end" in w:
            mid = (w["start"] + w["end"]) / 2.0
            cur = min(int(np.searchsorted(bounds, mid, side="right")) - 1, n_seg - 1)
            cur = max(cur, 0)
            if "score" in w and w["score"] is not None:
                seg_scores[cur].append(w["score"])
        seg_words[cur].append(tw)

    out = []
    for k in range(n_seg):
        s0, s1 = bounds[k], bounds[k + 1]
        dur = s1 - s0
        txt = " ".join(seg_words[k]).strip()
        score = float(np.mean(seg_scores[k])) if seg_scores[k] else None
        timed_frac = len(seg_scores[k]) / max(len(seg_words[k]), 1)
        keep = bool(txt) and dur >= MIN_SEG_S
        reason = None
        if not txt:
            reason = "empty_text"
        elif dur < MIN_SEG_S:
            reason = "too_short"
        elif score is not None and score < SCORE_DROP_THRESHOLD:
            keep, reason = False, "low_align_score"
        elif score is None or timed_frac < 0.5:
            keep, reason = False, "untimed_words"
        out.append({
            "segment_idx": k, "start_offset": round(s0, 3), "end_offset": round(s1, 3),
            "duration": round(dur, 3), "text": txt, "align_score": score,
            "n_words": len(seg_words[k]), "keep": keep, "drop_reason": reason,
            "audio_slice": wav[int(s0 * SR):int(s1 * SR)] if keep else None,
        })
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit-samples", type=int, default=None, help="smoke test: stop after N samples")
    ap.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    args = ap.parse_args()
    device = args.device

    os.makedirs(OUT_DIR, exist_ok=True)
    with open("/root/Palestinian-ASR/segmented/flagged_over30s.json") as f:
        flagged = json.load(f)
    by_file = defaultdict(list)
    for r in flagged:
        by_file[r["file"]].append(r)

    ckpt_path = os.path.join(OUT_DIR, "checkpoint.txt")
    done = set()
    if os.path.exists(ckpt_path):
        done = set(open(ckpt_path).read().split())

    log(f"loading models on {device} ...")
    import whisperx
    from silero_vad import load_silero_vad, get_speech_timestamps
    align_model, align_meta = whisperx.load_align_model(language_code="ar", device=device)
    vad_model = load_silero_vad()
    log("models loaded")

    report = defaultdict(lambda: {"samples": 0, "segments_kept": 0, "segments_dropped": 0,
                                  "hours_in": 0.0, "hours_kept": 0.0, "errors": 0})
    drop_log, err_log = [], []
    n_samples = 0

    for fname in sorted(by_file):
        if fname in done:
            log(f"skip (done): {fname}")
            continue
        rows = by_file[fname]
        grp = rows[0]["group"]
        text_col, raw_col = TEXT_COLS[grp]
        id_col = ID_COLS[grp]
        path = os.path.join(DATA_DIR, "clean", fname)
        if not os.path.exists(path):
            path = os.path.join(DATA_DIR, fname)
        table = pq.read_table(path)
        out_rows = []
        for r in rows:
            if args.limit_samples and n_samples >= args.limit_samples:
                break
            n_samples += 1
            idx = r["row_index"]
            row = table.slice(idx, 1).to_pylist()[0]
            text = (row.get(text_col) or row.get(raw_col) or "").strip()
            rep = report[grp]
            try:
                wav = decode_audio(row["audio"])
                dur = len(wav) / SR
                rep["samples"] += 1
                rep["hours_in"] += dur / 3600
                if not text:
                    raise ValueError("empty transcript")
                st = get_speech_timestamps(torch.from_numpy(wav), vad_model,
                                           sampling_rate=SR, return_seconds=True)
                speech_regions = [(s["start"], s["end"]) for s in st]
                words = align_words(align_model, align_meta, wav, text, device)
                segs = slice_sample(wav, text, words, speech_regions)
            except Exception as e:
                rep["errors"] += 1
                err_log.append({"file": fname, "row_index": idx, "error": repr(e)})
                log(f"ERROR {fname}[{idx}]: {e!r}")
                traceback.print_exc()
                continue
            orig_id = str(row.get(id_col) or f"{fname}:{idx}")
            for s in segs:
                if s["keep"]:
                    rep["segments_kept"] += 1
                    rep["hours_kept"] += s["duration"] / 3600
                    out_rows.append({
                        "audio": {"bytes": encode_wav(s["audio_slice"]), "path": None},
                        "text": s["text"], "group": grp, "orig_file": fname,
                        "orig_row_index": idx, "orig_id": orig_id,
                        "segment_idx": s["segment_idx"], "start_offset": s["start_offset"],
                        "end_offset": s["end_offset"], "duration": s["duration"],
                        "align_score": s["align_score"], "n_words": s["n_words"],
                    })
                else:
                    rep["segments_dropped"] += 1
                    drop_log.append({"file": fname, "row_index": idx, "orig_id": orig_id,
                                     **{k: s[k] for k in ("segment_idx", "start_offset",
                                        "end_offset", "duration", "align_score",
                                        "n_words", "drop_reason", "text")}})
            log(f"{fname}[{idx}] {dur:.1f}s -> {sum(1 for s in segs if s['keep'])} kept"
                f" / {sum(1 for s in segs if not s['keep'])} dropped")
            del wav
        if out_rows:
            out_name = fname.replace(".parquet", "") + "__segmented.parquet"
            schema = pa.schema([
                ("audio", pa.struct([("bytes", pa.binary()), ("path", pa.string())])),
                ("text", pa.string()), ("group", pa.string()), ("orig_file", pa.string()),
                ("orig_row_index", pa.int64()), ("orig_id", pa.string()),
                ("segment_idx", pa.int64()), ("start_offset", pa.float64()),
                ("end_offset", pa.float64()), ("duration", pa.float64()),
                ("align_score", pa.float64()), ("n_words", pa.int64()),
            ])
            pq.write_table(pa.Table.from_pylist(out_rows, schema=schema),
                           os.path.join(OUT_DIR, out_name))
        if not (args.limit_samples and n_samples >= args.limit_samples):
            with open(ckpt_path, "a") as f:
                f.write(fname + "\n")
        del table
        gc.collect()
        if args.limit_samples and n_samples >= args.limit_samples:
            break

    with open(os.path.join(OUT_DIR, "segmentation_report.json"), "w") as f:
        json.dump({"config": {"target_max_s": TARGET_MAX_S, "min_seg_s": MIN_SEG_S,
                              "score_drop_threshold": SCORE_DROP_THRESHOLD,
                              "align_model": "whisperx default ar (wav2vec2)",
                              "vad": "silero-vad"},
                   "groups": report, "errors": err_log}, f, ensure_ascii=False, indent=1)
    with open(os.path.join(OUT_DIR, "dropped_segments.json"), "w") as f:
        json.dump(drop_log, f, ensure_ascii=False, indent=1)
    log("DONE")
    for g, s in report.items():
        log(f"  {g}: {s}")


if __name__ == "__main__":
    main()
