#!/usr/bin/env python3
"""Build a tiny raw-data fixture set so the unified pipeline can run end to end.

Where the real raw source still exists on this box, a genuinely small slice of it is
carved out (so the real parsers are exercised against real bytes):

- QASR      -> 2 recordings, XML trimmed to the segments inside the first N seconds,
               WAV truncated to the same window
- Omnilingual -> the first N rows of one Arrow shard, rewritten as a stream-format shard
- Layla     -> a few audio files truncated to N seconds, plus their transcripts,
               plus a synthetic ``normalized_*.json`` override for one of them

Where the raw source was deleted after the merge (MASC-Arabic2, Casablanca -- see
DATA_CURATION.md's post-merge cleanup section), a small synthetic fixture is generated
with the same column shape. Fixture rows deliberately include English text, digits and
sub-0.5s durations so every drop rule in the clean stage actually fires.

Usage:
    python tools/make_sample_data.py --output .sample_data
"""
from __future__ import annotations

import argparse
import json
import shutil
import wave
from pathlib import Path
from xml.etree import ElementTree

import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
import soundfile as sf

REPO_ROOT = Path(__file__).resolve().parent.parent

QASR_WAV_DIR = REPO_ROOT / "QASR/wav_all/alt/arabic-speech-web/mgb2.1/wav"
QASR_XML_DIR = REPO_ROOT / "QASR/mgb2.1/release/train_20210109/xml"
OMNI_DIR = REPO_ROOT / "omnilingual_selected/apc_north_levantine_all_splits"
LAYLA_DIR = REPO_ROOT / "Layla/Layla Witheeb Jordanian Arabic Acoustic Dataset"


def log(msg: str) -> None:
    print(f"[make_sample_data] {msg}", flush=True)


# ------------------------------------------------------------------ QASR


def carve_qasr(output: Path, n_recordings: int, window_sec: float) -> dict:
    wav_out = output / "QASR" / "wav"
    xml_out = output / "QASR" / "xml"
    wav_out.mkdir(parents=True, exist_ok=True)
    xml_out.mkdir(parents=True, exist_ok=True)

    if not QASR_WAV_DIR.exists() or not QASR_XML_DIR.exists():
        log("QASR source not available, skipping")
        return {"status": "skipped", "recordings": 0}

    picked = 0
    segments_kept = 0
    for xml_path in sorted(QASR_XML_DIR.glob("*.xml")):
        if picked >= n_recordings:
            break
        wav_path = QASR_WAV_DIR / f"{xml_path.stem}.wav"
        if not wav_path.exists():
            continue

        tree = ElementTree.parse(xml_path)
        root = tree.getroot()
        kept_here = 0
        for segments in root.iter("segments"):
            for segment in list(segments):
                try:
                    end = float(segment.get("endtime", "0"))
                except ValueError:
                    end = 0.0
                if end > window_sec or end <= 0.0:
                    segments.remove(segment)
                else:
                    kept_here += 1
        if kept_here == 0:
            continue

        tree.write(xml_out / xml_path.name, encoding="UTF-8", xml_declaration=True)

        info = sf.info(str(wav_path))
        frames = int(window_sec * info.samplerate)
        audio, rate = sf.read(str(wav_path), frames=frames, dtype="int16", always_2d=False)
        sf.write(str(wav_out / wav_path.name), audio, rate, subtype="PCM_16")

        picked += 1
        segments_kept += kept_here
        log(f"QASR {xml_path.stem}: {kept_here} segment(s) within {window_sec}s")

    return {"status": "ok", "recordings": picked, "segments": segments_kept}


# ----------------------------------------------------------- Omnilingual


def carve_omnilingual(output: Path, n_rows: int) -> dict:
    omni_out = output / "omnilingual" / "apc_north_levantine_all_splits"
    omni_out.mkdir(parents=True, exist_ok=True)

    shards = sorted(OMNI_DIR.glob("data-*.arrow")) if OMNI_DIR.exists() else []
    if not shards:
        log("Omnilingual source not available, skipping")
        return {"status": "skipped", "rows": 0}

    with pa.memory_map(str(shards[0]), "r") as source:
        reader = pa.ipc.open_stream(source)
        batches = []
        collected = 0
        for batch in reader:
            take = min(batch.num_rows, n_rows - collected)
            batches.append(batch.slice(0, take))
            collected += take
            if collected >= n_rows:
                break
        table = pa.Table.from_batches(batches)

    # Two crafted rows so the v2 -> v3 difference is actually exercised:
    #  - "recoverable": its only English sits inside a [...] span, so the v2
    #    placeholder pre-check leaves "background" behind (row dropped as English)
    #    but the v3 span pre-check removes the whole span (row recovered).
    #  - "stubborn": bare English that survives both, so still_dropped is non-empty.
    crafted = table.slice(0, 2).to_pylist()
    crafted[0]["raw_text"] = "مرحبا كيف حالك [background noise] اليوم الجو حلو"
    crafted[0]["segment_id"] = "s_recoverable"
    crafted[1]["raw_text"] = "هاي جملة فيها english كلمة برة الأقواس"
    crafted[1]["segment_id"] = "s_stubborn"
    table = pa.concat_tables([table, pa.Table.from_pylist(crafted, schema=table.schema)])

    dest = omni_out / "data-00000-of-00001.arrow"
    with pa.OSFile(str(dest), "wb") as sink:
        with pa.ipc.new_stream(sink, table.schema) as writer:
            writer.write_table(table)

    # A cache-* file the discovery globs must ignore, same as in the real directory.
    (omni_out / "cache-deadbeef.arrow").write_bytes(b"not a real shard")
    log(f"Omnilingual: {table.num_rows} row(s) -> {dest.name}")
    return {"status": "ok", "rows": table.num_rows}


# ----------------------------------------------------------------- Layla


def carve_layla(output: Path, n_files: int, window_sec: float) -> dict:
    layla_out = output / "Layla" / "dataset"
    layla_out.mkdir(parents=True, exist_ok=True)

    if not LAYLA_DIR.exists():
        log("Layla source not available, skipping")
        return {"status": "skipped", "files": 0}

    audio_paths = sorted(
        p for p in LAYLA_DIR.rglob("*") if p.is_file() and p.suffix in {".WAV", ".wav"}
    )[:n_files]
    if not audio_paths:
        log("Layla has no audio, skipping")
        return {"status": "skipped", "files": 0}

    copied = 0
    normalized_records = []
    for audio_path in audio_paths:
        rel = audio_path.relative_to(LAYLA_DIR)
        dest = layla_out / rel
        dest.parent.mkdir(parents=True, exist_ok=True)

        info = sf.info(str(audio_path))
        audio, rate = sf.read(
            str(audio_path), frames=int(window_sec * info.samplerate),
            dtype="int16", always_2d=False,
        )
        if audio.ndim > 1:
            audio = audio[:, 0]
        sf.write(str(dest), audio, rate, subtype="PCM_16")

        base = audio_path.with_suffix("")
        for suffix in (".txt", ".docx"):
            for candidate in (
                base.with_name(base.name + "_Arabic_transcription").with_suffix(suffix),
                base.with_suffix(suffix),
            ):
                if candidate.is_file():
                    shutil.copy2(candidate, layla_out / candidate.relative_to(LAYLA_DIR))
                    break
        copied += 1

        # One synthetic normalized-JSON override, mirroring the real
        # normalized_*.json files described in DATA_CURATION.md.
        if copied == 1:
            normalized_records.append(
                {
                    "source": f"./{base.relative_to(LAYLA_DIR)}_Arabic_transcription.txt",
                    "original": "عليتش وصوتش",
                    "normalized": "عليك وصوتك",
                }
            )

    (output / "Layla" / "normalized_sample_appended.json").write_text(
        json.dumps({"records": normalized_records, "word_conversions": []}, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    log(f"Layla: {copied} audio file(s) truncated to {window_sec}s")
    return {"status": "ok", "files": copied, "overrides": len(normalized_records)}


# ------------------------------------------- synthetic (deleted sources)


def _rows(prefix: str, n: int, extra: dict | None = None) -> list[dict]:
    """Rows covering every clean-stage outcome: keep / english / number / too short."""
    arabic = [
        "مرحبا كيف حالك اليوم",
        "الجو حلو اليوم بالقدس",
        "شو أخبارك يا صاحبي",
        "بدنا نروح على البلد بكرا",
        "هاي التسجيلة قصيرة كتير",
    ]
    out = []
    for i in range(n):
        kind = i % 5
        if kind == 0:
            text, duration = arabic[i % len(arabic)], 3.5
        elif kind == 1:
            text, duration = f"{arabic[i % len(arabic)]} hello there", 4.0
        elif kind == 2:
            text, duration = f"{arabic[i % len(arabic)]} ٢٠٢٥", 2.75
        elif kind == 3:
            text, duration = arabic[i % len(arabic)], 0.25
        else:
            text, duration = f"[laugh] {arabic[i % len(arabic)]}", 5.25
        row = {
            "seg_id": f"{prefix}_{i:04d}",
            "transcription": text,
            "duration": duration,
            "audio": (np.int16(np.sin(np.arange(1600) * 0.05) * 8000)).tobytes(),
            "sampling_rate": 16_000,
        }
        if extra:
            row.update({k: (v[i % len(v)] if isinstance(v, list) else v) for k, v in extra.items()})
        out.append(row)
    return out


def synth_masc(output: Path, n: int) -> dict:
    masc_out = output / "MASC-Arabic2" / "data"
    masc_out.mkdir(parents=True, exist_ok=True)
    rows = _rows("masc", n, extra={"type": ["c", "c", "s", "c", "c"]})
    for shard in range(2):
        chunk = rows[shard::2]
        pq.write_table(
            pa.Table.from_pylist(chunk),
            masc_out / f"train-{shard:05d}-of-00002.parquet",
        )
    kept = sum(1 for r in rows if r["type"] == "c")
    log(f"MASC synthetic: {len(rows)} row(s), {kept} with type=='c'")
    return {"status": "synthetic", "rows": len(rows), "type_c": kept}


def synth_casablanca(output: Path, n: int) -> dict:
    counts = {}
    for country in ("Palestine", "Jordan"):
        dest = output / "casablanca" / "levant" / country
        dest.mkdir(parents=True, exist_ok=True)
        rows = _rows(country.lower(), n, extra={"dialect": country})
        for split in ("validation", "test"):
            pq.write_table(
                pa.Table.from_pylist(rows),
                dest / f"{split}-00000-of-00001.parquet",
            )
        counts[country] = len(rows) * 2
    log(f"Casablanca synthetic: {counts}")
    return {"status": "synthetic", **counts}


# ------------------------------------------------------------------ main


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=REPO_ROOT / ".sample_data")
    parser.add_argument("--qasr-recordings", type=int, default=2)
    parser.add_argument("--qasr-window-sec", type=float, default=60.0)
    parser.add_argument("--omni-rows", type=int, default=8)
    parser.add_argument("--layla-files", type=int, default=3)
    parser.add_argument("--layla-window-sec", type=float, default=3.0)
    parser.add_argument("--synthetic-rows", type=int, default=20)
    parser.add_argument("--keep", action="store_true", help="Do not wipe the output dir first.")
    args = parser.parse_args()

    output = args.output.resolve()
    if output.exists() and not args.keep:
        shutil.rmtree(output)
    output.mkdir(parents=True, exist_ok=True)

    summary = {
        "output": str(output),
        "qasr": carve_qasr(output, args.qasr_recordings, args.qasr_window_sec),
        "omnilingual": carve_omnilingual(output, args.omni_rows),
        "layla": carve_layla(output, args.layla_files, args.layla_window_sec),
        "masc": synth_masc(output, args.synthetic_rows),
        "casablanca": synth_casablanca(output, args.synthetic_rows),
    }

    (output / "sample_manifest.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    total = sum(
        int(p.stat().st_size) for p in output.rglob("*") if p.is_file()
    )
    log(f"done -> {output}  ({total / 1e6:.1f} MB)")
    print(json.dumps(summary, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
