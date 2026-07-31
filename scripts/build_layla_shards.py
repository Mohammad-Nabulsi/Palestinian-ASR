#!/usr/bin/env python3
"""Build Layla parquet training shards from the raw Layla dataset.

This is the re-runnable replacement for the shard-writing half of
`scripts/finalize_data_with_layla.py`. That script read four hand-merged
`normalized_*.json` files (the manually-prompted phonetic-artifact
normalization pass) which no longer exist on this box, and it wrote straight
into `data/`, which does not exist yet at this stage of the pipeline.

Differences from the original:

- Transcript text is read from the `*_Arabic_transcription.docx` sources
  directly (case-insensitive; two Beduin_center files use a lowercase
  `_arabic_transcription` suffix that the original path derivation missed).
- If the merged normalization JSONs are supplied via `--normalized-json`, the
  `normalized` text wins over the raw docx text on a per-`source` basis, so
  this script keeps producing the original output once those files come back.
- `gender` is filled from the speaker-directory code (trailing M/F), matching
  the column present in `data_cleaned_text_merged_v1/clean/`.
- Output goes to a standalone shard root instead of `data/`.
"""
from __future__ import annotations

import argparse
import json
import re
import wave
import zipfile
from pathlib import Path
from xml.etree import ElementTree

import pyarrow as pa
import pyarrow.parquet as pq

WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
TRANSCRIPT_SUFFIX_RE = re.compile(r"_arabic_transcription$", re.IGNORECASE)
SPEAKER_CODE_RE = re.compile(r"^([A-Za-z]+)(\d+)([MF])$")

DEFAULT_DATASET_ROOT = Path(
    "/workspace/asr/Palestinian-ASR/Layla/Layla Witheeb Jordanian Arabic Acoustic Dataset"
)
DEFAULT_OUTPUT_ROOT = Path("/workspace/asr/Palestinian-ASR/processed_layla_shards_v1")
DEFAULT_NUM_SHARDS = 4


def extract_docx_text(docx_path: Path) -> str:
    """Same paragraph-join extraction as scripts/stage_raw_datasets.py."""
    with zipfile.ZipFile(docx_path) as docx_zip:
        xml_bytes = docx_zip.read("word/document.xml")
    root = ElementTree.fromstring(xml_bytes)
    paragraphs = []
    for para in root.iter(f"{WORD_NAMESPACE}p"):
        parts = [node.text or "" for node in para.iter(f"{WORD_NAMESPACE}t")]
        if parts:
            paragraphs.append("".join(parts))
    return "\n".join(paragraphs).strip() + "\n"


def read_wav_bytes_and_duration(path: Path) -> tuple[bytes, float]:
    audio_bytes = path.read_bytes()
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    return audio_bytes, (frames / rate if rate else 0.0)


def derive_audio_path(docx_path: Path) -> Path:
    """Resolve the recording paired with a transcript docx.

    The transcript is `<stem>_Arabic_transcription.docx`; the audio is
    `<stem>.WAV` or `<stem>.wav` in the same directory.
    """
    stem = TRANSCRIPT_SUFFIX_RE.sub("", docx_path.stem)
    base = docx_path.with_name(stem)
    for candidate in (base.with_suffix(".WAV"), base.with_suffix(".wav")):
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No audio file found for {docx_path}")


def derive_gender(docx_path: Path, dataset_root: Path) -> str:
    """Speaker directory codes end in M/F, e.g. Amman/AB5F, Karak/MK10M."""
    rel = docx_path.relative_to(dataset_root)
    for part in rel.parts:
        match = SPEAKER_CODE_RE.match(part)
        if match:
            return "female" if match.group(3).upper() == "F" else "male"
    return ""


def staged_source_value(docx_path: Path, dataset_root: Path) -> str:
    """`./<rel path>` with a .txt suffix, matching the staged-transcript form
    that the original merged JSONs used in their `source` field."""
    rel = docx_path.relative_to(dataset_root).with_suffix(".txt")
    return f"./{rel.as_posix()}"


def load_normalized_map(json_paths: list[Path]) -> dict[str, str]:
    """Map `source` -> `normalized` from the hand-merged normalization JSONs."""
    mapping: dict[str, str] = {}
    for json_path in json_paths:
        obj = json.loads(json_path.read_text(encoding="utf-8"))
        for record in obj.get("records", []):
            source = record.get("source")
            normalized = record.get("normalized")
            if source and normalized:
                mapping[source.replace("./", "")] = normalized
    return mapping


def build_rows(dataset_root: Path, normalized_map: dict[str, str]) -> list[dict]:
    docx_paths = sorted(
        p for p in dataset_root.rglob("*.docx") if TRANSCRIPT_SUFFIX_RE.search(p.stem)
    )
    if not docx_paths:
        raise SystemExit(f"No *_Arabic_transcription.docx found under {dataset_root}")

    rows = []
    normalized_hits = 0
    for docx_path in docx_paths:
        audio_path = derive_audio_path(docx_path)
        audio_bytes, duration = read_wav_bytes_and_duration(audio_path)
        source_value = staged_source_value(docx_path, dataset_root)
        key = source_value.replace("./", "")

        raw_text = extract_docx_text(docx_path).strip()
        normalized_text = normalized_map.get(key)
        if normalized_text:
            normalized_hits += 1
        transcription = (normalized_text or raw_text).strip()

        rel_no_suffix = TRANSCRIPT_SUFFIX_RE.sub(
            "", docx_path.relative_to(dataset_root).with_suffix("").as_posix()
        )
        rows.append(
            {
                "audio": {"bytes": audio_bytes, "path": str(audio_path)},
                "seg_id": "layla_" + rel_no_suffix.replace("/", "_"),
                "transcription": transcription,
                "gender": derive_gender(docx_path, dataset_root),
                "duration": float(duration),
                "source_file": source_value,
                "transcript_source": "normalized_json" if normalized_text else "raw_docx",
            }
        )

    print(f"Built {len(rows)} rows ({normalized_hits} using normalized JSON text)")
    return rows


def write_shards(rows: list[dict], output_root: Path, num_shards: int) -> list[Path]:
    # Nested under a `layla/` dir so the cleaning pass's dataset_from_file()
    # fallback (first path part below its INPUT_ROOT) labels these rows "layla"
    # in the reports rather than the shard-root directory name.
    train_dir = output_root / "layla"
    train_dir.mkdir(parents=True, exist_ok=True)

    # Even split, remainder spread across the leading shards, mirroring the
    # original four-way `layla__data-0000X-of-00004.parquet` layout.
    base, extra = divmod(len(rows), num_shards)
    written = []
    start = 0
    for idx in range(num_shards):
        size = base + (1 if idx < extra else 0)
        chunk = rows[start : start + size]
        start += size
        out_path = train_dir / f"layla__data-{idx:05d}-of-{num_shards:05d}.parquet"
        pq.write_table(pa.Table.from_pylist(chunk), out_path, compression="zstd")
        print(f"  {out_path.name}: {len(chunk)} rows, {out_path.stat().st_size / 1e6:.1f} MB")
        written.append(out_path)
    return written


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-root", type=Path, default=DEFAULT_DATASET_ROOT)
    parser.add_argument("--output-root", type=Path, default=DEFAULT_OUTPUT_ROOT)
    parser.add_argument("--num-shards", type=int, default=DEFAULT_NUM_SHARDS)
    parser.add_argument(
        "--normalized-json",
        type=Path,
        nargs="*",
        default=[],
        help="Merged normalized_*.json files; their `normalized` text overrides docx text.",
    )
    args = parser.parse_args()

    if not args.dataset_root.is_dir():
        raise SystemExit(f"Missing Layla dataset root: {args.dataset_root}")

    normalized_map = load_normalized_map(args.normalized_json)
    if args.normalized_json:
        print(f"Loaded {len(normalized_map)} normalized records")

    rows = build_rows(args.dataset_root, normalized_map)
    written = write_shards(rows, args.output_root, args.num_shards)

    total_duration = sum(r["duration"] for r in rows)
    manifest = {
        "script": str(Path(__file__).resolve()),
        "dataset_root": str(args.dataset_root),
        "output_root": str(args.output_root),
        "normalized_json_inputs": [str(p) for p in args.normalized_json],
        "rows": len(rows),
        "rows_from_normalized_json": sum(
            1 for r in rows if r["transcript_source"] == "normalized_json"
        ),
        "rows_from_raw_docx": sum(1 for r in rows if r["transcript_source"] == "raw_docx"),
        "total_duration_sec": total_duration,
        "total_duration_hours": total_duration / 3600.0,
        "shards": [p.name for p in written],
    }
    reports_dir = args.output_root / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    manifest_path = reports_dir / "layla_shard_manifest.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")

    print(f"Wrote {len(written)} shards to {args.output_root}")
    print(f"Total audio: {total_duration / 3600.0:.2f} h")
    print(f"Manifest: {manifest_path}")


if __name__ == "__main__":
    main()
