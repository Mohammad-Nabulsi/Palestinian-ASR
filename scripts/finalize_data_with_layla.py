#!/usr/bin/env python3
from __future__ import annotations

import json
import os
import shutil
import wave
from pathlib import Path

import pyarrow as pa
import pyarrow.parquet as pq

ROOT = Path("/home/MohammadNabulsi/whisper")
DATA_ROOT = ROOT / "data"
LAYLA_ROOT = ROOT / "Layla"
LAYLA_DATASET_ROOT = LAYLA_ROOT / "Layla Witheeb Jordanian Arabic Acoustic Dataset"
LAYLA_JSONS = [
    LAYLA_ROOT / "normalized_output_appended.json",
    LAYLA_ROOT / "normalized_layla_batch_130_appended_131.json",
    LAYLA_ROOT / "normalized_pasted_132_133_appended.json",
    LAYLA_ROOT / "normalized_pasted_text_134_135_136_appended.json",
]
INTERMEDIATE_LAYLA_ROOT = ROOT / ".intermediate_data" / "Layla"
DOC_PATH = ROOT / "DATA_CURATION.md"
DOC_MARKER = "## Related Utility Scripts"
KEEP_GENERATED_REPORTS = {"omnilingual_final_kept_manifest.json"}


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def read_wav_bytes_and_duration(path: Path) -> tuple[bytes, float]:
    audio_bytes = path.read_bytes()
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    duration = frames / rate if rate else 0.0
    return audio_bytes, duration


def derive_audio_path(source_value: str) -> Path:
    rel = source_value.replace("./", "")
    txt_path = LAYLA_DATASET_ROOT / rel
    stem = txt_path.name.replace("_Arabic_transcription.txt", "")
    base = txt_path.with_name(stem)
    candidates = [
        base.with_suffix(".WAV"),
        base.with_suffix(".wav"),
    ]
    for candidate in candidates:
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"No audio file found for {source_value}")


def build_layla_rows(json_path: Path) -> list[dict]:
    obj = json.loads(json_path.read_text(encoding="utf-8"))
    rows = []
    records = obj.get("records", [])
    for idx, record in enumerate(records):
        audio_path = derive_audio_path(record["source"])
        audio_bytes, duration = read_wav_bytes_and_duration(audio_path)
        rows.append(
            {
                "audio": {"bytes": audio_bytes, "path": str(audio_path)},
                "seg_id": f"layla_{json_path.stem}_{idx:05d}",
                "transcription": record["normalized"],
                "duration": float(duration),
                "source_file": record["source"],
            }
        )
    return rows


def write_layla_shards() -> list[Path]:
    written = []
    total = len(LAYLA_JSONS)
    for idx, json_path in enumerate(LAYLA_JSONS):
        rows = build_layla_rows(json_path)
        file_name = f"layla__data-{idx:05d}-of-{total:05d}.parquet"
        out_path = DATA_ROOT / file_name
        table = pa.Table.from_pylist(rows)
        pq.write_table(table, out_path, compression="zstd")
        written.append(out_path)
    return written


def flatten_clean_into_data() -> list[Path]:
    clean_dir = DATA_ROOT / "clean"
    if not clean_dir.exists():
        return []
    moved = []
    for path in sorted(clean_dir.glob("*.parquet")):
        dest = DATA_ROOT / path.name
        if dest.exists():
            raise SystemExit(f"Refusing to overwrite existing file: {dest}")
        shutil.move(str(path), str(dest))
        moved.append(dest)
    clean_dir.rmdir()
    return moved


def prune_stale_reports() -> None:
    generated_dir = DATA_ROOT / "reports" / "generated"
    if generated_dir.exists():
        for path in generated_dir.iterdir():
            if path.is_file() and path.name not in KEEP_GENERATED_REPORTS:
                path.unlink()


def move_layla_to_intermediate() -> None:
    if INTERMEDIATE_LAYLA_ROOT.exists():
        shutil.rmtree(INTERMEDIATE_LAYLA_ROOT)
    shutil.move(str(LAYLA_ROOT), str(INTERMEDIATE_LAYLA_ROOT))


def update_data_curation(layla_shards: list[Path], moved_clean_files: list[Path]) -> None:
    text = DOC_PATH.read_text(encoding="utf-8")
    section = f"""
## Layla Prompt Merge and Sharding

For the Layla normalization pass, the prompted outputs were manually uploaded and merged into these four JSON files:

- `normalized_output_appended.json`
- `normalized_layla_batch_130_appended_131.json`
- `normalized_pasted_132_133_appended.json`
- `normalized_pasted_text_134_135_136_appended.json`

These JSON files contain the normalized text under `normalized`, and that text was written into the training column named `transcription`.

The Layla sharding step then:

1. Read the four merged JSON files.
2. Matched each JSON `source` entry to the corresponding Layla audio file by removing `_Arabic_transcription` from the transcript filename and resolving the paired `.WAV` or `.wav`.
3. Built Parquet training shards with:
   - `audio`
   - `seg_id`
   - `transcription`
   - `duration`
   - `source_file`
4. Wrote the Layla shards directly into `data/`.
5. Moved the original `Layla/` source directory into:
   - `.intermediate_data/Layla/`

Layla shards written into `data/`:

{os.linesep.join(f"- `{p.name}`" for p in layla_shards)}

## Flattening `data/clean` Into `data/`

After preparing the working dataset, the shard files under `data/clean/` were unpacked into `data/` directly so the training shards now live at the top level of `data/`.

Moved shard files from `data/clean/` into `data/`:

{os.linesep.join(f"- `{p.name}`" for p in moved_clean_files[:12])}
{"- `...`" if len(moved_clean_files) > 12 else ""}
""".strip()
    if "## Layla Prompt Merge and Sharding" in text:
        start = text.index("## Layla Prompt Merge and Sharding")
        end = text.index(DOC_MARKER, start)
        updated = text[:start] + section + "\n\n" + text[end:]
    else:
        updated = text.replace(DOC_MARKER, section + "\n\n" + DOC_MARKER, 1)
    DOC_PATH.write_text(updated, encoding="utf-8")


def main() -> None:
    ensure_dir(DATA_ROOT)
    prune_stale_reports()
    moved_clean_files = flatten_clean_into_data()
    layla_shards = write_layla_shards()
    update_data_curation(layla_shards, moved_clean_files)
    move_layla_to_intermediate()
    print(f"Flattened clean shards into {DATA_ROOT}")
    print(f"Wrote {len(layla_shards)} Layla shards into {DATA_ROOT}")
    print(f"Moved Layla to {INTERMEDIATE_LAYLA_ROOT}")


if __name__ == "__main__":
    main()
