import json
import random
import wave
from contextlib import closing
from io import BytesIO
from pathlib import Path

import pyarrow.ipc as ipc
import pyarrow.parquet as pq


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "segments"
QASR_DIR = ROOT / "processed_qasr_segments" / "train"
MASC_DIR = ROOT / "MASC-Arabic2" / "data"
SAMPLES_PER_DATASET = 20
RNG_SEED = 42


def safe_audio_metadata(audio_bytes):
    meta = {"byte_length": len(audio_bytes)}
    try:
        with closing(wave.open(BytesIO(audio_bytes), "rb")) as wav_file:
            frame_rate = wav_file.getframerate()
            num_frames = wav_file.getnframes()
            meta.update(
                {
                    "channels": wav_file.getnchannels(),
                    "sample_width": wav_file.getsampwidth(),
                    "sample_rate": frame_rate,
                    "num_frames": num_frames,
                    "duration_seconds": num_frames / frame_rate if frame_rate else None,
                }
            )
    except wave.Error:
        meta["wave_parse_error"] = "unrecognized_wav_format"
    return meta


def wrap_pcm16le_as_wav(audio_bytes, sample_rate, channels=1, sample_width=2):
    buffer = BytesIO()
    with closing(wave.open(buffer, "wb")) as wav_file:
        wav_file.setnchannels(channels)
        wav_file.setsampwidth(sample_width)
        wav_file.setframerate(sample_rate)
        wav_file.writeframes(audio_bytes)
    return buffer.getvalue()


def serialize_value(value):
    if isinstance(value, bytes):
        return f"<bytes:{len(value)}>"
    if isinstance(value, dict):
        return {k: serialize_value(v) for k, v in value.items()}
    if isinstance(value, list):
        return [serialize_value(v) for v in value]
    return value


def random_rows_arrow(file_paths, sample_count, rng):
    chosen_files = rng.sample(file_paths, k=min(sample_count, len(file_paths)))
    rows = []
    for path in chosen_files:
        reader = ipc.open_file(str(path))
        table = reader.read_all()
        row_index = rng.randrange(table.num_rows)
        rows.append((path, row_index, table.slice(row_index, 1).to_pylist()[0]))
    return rows


def random_rows_parquet(file_paths, sample_count, rng):
    chosen_files = rng.sample(file_paths, k=min(sample_count, len(file_paths)))
    rows = []
    for path in chosen_files:
        table = pq.read_table(str(path))
        row_index = rng.randrange(table.num_rows)
        rows.append((path, row_index, table.slice(row_index, 1).to_pylist()[0]))
    return rows


def export_qasr(rows):
    manifest = []
    for idx, (path, row_index, row) in enumerate(rows, start=1):
        audio_bytes = row["audio"]
        wav_bytes = wrap_pcm16le_as_wav(audio_bytes, row["sampling_rate"])
        out_name = f"qasr_{idx:02d}.wav"
        out_path = OUTPUT_DIR / out_name
        out_path.write_bytes(wav_bytes)

        row_meta = {k: serialize_value(v) for k, v in row.items() if k != "audio"}
        parsed_metadata_json = None
        if isinstance(row.get("metadata_json"), str) and row["metadata_json"].strip():
            try:
                parsed_metadata_json = json.loads(row["metadata_json"])
            except json.JSONDecodeError:
                parsed_metadata_json = row["metadata_json"]

        manifest.append(
            {
                "exported_file": out_name,
                "dataset": "processed_qasr_segments/train",
                "source_shard": str(path.relative_to(ROOT)),
                "source_row_index_in_sampled_shard": row_index,
                "audio_metadata": safe_audio_metadata(wav_bytes),
                "source_audio_encoding": "pcm_s16le_mono_wrapped_as_wav",
                "row_metadata": row_meta,
                "parsed_metadata_json": parsed_metadata_json,
            }
        )
    return manifest


def export_masc(rows):
    manifest = []
    for idx, (path, row_index, row) in enumerate(rows, start=1):
        audio_info = row["audio"] or {}
        audio_bytes = audio_info.get("bytes", b"")
        out_name = f"masc_{idx:02d}.wav"
        out_path = OUTPUT_DIR / out_name
        out_path.write_bytes(audio_bytes)

        row_meta = {k: serialize_value(v) for k, v in row.items() if k != "audio"}
        manifest.append(
            {
                "exported_file": out_name,
                "dataset": "MASC-Arabic2/data",
                "source_shard": str(path.relative_to(ROOT)),
                "source_row_index_in_sampled_shard": row_index,
                "audio_metadata": safe_audio_metadata(audio_bytes),
                "audio_path_field": audio_info.get("path"),
                "row_metadata": row_meta,
            }
        )
    return manifest


def main():
    rng = random.Random(RNG_SEED)
    OUTPUT_DIR.mkdir(exist_ok=True)

    qasr_files = sorted(QASR_DIR.glob("*.arrow"))
    masc_files = sorted(MASC_DIR.glob("*.parquet"))

    qasr_rows = random_rows_arrow(qasr_files, SAMPLES_PER_DATASET, rng)
    masc_rows = random_rows_parquet(masc_files, SAMPLES_PER_DATASET, rng)

    manifest = {
        "seed": RNG_SEED,
        "samples_per_dataset": SAMPLES_PER_DATASET,
        "exports": export_qasr(qasr_rows) + export_masc(masc_rows),
    }

    manifest_path = OUTPUT_DIR / "metadata.json"
    manifest_path.write_text(json.dumps(manifest, ensure_ascii=False, indent=2))

    print(f"Exported {len(manifest['exports'])} audio files to {OUTPUT_DIR}")
    print(f"Wrote metadata manifest to {manifest_path}")


if __name__ == "__main__":
    main()
