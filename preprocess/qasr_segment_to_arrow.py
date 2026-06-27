#!/usr/bin/env python3
"""Segment QASR WAV files from XML timings into sharded Arrow files.

Audio files are treated as the source of truth. For each WAV file, the script
looks for a same-stem XML file, extracts segment timing/text metadata, reads
only the required audio slices from disk, and writes segment rows into Arrow
shards that rotate near a target size.
"""

from __future__ import annotations

import argparse
import json
import logging
import math
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import numpy as np
import pyarrow as pa
import pyarrow.ipc as pa_ipc
import soundfile as sf


DEFAULT_WAV_DIR = Path(
    "/home/MohammadNabulsi/whisper/QASR/alt/alt/arabic-speech-web/mgb2.1/wav"
)
DEFAULT_XML_DIR = Path(
    "/home/MohammadNabulsi/whisper/QASR/mgb2.1/release/train_20210109/xml"
)
DEFAULT_OUTPUT_DIR = Path("/home/MohammadNabulsi/whisper/processed_qasr_segments")
TARGET_SAMPLE_RATE = 16000
MIN_DURATION_SECONDS = 0.05
PREFERRED_ANNOTATION_ID = "transcript_align"
PUNCT_NO_SPACE_BEFORE = set("،؛:,.!?؟)]}»")
PUNCT_NO_SPACE_AFTER = set("([{'\"«")

ARROW_SCHEMA = pa.schema(
    [
        pa.field("uid", pa.string()),
        pa.field("audio", pa.binary()),
        pa.field("sampling_rate", pa.int32()),
        pa.field("transcript", pa.string()),
        pa.field("normalized_transcript", pa.string()),
        pa.field("duration", pa.float32()),
        pa.field("source", pa.string()),
        pa.field("split", pa.string()),
        pa.field("speaker_id", pa.string()),
        pa.field("gender", pa.string()),
        pa.field("dialect", pa.string()),
        pa.field("language", pa.string()),
        pa.field("task_type", pa.string()),
        pa.field("segment_id", pa.string()),
        pa.field("recording_id", pa.string()),
        pa.field("original_audio_path", pa.string()),
        pa.field("original_annotation_path", pa.string()),
        pa.field("source_file", pa.string()),
        pa.field("source_row_index", pa.int64()),
        pa.field("start_time", pa.float32()),
        pa.field("end_time", pa.float32()),
        pa.field("metadata_json", pa.string()),
    ]
)


class TqdmLoggingHandler(logging.Handler):
    def emit(self, record: logging.LogRecord) -> None:
        try:
            print(self.format(record), flush=True)
        except Exception:
            self.handleError(record)


@dataclass
class SegmentSpec:
    segment_id: str
    transcript: str
    raw_start_time: Optional[float]
    raw_end_time: Optional[float]
    speaker_id: Optional[str]
    speaker_meta: Dict[str, Any]
    segment_meta: Dict[str, Any]
    start_time: Optional[float] = None
    end_time: Optional[float] = None


@dataclass
class ParsedXml:
    recording_id: str
    annotation_id: Optional[str]
    recording_meta: Dict[str, Any]
    speakers: Dict[str, Dict[str, Any]]
    segments: List[SegmentSpec]


@dataclass
class Counters:
    audio_files_seen: int = 0
    audio_files_with_xml: int = 0
    audio_files_without_xml: int = 0
    audio_files_with_segments: int = 0
    malformed_xml_files: int = 0
    empty_transcript_segments: int = 0
    invalid_segments: int = 0
    too_short_segments: int = 0
    emitted_segments: int = 0
    emitted_seconds: float = 0.0


@dataclass
class BuildContext:
    output_dir: Path
    target_shard_bytes: int
    skip_existing: bool
    logger: logging.Logger
    counters: Counters = field(default_factory=Counters)
    shard_index_rows: List[Dict[str, Any]] = field(default_factory=list)
    segment_index_rows: List[Dict[str, Any]] = field(default_factory=list)
    audio_index_rows: List[Dict[str, Any]] = field(default_factory=list)
    missing_xml_rows: List[Dict[str, Any]] = field(default_factory=list)
    malformed_xml_rows: List[Dict[str, Any]] = field(default_factory=list)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--wav-dir", type=Path, default=DEFAULT_WAV_DIR)
    parser.add_argument("--xml-dir", type=Path, default=DEFAULT_XML_DIR)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--target-shard-mb",
        type=int,
        default=500,
        help="Approximate shard size in MiB before rotating.",
    )
    parser.add_argument(
        "--max-audio-files",
        type=int,
        default=None,
        help="Optional limit for smoke tests.",
    )
    parser.add_argument(
        "--audio-glob",
        default="*.wav",
        help="Glob used under --wav-dir when discovering audio files.",
    )
    parser.add_argument(
        "--skip-existing",
        action="store_true",
        help="Skip work if the output directory already has shard files.",
    )
    return parser.parse_args()


def setup_logging(output_dir: Path) -> logging.Logger:
    logger = logging.getLogger("qasr_segment_to_arrow")
    logger.setLevel(logging.INFO)
    logger.handlers.clear()

    logs_dir = output_dir / "logs"
    logs_dir.mkdir(parents=True, exist_ok=True)
    log_path = logs_dir / "run.log"
    formatter = logging.Formatter(
        fmt="%(asctime)s | %(levelname)s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = TqdmLoggingHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)
    return logger


def ensure_output_layout(output_dir: Path) -> None:
    for rel in ("train", "index", "logs"):
        (output_dir / rel).mkdir(parents=True, exist_ok=True)


def atomic_write_jsonl(path: Path, rows: Sequence[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def atomic_write_json(path: Path, payload: Dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", encoding="utf-8", dir=path.parent, delete=False) as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2, sort_keys=True)
        handle.write("\n")
        tmp_path = Path(handle.name)
    tmp_path.replace(path)


def normalize_arabic_text(text: str) -> str:
    if not text:
        return ""
    text = str(text)
    text = text.replace("\ufeff", " ").replace("\u200f", " ").replace("\u200e", " ")
    for src, dst in {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
        "ى": "ي",
        "ؤ": "و",
        "ئ": "ي",
        "ة": "ه",
        "ـ": "",
    }.items():
        text = text.replace(src, dst)
    text = " ".join(text.split())
    return text.strip()


def join_qasr_tokens(tokens: Sequence[str]) -> str:
    pieces: List[str] = []
    for token in tokens:
        token = (token or "").strip()
        if not token:
            continue
        if not pieces:
            pieces.append(token)
            continue
        if token in PUNCT_NO_SPACE_BEFORE:
            pieces[-1] = pieces[-1].rstrip() + token
        elif pieces[-1] and pieces[-1][-1] in PUNCT_NO_SPACE_AFTER:
            pieces[-1] = pieces[-1] + token
        else:
            pieces.append(token)
    return " ".join(pieces).strip()


def safe_float(value: Any) -> Optional[float]:
    try:
        if value is None or value == "":
            return None
        return float(value)
    except (TypeError, ValueError):
        return None


def pcm_duration_seconds(audio_bytes: bytes, sample_rate: int = TARGET_SAMPLE_RATE) -> float:
    if not audio_bytes:
        return 0.0
    return float(len(audio_bytes) / 2 / sample_rate)


def to_mono_int16_bytes(waveform: np.ndarray) -> bytes:
    audio = np.asarray(waveform)
    if audio.size == 0:
        return b""
    if audio.ndim == 2:
        if audio.shape[0] <= 8 and audio.shape[0] < audio.shape[1]:
            audio = audio.mean(axis=0)
        else:
            audio = audio.mean(axis=1)
    elif audio.ndim > 2:
        audio = audio.reshape(-1, audio.shape[-1]).mean(axis=0)

    if np.issubdtype(audio.dtype, np.integer):
        max_abs = max(abs(np.iinfo(audio.dtype).min), np.iinfo(audio.dtype).max)
        audio = audio.astype(np.float32) / float(max_abs)
    else:
        audio = audio.astype(np.float32, copy=False)

    audio = np.clip(audio, -1.0, 1.0)
    pcm = np.round(audio * 32767.0).astype(np.int16)
    return pcm.tobytes()


def build_row(
    *,
    uid: str,
    audio_bytes: bytes,
    transcript: str,
    duration: float,
    segment_id: str,
    recording_id: str,
    speaker_id: Optional[str],
    gender: Optional[str],
    dialect: Optional[str],
    original_audio_path: str,
    original_annotation_path: str,
    source_row_index: int,
    start_time: float,
    end_time: float,
    metadata: Dict[str, Any],
) -> Dict[str, Any]:
    return {
        "uid": uid,
        "audio": audio_bytes,
        "sampling_rate": TARGET_SAMPLE_RATE,
        "transcript": transcript,
        "normalized_transcript": normalize_arabic_text(transcript),
        "duration": float(duration),
        "source": "qasr",
        "split": "train",
        "speaker_id": speaker_id,
        "gender": gender,
        "dialect": dialect,
        "language": "ar",
        "task_type": "asr",
        "segment_id": segment_id,
        "recording_id": recording_id,
        "original_audio_path": original_audio_path,
        "original_annotation_path": original_annotation_path,
        "source_file": original_annotation_path,
        "source_row_index": source_row_index,
        "start_time": float(start_time),
        "end_time": float(end_time),
        "metadata_json": json.dumps(metadata, ensure_ascii=False, sort_keys=True),
    }


def parse_qasr_xml(xml_path: Path) -> ParsedXml:
    try:
        root = ET.parse(xml_path).getroot()
    except ET.ParseError as exc:
        raise ValueError(f"Malformed XML: {exc}") from exc

    recording = root.find("./head/recording")
    if recording is None:
        raise ValueError("Missing <recording> metadata")

    speakers: Dict[str, Dict[str, Any]] = {}
    for speaker in root.findall("./head/speakers/speaker"):
        speaker_id = speaker.attrib.get("id", "")
        speakers[speaker_id] = {
            "name": speaker.attrib.get("name"),
            "normalized_name": speaker.attrib.get("normalizedName"),
            "gender": speaker.attrib.get("speakerGender"),
        }

    selected_segments = None
    for segments in root.findall("./body/segments"):
        if segments.attrib.get("annotation_id") == PREFERRED_ANNOTATION_ID:
            selected_segments = segments
            break
    if selected_segments is None:
        selected_segments = root.find("./body/segments")
    if selected_segments is None:
        raise ValueError("Missing <segments>")

    parsed_segments: List[SegmentSpec] = []
    for idx, segment in enumerate(selected_segments.findall("./segment")):
        words = [
            (element.text or "").strip()
            for element in segment.findall("./element")
            if (element.text or "").strip()
        ]
        speaker_id = segment.attrib.get("who")
        parsed_segments.append(
            SegmentSpec(
                segment_id=segment.attrib.get("id") or f"{xml_path.stem}_segment_{idx:06d}",
                transcript=join_qasr_tokens(words),
                raw_start_time=safe_float(segment.attrib.get("starttime")),
                raw_end_time=safe_float(segment.attrib.get("endtime")),
                speaker_id=speaker_id,
                speaker_meta=dict(speakers.get(speaker_id or "", {})),
                segment_meta={
                    "AWD": segment.attrib.get("AWD"),
                    "PMER": segment.attrib.get("PMER"),
                    "WMER": segment.attrib.get("WMER"),
                    "who": speaker_id,
                },
            )
        )

    return ParsedXml(
        recording_id=recording.attrib.get("filename") or xml_path.stem,
        annotation_id=selected_segments.attrib.get("annotation_id"),
        recording_meta=dict(recording.attrib),
        speakers=speakers,
        segments=parsed_segments,
    )


def resolve_segment_times(
    segments: Sequence[SegmentSpec],
    audio_duration: float,
) -> Iterator[Tuple[int, SegmentSpec]]:
    prev_end = 0.0
    for idx, segment in enumerate(segments):
        next_start = None
        for future in segments[idx + 1 :]:
            if future.raw_start_time is not None:
                next_start = future.raw_start_time
                break

        start_time = segment.raw_start_time
        end_time = segment.raw_end_time

        if start_time is None:
            start_time = prev_end
        if end_time is None:
            if next_start is not None and next_start >= start_time:
                end_time = next_start
            elif idx == len(segments) - 1:
                end_time = audio_duration
            else:
                end_time = start_time

        start_time = max(0.0, float(start_time))
        end_time = min(audio_duration, float(end_time))
        if end_time < start_time:
            end_time = start_time

        segment.start_time = start_time
        segment.end_time = end_time
        prev_end = end_time
        yield idx, segment


class ShardAccumulator:
    def __init__(self, context: BuildContext) -> None:
        self.context = context
        self.rows: List[Dict[str, Any]] = []
        self.row_metadata: List[Dict[str, Any]] = []
        self.audio_bytes = 0
        self.shard_id = 0

    def add_row(self, row: Dict[str, Any], row_metadata: Dict[str, Any]) -> None:
        self.rows.append(row)
        self.row_metadata.append(row_metadata)
        self.audio_bytes += len(row["audio"] or b"")
        if self.audio_bytes >= self.context.target_shard_bytes:
            self.flush()

    def flush(self) -> None:
        if not self.rows:
            return
        shard_path = self.context.output_dir / "train" / f"shard_{self.shard_id:06d}.arrow"
        write_shard(shard_path, self.rows, skip_existing=self.context.skip_existing)

        total_duration = float(sum(float(row["duration"]) for row in self.rows))
        shard_row = {
            "split": "train",
            "shard_id": self.shard_id,
            "shard_path": str(shard_path),
            "row_count": len(self.rows),
            "audio_bytes": self.audio_bytes,
            "duration_seconds": total_duration,
        }
        self.context.shard_index_rows.append(shard_row)

        for idx, (row, meta) in enumerate(zip(self.rows, self.row_metadata, strict=True)):
            segment_row = dict(meta)
            segment_row["shard_path"] = str(shard_path)
            segment_row["row_index_in_shard"] = idx
            segment_row["uid"] = row["uid"]
            self.context.segment_index_rows.append(segment_row)

        self.rows = []
        self.row_metadata = []
        self.audio_bytes = 0
        self.shard_id += 1

    def flush_all(self) -> None:
        self.flush()


def write_shard(path: Path, rows: Sequence[Dict[str, Any]], skip_existing: bool) -> None:
    if skip_existing and path.exists():
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = path.with_suffix(path.suffix + ".tmp")
    if temp_path.exists():
        temp_path.unlink()
    table = pa.Table.from_pylist(list(rows), schema=ARROW_SCHEMA)
    with pa.OSFile(str(temp_path), "wb") as sink:
        with pa_ipc.new_file(sink, ARROW_SCHEMA) as writer:
            writer.write_table(table)
    temp_path.replace(path)


def discover_audio_files(wav_dir: Path, pattern: str) -> List[Path]:
    return sorted(path for path in wav_dir.rglob(pattern) if path.is_file())


def existing_shards(output_dir: Path) -> bool:
    return any((output_dir / "train").glob("*.arrow"))


def process_audio_file(
    wav_path: Path,
    xml_dir: Path,
    context: BuildContext,
    shard_accumulator: ShardAccumulator,
) -> None:
    context.counters.audio_files_seen += 1
    xml_path = xml_dir / f"{wav_path.stem}.xml"

    audio_row: Dict[str, Any] = {
        "recording_id": wav_path.stem,
        "audio_path": str(wav_path),
        "xml_path": str(xml_path),
        "xml_exists": xml_path.exists(),
    }

    if not xml_path.exists():
        context.counters.audio_files_without_xml += 1
        audio_row["status"] = "missing_xml"
        context.audio_index_rows.append(audio_row)
        context.missing_xml_rows.append(audio_row)
        return

    context.counters.audio_files_with_xml += 1
    try:
        parsed = parse_qasr_xml(xml_path)
    except Exception as exc:
        context.counters.malformed_xml_files += 1
        audio_row["status"] = "malformed_xml"
        audio_row["error"] = str(exc)
        context.audio_index_rows.append(audio_row)
        context.malformed_xml_rows.append(audio_row)
        context.logger.warning("Malformed XML for %s: %s", wav_path.name, exc)
        return

    with sf.SoundFile(str(wav_path)) as snd:
        source_sr = int(snd.samplerate)
        channels = int(snd.channels)
        total_frames = len(snd)
        audio_duration = float(total_frames / source_sr) if source_sr else 0.0

        audio_row.update(
            {
                "status": "processed",
                "samplerate": source_sr,
                "channels": channels,
                "frames": total_frames,
                "duration_seconds": audio_duration,
                "annotation_id": parsed.annotation_id,
                "recording_meta": parsed.recording_meta,
                "segment_count_in_xml": len(parsed.segments),
            }
        )

        emitted_for_audio = 0
        for row_index, segment in resolve_segment_times(parsed.segments, audio_duration):
            transcript = (segment.transcript or "").strip()
            if not transcript:
                context.counters.empty_transcript_segments += 1
                continue

            start_time = segment.start_time
            end_time = segment.end_time
            if start_time is None or end_time is None or end_time <= start_time:
                context.counters.invalid_segments += 1
                continue

            start_frame = max(0, int(math.floor(start_time * source_sr)))
            end_frame = min(total_frames, int(math.ceil(end_time * source_sr)))
            if end_frame <= start_frame:
                context.counters.invalid_segments += 1
                continue

            snd.seek(start_frame)
            waveform = snd.read(end_frame - start_frame, dtype="float32", always_2d=False)
            pcm = to_mono_int16_bytes(np.asarray(waveform))
            duration = pcm_duration_seconds(pcm)
            if duration < MIN_DURATION_SECONDS:
                context.counters.too_short_segments += 1
                continue

            speaker_meta = segment.speaker_meta or {}
            metadata = {
                "annotation_id": parsed.annotation_id,
                "recording_meta": parsed.recording_meta,
                "speaker_meta": speaker_meta,
                "segment_meta": segment.segment_meta,
                "audio_samplerate_original": source_sr,
                "audio_channels_original": channels,
                "timing_inference": {
                    "raw_start_time": segment.raw_start_time,
                    "raw_end_time": segment.raw_end_time,
                    "resolved_start_time": start_time,
                    "resolved_end_time": end_time,
                },
            }
            uid = f"qasr:{wav_path.stem}:{segment.segment_id}"
            row = build_row(
                uid=uid,
                audio_bytes=pcm,
                transcript=transcript,
                duration=duration,
                segment_id=segment.segment_id,
                recording_id=wav_path.stem,
                speaker_id=segment.speaker_id,
                gender=speaker_meta.get("gender"),
                dialect=parsed.recording_meta.get("service_name"),
                original_audio_path=str(wav_path),
                original_annotation_path=str(xml_path),
                source_row_index=row_index,
                start_time=start_time,
                end_time=end_time,
                metadata=metadata,
            )
            shard_accumulator.add_row(
                row,
                row_metadata={
                    "recording_id": wav_path.stem,
                    "audio_path": str(wav_path),
                    "xml_path": str(xml_path),
                    "segment_id": segment.segment_id,
                    "source_row_index": row_index,
                    "start_time": start_time,
                    "end_time": end_time,
                    "duration_seconds": duration,
                    "speaker_id": segment.speaker_id,
                    "gender": speaker_meta.get("gender"),
                    "transcript_chars": len(transcript),
                },
            )
            emitted_for_audio += 1
            context.counters.emitted_segments += 1
            context.counters.emitted_seconds += duration

        audio_row["emitted_segments"] = emitted_for_audio
        if emitted_for_audio > 0:
            context.counters.audio_files_with_segments += 1

    context.audio_index_rows.append(audio_row)


def build_summary(context: BuildContext, args: argparse.Namespace) -> Dict[str, Any]:
    return {
        "wav_dir": str(args.wav_dir),
        "xml_dir": str(args.xml_dir),
        "output_dir": str(args.output_dir),
        "target_shard_mb": args.target_shard_mb,
        "max_audio_files": args.max_audio_files,
        "counts": {
            "audio_files_seen": context.counters.audio_files_seen,
            "audio_files_with_xml": context.counters.audio_files_with_xml,
            "audio_files_without_xml": context.counters.audio_files_without_xml,
            "audio_files_with_segments": context.counters.audio_files_with_segments,
            "malformed_xml_files": context.counters.malformed_xml_files,
            "empty_transcript_segments": context.counters.empty_transcript_segments,
            "invalid_segments": context.counters.invalid_segments,
            "too_short_segments": context.counters.too_short_segments,
            "emitted_segments": context.counters.emitted_segments,
            "emitted_seconds": round(context.counters.emitted_seconds, 3),
            "written_shards": len(context.shard_index_rows),
        },
        "index_files": {
            "audio_index": str(context.output_dir / "index" / "audio_index.jsonl"),
            "segment_index": str(context.output_dir / "index" / "segment_index.jsonl"),
            "shard_index": str(context.output_dir / "index" / "shard_index.jsonl"),
            "missing_xml": str(context.output_dir / "index" / "missing_xml.jsonl"),
            "malformed_xml": str(context.output_dir / "index" / "malformed_xml.jsonl"),
        },
    }


def maybe_limit(items: Iterable[Path], limit: Optional[int]) -> Iterator[Path]:
    if limit is None:
        yield from items
        return
    emitted = 0
    for item in items:
        if emitted >= limit:
            break
        yield item
        emitted += 1


def main() -> None:
    args = parse_args()
    ensure_output_layout(args.output_dir)
    logger = setup_logging(args.output_dir)

    if args.skip_existing and existing_shards(args.output_dir):
        logger.info("Existing shard files found in %s, exiting due to --skip-existing.", args.output_dir)
        return

    context = BuildContext(
        output_dir=args.output_dir,
        target_shard_bytes=args.target_shard_mb * 1024 * 1024,
        skip_existing=args.skip_existing,
        logger=logger,
    )
    shard_accumulator = ShardAccumulator(context)

    wav_files = discover_audio_files(args.wav_dir, args.audio_glob)
    logger.info("Discovered %d WAV files under %s", len(wav_files), args.wav_dir)

    for wav_path in maybe_limit(wav_files, args.max_audio_files):
        process_audio_file(wav_path, args.xml_dir, context, shard_accumulator)
        if context.counters.audio_files_seen % 25 == 0:
            logger.info(
                "Progress: %d audio files, %d emitted segments, %d shards pending=%d rows",
                context.counters.audio_files_seen,
                context.counters.emitted_segments,
                len(context.shard_index_rows),
                len(shard_accumulator.rows),
            )

    shard_accumulator.flush_all()

    index_dir = args.output_dir / "index"
    atomic_write_jsonl(index_dir / "audio_index.jsonl", context.audio_index_rows)
    atomic_write_jsonl(index_dir / "segment_index.jsonl", context.segment_index_rows)
    atomic_write_jsonl(index_dir / "shard_index.jsonl", context.shard_index_rows)
    atomic_write_jsonl(index_dir / "missing_xml.jsonl", context.missing_xml_rows)
    atomic_write_jsonl(index_dir / "malformed_xml.jsonl", context.malformed_xml_rows)

    summary = build_summary(context, args)
    atomic_write_json(args.output_dir / "summary.json", summary)
    logger.info(
        "Done. audio_files=%d emitted_segments=%d shards=%d",
        context.counters.audio_files_seen,
        context.counters.emitted_segments,
        len(context.shard_index_rows),
    )


if __name__ == "__main__":
    main()
