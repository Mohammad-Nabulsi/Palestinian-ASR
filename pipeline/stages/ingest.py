"""Stage 1 — ingest: turn heterogeneous raw sources into shard trees.

This is the one stage where per-dataset differences are real, so it is built as a
small adapter registry rather than a single code path. Each adapter replaces a
previously standalone script:

============================  ==========================================================
adapter                       replaces
============================  ==========================================================
``qasr_xml_wav``              ``preprocess/qasr_segment_to_arrow.py`` (invoked in-process)
``parquet_filter``            ``scripts/filter_masc_c_only.py``
``passthrough``               the symlink half of ``scripts/stage_raw_datasets.py``
``audio_text_pairs``          the Layla half of ``scripts/stage_raw_datasets.py``
                              + the sharding half of ``scripts/finalize_data_with_layla.py``
============================  ==========================================================
"""
from __future__ import annotations

import json
import os
import shutil
import sys
import wave
import zipfile
from pathlib import Path
from typing import Any, Callable
from xml.etree import ElementTree

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq

from ..config import REPO_ROOT, StageSpec
from ..context import RunContext, StageResult
from ..shards import RollingShardWriter, count_rows

WORD_NAMESPACE = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"

AdapterFn = Callable[[RunContext, dict[str, Any], Path], dict[str, Any]]
ADAPTERS: dict[str, AdapterFn] = {}


def adapter(name: str) -> Callable[[AdapterFn], AdapterFn]:
    def wrap(fn: AdapterFn) -> AdapterFn:
        ADAPTERS[name] = fn
        return fn

    return wrap


# ---------------------------------------------------------------- QASR


@adapter("qasr_xml_wav")
def ingest_qasr(ctx: RunContext, spec: dict[str, Any], output: Path) -> dict[str, Any]:
    """Segment QASR wav+xml pairs into Arrow shards.

    Runs ``preprocess/qasr_segment_to_arrow.py`` in-process (argv patched) instead of
    duplicating its 745 lines of XML/PCM handling. Every historical variant of this
    step -- the original 961-wav run, the ``--wav-stem-list`` part-2 run, and a future
    full ``wav_all/`` run -- is now just a different set of options here.
    """
    sys.path.insert(0, str(REPO_ROOT / "preprocess"))
    try:
        import qasr_segment_to_arrow as qasr
    finally:
        sys.path.pop(0)

    argv = [
        "qasr_segment_to_arrow.py",
        "--wav-dir", str(Path(spec["wav_dir"])),
        "--xml-dir", str(Path(spec["xml_dir"])),
        "--output-dir", str(output),
    ]
    for flag, key in (
        ("--target-shard-mb", "target_shard_mb"),
        ("--max-audio-files", "max_audio_files"),
        ("--audio-glob", "audio_glob"),
        ("--wav-stem-list", "wav_stem_list"),
    ):
        if spec.get(key) is not None:
            argv += [flag, str(spec[key])]
    if spec.get("skip_existing"):
        argv.append("--skip-existing")

    ctx.log(f"qasr_segment_to_arrow {' '.join(argv[1:])}")
    saved_argv = sys.argv
    sys.argv = argv
    try:
        qasr.main()
    finally:
        sys.argv = saved_argv

    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text(encoding="utf-8")) if summary_path.exists() else {}
    counts = summary.get("counts", {})
    return {
        "rows": counts.get("emitted_segments", 0),
        "shards": counts.get("written_shards", 0),
        "audio_files": counts.get("audio_files_seen", 0),
        "summary": str(summary_path),
    }


# ------------------------------------------------------- parquet filter


@adapter("parquet_filter")
def ingest_parquet_filter(ctx: RunContext, spec: dict[str, Any], output: Path) -> dict[str, Any]:
    """Keep only rows where ``column == equals`` across a tree of parquet shards.

    Generalized from ``filter_masc_c_only.py`` (which hardcoded ``type == "c"``).
    """
    source = Path(spec["input"])
    column = spec.get("column", "type")
    equals = spec.get("equals", "c")
    pattern = spec.get("glob", "*.parquet")

    if not source.exists():
        raise FileNotFoundError(f"parquet_filter source does not exist: {source}")

    output.mkdir(parents=True, exist_ok=True)
    total = kept = shards = 0
    for input_path in sorted(source.glob(pattern)):
        table = pq.read_table(input_path)
        total += table.num_rows
        if column in table.column_names:
            table = table.filter(pc.equal(table.column(column), pa.scalar(equals)))
        pq.write_table(table, output / input_path.name)
        kept += table.num_rows
        shards += 1
        ctx.log(f"{input_path.name}: kept {table.num_rows} rows")

    for name in spec.get("copy_metadata", ["README.md"]):
        src = source.parent / name
        if src.is_file():
            shutil.copy2(src, output.parent / name)

    return {"rows_in": total, "rows": kept, "shards": shards}


# ---------------------------------------------------------- passthrough


@adapter("passthrough")
def ingest_passthrough(ctx: RunContext, spec: dict[str, Any], output: Path) -> dict[str, Any]:
    """Expose an already-shard-shaped source under the ingest root.

    ``mode: symlink`` reproduces ``stage_raw_datasets.py``; ``hardlink`` and ``copy``
    are available when the consumer must not follow links (e.g. writing back in place).
    """
    source = Path(spec["input"])
    mode = spec.get("mode", "symlink")
    includes = spec.get("include", ["*"])
    excludes = spec.get("exclude", [])

    if not source.exists():
        raise FileNotFoundError(f"passthrough source does not exist: {source}")

    matched: list[Path] = []
    for pattern in includes:
        matched.extend(sorted(source.glob(pattern)))
    matched = [p for p in dict.fromkeys(matched) if p.is_file()]
    if excludes:
        matched = [
            p for p in matched
            if not any(p.match(pattern) for pattern in excludes)
        ]

    if not matched:
        raise FileNotFoundError(
            f"passthrough matched no files under {source} for include={includes}"
        )

    output.mkdir(parents=True, exist_ok=True)
    for path in matched:
        dest = output / path.name
        if dest.exists() or dest.is_symlink():
            dest.unlink()
        if mode == "symlink":
            dest.symlink_to(os.path.relpath(path, start=output))
        elif mode == "hardlink":
            os.link(path, dest)
        elif mode == "copy":
            shutil.copy2(path, dest)
        else:
            raise ValueError(f"Unknown passthrough mode: {mode}")

    rows = None
    if spec.get("count_rows", True):
        rows = sum(count_rows(p) for p in matched)
    return {"files": len(matched), "rows": rows, "mode": mode}


# ------------------------------------------------------ audio/text pairs


def extract_docx_text(docx_path: Path) -> str:
    with zipfile.ZipFile(docx_path) as docx_zip:
        xml_bytes = docx_zip.read("word/document.xml")
    root = ElementTree.fromstring(xml_bytes)
    paragraphs = []
    for para in root.iter(f"{WORD_NAMESPACE}p"):
        parts = [node.text or "" for node in para.iter(f"{WORD_NAMESPACE}t")]
        if parts:
            paragraphs.append("".join(parts))
    return "\n".join(paragraphs).strip()


def read_wav_bytes_and_duration(path: Path) -> tuple[bytes, float, int]:
    audio_bytes = path.read_bytes()
    with wave.open(str(path), "rb") as wf:
        frames = wf.getnframes()
        rate = wf.getframerate()
    return audio_bytes, (frames / rate if rate else 0.0), rate


def load_normalized_overrides(paths: list[Path]) -> dict[str, str]:
    """Map ``source`` -> normalized transcript from the LLM-normalization JSONs.

    These are the four ``normalized_*.json`` files described in DATA_CURATION.md's
    "Layla Prompt Merge and Sharding" section.
    """
    overrides: dict[str, str] = {}
    for path in paths:
        if not path.exists():
            continue
        obj = json.loads(path.read_text(encoding="utf-8"))
        for record in obj.get("records", []):
            source = record.get("source")
            normalized = record.get("normalized")
            if source and normalized:
                overrides[str(source).replace("./", "")] = normalized
    return overrides


@adapter("audio_text_pairs")
def ingest_audio_text_pairs(ctx: RunContext, spec: dict[str, Any], output: Path) -> dict[str, Any]:
    """Pair audio files with sibling transcripts and write parquet training shards.

    Transcript resolution order (first hit wins):
      1. a normalized-JSON override keyed by the transcript's relative path
      2. a sibling ``.txt``
      3. a sibling ``.docx`` (converted in-process; replaces the docx->txt staging step)
    """
    source = Path(spec["input"])
    if not source.exists():
        raise FileNotFoundError(f"audio_text_pairs source does not exist: {source}")

    audio_suffixes = spec.get("audio_suffixes", [".WAV", ".wav"])
    marker = spec.get("transcript_marker", "_Arabic_transcription")
    dataset_name = spec.get("dataset", output.name)
    overrides = load_normalized_overrides(
        [Path(p) for p in spec.get("normalized_json", [])]
    )

    audio_paths = sorted(
        p for p in source.rglob("*") if p.is_file() and p.suffix in set(audio_suffixes)
    )
    if not audio_paths:
        raise FileNotFoundError(f"No audio files with suffixes {audio_suffixes} under {source}")

    counts = {
        "audio_files": len(audio_paths),
        "from_override": 0,
        "from_txt": 0,
        "from_docx": 0,
        "missing_transcript": 0,
    }
    rows: list[dict[str, Any]] = []
    missing: list[str] = []

    for audio_path in audio_paths:
        base = audio_path.with_suffix("")
        rel_txt = f"{base.relative_to(source)}{marker}.txt"
        text = overrides.get(rel_txt)
        if text is not None:
            counts["from_override"] += 1
        else:
            txt_candidates = [
                base.with_name(base.name + marker).with_suffix(".txt"),
                base.with_suffix(".txt"),
            ]
            docx_candidates = [
                base.with_name(base.name + marker).with_suffix(".docx"),
                base.with_suffix(".docx"),
            ]
            for candidate in txt_candidates:
                if candidate.is_file():
                    text = candidate.read_text(encoding="utf-8", errors="replace").strip()
                    counts["from_txt"] += 1
                    break
            if text is None:
                for candidate in docx_candidates:
                    if candidate.is_file():
                        text = extract_docx_text(candidate)
                        counts["from_docx"] += 1
                        break

        if not text:
            counts["missing_transcript"] += 1
            missing.append(str(audio_path.relative_to(source)))
            continue

        audio_bytes, duration, sampling_rate = read_wav_bytes_and_duration(audio_path)
        rows.append(
            {
                "seg_id": f"{dataset_name}:{base.relative_to(source)}",
                "audio": audio_bytes,
                "sampling_rate": sampling_rate,
                "transcription": text,
                "duration": duration,
                "source": dataset_name,
                "source_file": str(audio_path),
            }
        )

    schema = pa.schema(
        [
            pa.field("seg_id", pa.string()),
            pa.field("audio", pa.binary()),
            pa.field("sampling_rate", pa.int32()),
            pa.field("transcription", pa.string()),
            pa.field("duration", pa.float32()),
            pa.field("source", pa.string()),
            pa.field("source_file", pa.string()),
        ]
    )
    rows_per_shard = int(spec.get("rows_per_shard", 500))
    with RollingShardWriter(output, dataset_name, rows_per_shard=rows_per_shard) as writer:
        for start in range(0, len(rows), rows_per_shard):
            chunk = rows[start : start + rows_per_shard]
            writer.append(pa.Table.from_pylist(chunk, schema=schema))
        shards = len(writer.written_paths)

    if missing:
        ctx.write_json(
            ctx.stage_reports_dir("ingest") / f"{dataset_name}_missing_transcripts.json",
            {"count": len(missing), "files": missing[:200]},
        )

    counts["rows"] = len(rows)
    counts["shards"] = shards
    return counts


# ------------------------------------------------------------- runner


def run(ctx: RunContext, spec: StageSpec) -> StageResult:
    result = StageResult(stage_id=spec.id, stage=spec.stage)
    sources = spec.require("sources")
    root = spec.require_path("output_root")

    for source in sources:
        name = source["name"]
        if not source.get("enabled", True):
            ctx.log(f"skip source {name} (disabled)")
            result.counts[name] = {"status": "disabled"}
            continue

        adapter_name = source["adapter"]
        if adapter_name not in ADAPTERS:
            raise SystemExit(
                f"Unknown ingest adapter {adapter_name!r}. Known: {sorted(ADAPTERS)}"
            )

        output = Path(source.get("output", root / name))
        ctx.log(f"ingest {name} via {adapter_name} -> {output}")
        if ctx.dry_run:
            result.counts[name] = {"status": "dry_run", "adapter": adapter_name}
            result.outputs[name] = str(output)
            continue

        if output.exists() and source.get("overwrite", True):
            shutil.rmtree(output) if output.is_dir() and not output.is_symlink() else output.unlink()

        counts = ADAPTERS[adapter_name](ctx, source, output)
        counts["adapter"] = adapter_name
        result.counts[name] = counts
        result.outputs[name] = str(output)
        ctx.log(f"ingest {name} done: {counts}")

    return result
