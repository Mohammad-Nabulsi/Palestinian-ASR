"""Stage 4 — dialect: per-row dialect identification (text or audio).

Unifies the two scan scripts, which had the same shape (iterate shards, score rows,
append to ``row_probabilities.jsonl``, support ``--resume``) but different models:

- ``dialect_identifiaction/arabic_text_dialect_scan_marbertv2_written.py``  -> ``backend: marbertv2_text``
- ``dialect_identifiaction/arabic_dialect_scan_badrex_mms300m.py``          -> ``backend: badrex_audio``

The two subprocess wrappers around them
(``scripts/repair_qasr_audio_and_rebuild_levant_binary.py``,
``scripts/rebuild_qasr_only_levant_binary.py``) disappear entirely: a repair run is
just this stage with different options followed by the ``split`` stage.

The audio backend carries the QASR fix described in DATA_CURATION.md's "QASR Audio
Classification Repair": ``audio`` payloads are decoded as an encoded audio stream
when they carry a WAV/FLAC/OGG header, and as raw PCM ``int16`` otherwise (using the
row's ``sampling_rate``). The pre-fix code assumed encoded streams only and failed
68% of QASR rows with ``LibsndfileError``.

Backends requiring torch/transformers are imported lazily, so ``backend: hash_stub``
runs anywhere -- that is what the sample/smoke config uses.
"""
from __future__ import annotations

import hashlib
import io
import json
import struct
from pathlib import Path
from typing import Any, Iterator

import numpy as np

from ..config import StageSpec
from ..context import RunContext, StageResult
from ..shards import find_first_column, iter_table_batches
from ..textnorm import normalize_arabic_transcript

TEXT_FIELD_CANDIDATES = [
    "manual_normalized_transcript",
    "transcript",
    "transcription",
    "text",
    "raw_text",
    "sentence",
]

DEFAULT_TEXT_LABELS = ("LEV", "EGY", "GLF", "NOR", "MSA")
DEFAULT_AUDIO_LABELS = ("Levantine", "Egyptian", "Gulf", "Maghrebi", "MSA")

AUDIO_MAGIC = (b"RIFF", b"fLaC", b"OggS", b"FORM")


# ------------------------------------------------------------ decoding


def looks_like_encoded_audio(payload: bytes) -> bool:
    return any(payload[: len(magic)] == magic for magic in AUDIO_MAGIC)


def decode_audio(payload: Any, sampling_rate: int | None, target_sr: int) -> tuple[np.ndarray, int]:
    """Decode either an encoded audio stream or raw PCM int16 bytes.

    This dual mode is the QASR repair: the segment builder stores raw PCM int16 in
    ``audio`` with the rate in a separate ``sampling_rate`` column.
    """
    if isinstance(payload, dict):  # HF Audio feature with decode=False
        if payload.get("array") is not None:
            return np.asarray(payload["array"], dtype=np.float32), int(
                payload.get("sampling_rate") or sampling_rate or target_sr
            )
        payload = payload.get("bytes")

    if payload is None:
        raise ValueError("row has no audio payload")

    payload = bytes(payload)
    if looks_like_encoded_audio(payload):
        import soundfile as sf

        array, rate = sf.read(io.BytesIO(payload), dtype="float32", always_2d=False)
        if array.ndim > 1:
            array = array.mean(axis=1)
        return array.astype(np.float32), int(rate)

    if not sampling_rate:
        raise ValueError("raw PCM payload without a sampling_rate column")
    if len(payload) % 2:
        payload = payload[:-1]
    array = np.frombuffer(payload, dtype="<i2").astype(np.float32) / 32768.0
    return array, int(sampling_rate)


def resample(array: np.ndarray, source_sr: int, target_sr: int) -> np.ndarray:
    if source_sr == target_sr or array.size == 0:
        return array
    n_target = int(round(array.size * target_sr / source_sr))
    if n_target <= 0:
        return array[:0]
    return np.interp(
        np.linspace(0.0, array.size - 1, n_target, dtype=np.float64),
        np.arange(array.size, dtype=np.float64),
        array,
    ).astype(np.float32)


# ------------------------------------------------------------ backends


class HashStubBackend:
    """Deterministic offline scorer. No model download, no torch.

    Produces stable pseudo-probabilities from a hash of the row's text. It exists so
    the pipeline is runnable end-to-end (and unit-testable) on a box with no GPU and
    no model cache. It is NOT a dialect classifier -- never use it for real curation.
    """

    def __init__(self, labels: tuple[str, ...], modality: str) -> None:
        self.labels = labels
        self.modality = modality
        self.name = "hash_stub"

    def score_batch(self, items: list[dict[str, Any]]) -> list[dict[str, float]]:
        out = []
        for item in items:
            seed = item.get("text") or ""
            if not seed and item.get("audio_len"):
                seed = f"audio:{item['audio_len']}"
            digest = hashlib.sha256(str(seed).encode("utf-8")).digest()
            raw = np.array(
                struct.unpack(f"{len(self.labels)}H", digest[: 2 * len(self.labels)]),
                dtype=np.float64,
            )
            probs = raw / raw.sum() if raw.sum() else np.full(len(self.labels), 1 / len(self.labels))
            out.append({label: float(p) for label, p in zip(self.labels, probs)})
        return out


class TransformersTextBackend:
    """MarBERTv2 written-dialect classifier (``backend: marbertv2_text``)."""

    def __init__(self, model_id: str, device: str, max_length: int) -> None:
        import torch
        from transformers import AutoModelForSequenceClassification, AutoTokenizer

        self.torch = torch
        self.device = _pick_device(device, torch)
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.max_length = max_length
        self.labels = tuple(self.model.config.id2label[i] for i in range(self.model.config.num_labels))
        self.name = model_id

    def score_batch(self, items: list[dict[str, Any]]) -> list[dict[str, float]]:
        texts = [item.get("text") or "" for item in items]
        encoded = self.tokenizer(
            texts, padding=True, truncation=True, max_length=self.max_length, return_tensors="pt"
        ).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**encoded).logits
            probs = self.torch.softmax(logits, dim=-1).cpu().numpy()
        return [{label: float(p) for label, p in zip(self.labels, row)} for row in probs]


class TransformersAudioBackend:
    """badrex mms-300m audio dialect identifier (``backend: badrex_audio``)."""

    def __init__(self, model_id: str, device: str, target_sr: int) -> None:
        import torch
        from transformers import AutoFeatureExtractor, AutoModelForAudioClassification

        self.torch = torch
        self.device = _pick_device(device, torch)
        self.feature_extractor = AutoFeatureExtractor.from_pretrained(model_id)
        self.model = AutoModelForAudioClassification.from_pretrained(model_id).to(self.device)
        self.model.eval()
        self.target_sr = target_sr
        self.labels = tuple(self.model.config.id2label[i] for i in range(self.model.config.num_labels))
        self.name = model_id

    def score_batch(self, items: list[dict[str, Any]]) -> list[dict[str, float]]:
        arrays = [item["audio"] for item in items]
        inputs = self.feature_extractor(
            arrays, sampling_rate=self.target_sr, return_tensors="pt", padding=True
        ).to(self.device)
        with self.torch.no_grad():
            logits = self.model(**inputs).logits
            probs = self.torch.softmax(logits, dim=-1).cpu().numpy()
        return [{label: float(p) for label, p in zip(self.labels, row)} for row in probs]


def _pick_device(device: str, torch: Any) -> str:
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    return device


def build_backend(spec: StageSpec, modality: str) -> Any:
    backend = spec.get("backend", "hash_stub")
    if backend == "hash_stub":
        labels = tuple(
            spec.get("labels")
            or (DEFAULT_AUDIO_LABELS if modality == "audio" else DEFAULT_TEXT_LABELS)
        )
        return HashStubBackend(labels, modality)
    if backend == "marbertv2_text":
        return TransformersTextBackend(
            spec.get("model_id", "IbrahimAmin/marbertv2-arabic-written-dialect-classifier"),
            spec.get("device", "auto"),
            int(spec.get("max_length", 512)),
        )
    if backend == "badrex_audio":
        return TransformersAudioBackend(
            spec.get("model_id", "badrex/mms-300m-arabic-dialect-identifier"),
            spec.get("device", "auto"),
            int(spec.get("target_sr", 16_000)),
        )
    raise SystemExit(
        f"Unknown dialect backend {backend!r}. "
        "Known: hash_stub, marbertv2_text, badrex_audio"
    )


# ------------------------------------------------------------- helpers


def load_candidates(path: Path, label: str, threshold: float) -> dict[str, set[int]]:
    """Rows a prior text pass scored at/above ``threshold``, keyed by source file."""
    candidates: dict[str, set[int]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            score = record.get("target_label_score")
            if score is None:
                score = (record.get("label_scores") or {}).get(label, 0.0)
            if float(score) >= threshold:
                key = str(Path(record["source_file"]).resolve())
                candidates.setdefault(key, set()).add(int(record["row_idx"]))
    return candidates


def load_text_scores(path: Path, label: str) -> dict[tuple[str, int], dict[str, float]]:
    scores: dict[tuple[str, int], dict[str, float]] = {}
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            key = (str(Path(record["source_file"]).resolve()), int(record["row_idx"]))
            scores[key] = record.get("label_scores") or {}
    return scores


def completed_keys(path: Path) -> set[tuple[str, int]]:
    done: set[tuple[str, int]] = set()
    if not path.exists():
        return done
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                continue
            done.add((str(Path(record["source_file"]).resolve()), int(record["row_idx"])))
    return done


def gather_shards(root: Path, sources: list[dict[str, Any]]) -> Iterator[tuple[str, Path]]:
    for entry in sources:
        name = entry["name"]
        for pattern in entry.get("include", ["*.parquet"]):
            for path in sorted(root.glob(pattern)):
                if path.is_file():
                    yield name, path


# -------------------------------------------------------------- runner


def run(ctx: RunContext, spec: StageSpec) -> StageResult:
    result = StageResult(stage_id=spec.id, stage=spec.stage)
    modality = spec.get("modality", "text")
    if modality not in {"text", "audio"}:
        raise SystemExit(f"dialect: modality must be 'text' or 'audio', got {modality!r}")

    input_root = spec.require_path("input_root")
    output_dir = spec.require_path("output_dir")
    output_dir.mkdir(parents=True, exist_ok=True)
    probs_path = output_dir / "row_probabilities.jsonl"

    target_label = spec.get("target_label", "Levantine" if modality == "audio" else "LEV")
    batch_size = int(spec.get("batch_size", 32))
    max_rows = spec.get("max_rows")
    target_sr = int(spec.get("target_sr", 16_000))
    min_seconds = float(spec.get("min_seconds", 0.0))
    resume = bool(spec.get("resume", False))

    candidates: dict[str, set[int]] | None = None
    text_scores: dict[tuple[str, int], dict[str, float]] = {}
    candidates_from = spec.get("candidates_from")
    if candidates_from:
        cpath = Path(candidates_from["path"])
        if not cpath.exists():
            raise FileNotFoundError(f"dialect: candidates_from path missing: {cpath}")
        clabel = candidates_from.get("label", "LEV")
        candidates = load_candidates(cpath, clabel, float(candidates_from.get("threshold", 0.8)))
        text_scores = load_text_scores(cpath, clabel)
        ctx.log(
            f"candidate filter: {sum(len(v) for v in candidates.values()):,} row(s) "
            f"from {cpath.name}"
        )

    already = completed_keys(probs_path) if resume else set()
    if resume and already:
        ctx.log(f"resume: {len(already):,} row(s) already scored")
    elif probs_path.exists():
        probs_path.unlink()

    backend = build_backend(spec, modality)
    ctx.log(f"backend: {backend.name} (modality={modality}, target_label={target_label})")
    if ctx.dry_run:
        result.counts["__totals__"] = {"status": "dry_run"}
        result.outputs["row_probabilities"] = str(probs_path)
        return result

    counts = {"scanned": 0, "scored": 0, "skipped_not_candidate": 0, "skipped_resume": 0, "errors": 0}
    per_source: dict[str, int] = {}
    pending: list[dict[str, Any]] = []
    handle = probs_path.open("a", encoding="utf-8")

    def flush() -> None:
        if not pending:
            return
        scored = backend.score_batch(pending)
        for item, label_scores in zip(pending, scored):
            record = {
                "source": item["source"],
                "source_file": item["source_file"],
                "row_idx": item["row_idx"],
                "status": "ok",
                "backend": backend.name,
                "modality": modality,
                "label_scores": label_scores,
                "predicted_label": max(label_scores, key=label_scores.get),
                "target_label": target_label,
                "target_label_score": float(label_scores.get(target_label, 0.0)),
            }
            key = (item["source_file"], item["row_idx"])
            if key in text_scores:
                record["text_label_scores"] = text_scores[key]
                record["text_target_label_score"] = float(
                    text_scores[key].get(spec.get("text_label", "LEV"), 0.0)
                )
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")
            counts["scored"] += 1
            per_source[item["source"]] = per_source.get(item["source"], 0) + 1
        pending.clear()

    def write_error(source: str, source_file: str, row_idx: int, message: str) -> None:
        handle.write(
            json.dumps(
                {
                    "source": source,
                    "source_file": source_file,
                    "row_idx": row_idx,
                    "status": "error",
                    "error": message,
                    "backend": backend.name,
                    "modality": modality,
                },
                ensure_ascii=False,
            )
            + "\n"
        )
        counts["errors"] += 1

    try:
        for source_name, path in gather_shards(input_root, spec.require("sources")):
            resolved = str(path.resolve())
            allowed = candidates.get(resolved) if candidates is not None else None
            if candidates is not None and not allowed:
                continue

            row_idx = -1
            for table in iter_table_batches(path, batch_size=max(batch_size, 256)):
                names = table.column_names
                text_col = find_first_column(names, spec.get("text_columns") or TEXT_FIELD_CANDIDATES)
                audio_col = "audio" if "audio" in names else None
                sr_col = "sampling_rate" if "sampling_rate" in names else None
                dur_col = find_first_column(names, ["duration", "duration_sec"])

                texts = table[text_col].to_pylist() if text_col else [None] * table.num_rows
                audios = table[audio_col].to_pylist() if audio_col else [None] * table.num_rows
                rates = table[sr_col].to_pylist() if sr_col else [None] * table.num_rows
                durations = table[dur_col].to_pylist() if dur_col else [None] * table.num_rows

                for i in range(table.num_rows):
                    row_idx += 1
                    counts["scanned"] += 1
                    if allowed is not None and row_idx not in allowed:
                        counts["skipped_not_candidate"] += 1
                        continue
                    if (resolved, row_idx) in already:
                        counts["skipped_resume"] += 1
                        continue
                    if max_rows is not None and counts["scored"] + len(pending) >= max_rows:
                        break

                    item: dict[str, Any] = {
                        "source": source_name,
                        "source_file": resolved,
                        "row_idx": row_idx,
                        "text": normalize_arabic_transcript(texts[i]) if texts[i] else "",
                    }

                    if modality == "audio":
                        try:
                            array, sr = decode_audio(audios[i], rates[i], target_sr)
                            array = resample(array, sr, target_sr)
                        except Exception as exc:  # decode failures are data, not crashes
                            write_error(source_name, resolved, row_idx, f"{type(exc).__name__}: {exc}")
                            continue
                        seconds = durations[i] if durations[i] else array.size / target_sr
                        if seconds < min_seconds:
                            write_error(
                                source_name, resolved, row_idx,
                                f"too_short: {seconds:.3f}s < {min_seconds}s",
                            )
                            continue
                        item["audio"] = array
                        item["audio_len"] = int(array.size)

                    pending.append(item)
                    if len(pending) >= batch_size:
                        flush()

                if max_rows is not None and counts["scored"] + len(pending) >= max_rows:
                    break

            flush()  # so the per-file counts below are actually current
            ctx.log(f"{path.name}: scored={counts['scored']:,} errors={counts['errors']:,}")
            if max_rows is not None and counts["scored"] + len(pending) >= max_rows:
                break

        flush()
    finally:
        handle.close()

    summary = {
        "modality": modality,
        "backend": backend.name,
        "target_label": target_label,
        "input_root": str(input_root),
        "output_dir": str(output_dir),
        "counts": counts,
        "per_source_scored": per_source,
    }
    ctx.write_json(output_dir / "summary.json", summary)
    ctx.write_json(ctx.stage_reports_dir(spec.id) / "summary.json", summary)
    result.counts = {**counts, "per_source": per_source}
    result.outputs["row_probabilities"] = str(probs_path)
    if backend.name == "hash_stub":
        result.notes.append(
            "hash_stub backend: deterministic placeholder scores, not real dialect ID"
        )
    ctx.log(f"dialect done: {counts}")
    return result
