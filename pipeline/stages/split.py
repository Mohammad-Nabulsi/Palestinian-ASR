"""Stage 5 — split: build the Levantine/non-Levantine binary curated layout.

Replaces ``scripts/create_levant_non_levant_splits.py`` plus its two subprocess
wrappers (``scripts/rebuild_qasr_only_levant_binary.py``,
``scripts/repair_qasr_audio_and_rebuild_levant_binary.py``) -- those wrappers only
re-ran the same routing with a different ``row_probabilities.jsonl``, which is now a
config value.

Routing rule (unchanged): a row from a *binary* leaf goes to ``lev`` when the text
stage scored it ``>= threshold`` for the text label AND the audio stage scored it
``>= threshold`` for the audio label; everything else goes to ``non_lev``. Leaves not
listed as binary are written whole.

Split assignment: when ``speaker_assignments`` is configured (the
``speaker_select`` stage's output), a row goes to the split its *speaker* was
assigned -- so every utterance of a qasr recording or a masc_c video lands in one
split and the sets are speaker-disjoint. This is the intended path; the older
behaviour, hashing ``(seed, leaf, source_file, row_idx)`` into fixed ratios, is
what let the same recording appear in train and test at once, and it survives
only for leaves with no speaker key at all (omni, layla, casa/*).

Rows whose speaker was never assigned -- ranked below the last block -- follow
``unassigned_split``: a split name to send them to, or ``drop`` (the default) to
leave them out, or ``hash`` for the legacy ratio assignment. Dropping is the
default because a speaker the selection did not choose should not silently
appear in a set.
"""
from __future__ import annotations

import hashlib
import json
import shutil
from collections import defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa

from ..config import StageSpec
from ..context import RunContext, StageResult
from ..shards import RollingShardWriter, iter_table_batches

SPLITS = ("train", "val", "test")


def canonical(path: str | Path) -> str:
    return str(Path(path).resolve())


def load_accepted_rows(
    path: Path,
    text_label: str,
    audio_label: str,
    threshold: float,
) -> tuple[dict[str, set[int]], dict[str, Any]]:
    """Rows passing both the text and audio thresholds, keyed by resolved source file."""
    accepted: dict[str, set[int]] = defaultdict(set)
    summary = {
        "threshold": threshold,
        "rows_scanned": 0,
        "rows_status_ok": 0,
        "rows_accepted": 0,
        "per_source_scanned": defaultdict(int),
        "per_source_accepted": defaultdict(int),
    }

    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            source = record.get("source", "unknown")
            summary["rows_scanned"] += 1
            summary["per_source_scanned"][source] += 1
            if record.get("status", "ok") != "ok":
                continue
            summary["rows_status_ok"] += 1

            audio_score = record.get("target_label_score")
            if audio_score is None:
                audio_score = (record.get("label_scores") or {}).get(audio_label, 0.0)
            text_score = record.get("text_target_label_score")
            if text_score is None:
                text_score = (record.get("text_label_scores") or {}).get(text_label, 0.0)

            if float(text_score) >= threshold and float(audio_score) >= threshold:
                accepted[canonical(record["source_file"])].add(int(record["row_idx"]))
                summary["rows_accepted"] += 1
                summary["per_source_accepted"][source] += 1

    summary["per_source_scanned"] = dict(summary["per_source_scanned"])
    summary["per_source_accepted"] = dict(summary["per_source_accepted"])
    return dict(accepted), summary


def load_speaker_assignments(path: Path) -> tuple[dict[tuple[str, str], str], dict[str, Any]]:
    """(source, speaker_key) -> split, from speaker_select's output."""
    payload = json.loads(path.read_text(encoding="utf-8"))
    mapping = {
        (row["source"], str(row["speaker_key"])): row["split"]
        for row in payload.get("assignments", [])
    }
    per_split: dict[str, int] = defaultdict(int)
    for split in mapping.values():
        per_split[split] += 1
    return mapping, {
        "path": str(path),
        "speakers": len(mapping),
        "speakers_per_split": dict(per_split),
        "meta": payload.get("meta", {}),
    }


def assign_split(seed: int, leaf: str, source_file: str, row_idx: int, ratios: dict[str, float]) -> str:
    key = f"{seed}|{leaf}|{source_file}|{row_idx}".encode("utf-8")
    draw = int.from_bytes(hashlib.blake2b(key, digest_size=8).digest(), "big") / 2**64
    cumulative = 0.0
    for split in SPLITS:
        cumulative += ratios[split]
        if draw < cumulative:
            return split
    return SPLITS[-1]


def run(ctx: RunContext, spec: StageSpec) -> StageResult:
    result = StageResult(stage_id=spec.id, stage=spec.stage)
    data_root = spec.require_path("input_root")
    output_root = spec.require_path("output_root")
    threshold = float(spec.get("threshold", 0.8))
    seed = int(spec.get("seed", 42))
    rows_per_shard = int(spec.get("rows_per_shard", 50_000))
    compression = spec.get("compression", "snappy")
    text_label = spec.get("text_label", "LEV")
    audio_label = spec.get("audio_label", "Levantine")
    unassigned_split = spec.get("unassigned_split", "drop")

    ratios = {
        "train": float(spec.get("train_ratio", 0.70)),
        "val": float(spec.get("val_ratio", 0.15)),
        "test": float(spec.get("test_ratio", 0.15)),
    }
    total_ratio = sum(ratios.values())
    if abs(total_ratio - 1.0) > 1e-6:
        raise SystemExit(f"split: train/val/test ratios must sum to 1.0, got {total_ratio}")

    if not data_root.exists():
        raise FileNotFoundError(f"split: input_root does not exist: {data_root}")

    speakers: dict[tuple[str, str], str] = {}
    speaker_summary: dict[str, Any] = {"path": None}
    assignments_path = spec.path("speaker_assignments")
    if assignments_path is not None:
        if not assignments_path.exists():
            raise FileNotFoundError(f"split: speaker_assignments not found: {assignments_path}")
        speakers, speaker_summary = load_speaker_assignments(assignments_path)
        ctx.log(
            f"speaker routing: {speaker_summary['speakers']:,} speaker(s) -> "
            f"{speaker_summary['speakers_per_split']}"
        )
        if unassigned_split not in {"drop", "hash"} and unassigned_split not in SPLITS:
            raise SystemExit(
                f"split: unassigned_split must be a split name, 'drop' or 'hash'; "
                f"got {unassigned_split!r}"
            )

    accepted: dict[str, set[int]] = {}
    routing_summary: dict[str, Any] = {"source": None}
    probs_path = spec.path("audio_row_probs")
    if probs_path is not None:
        if not probs_path.exists():
            raise FileNotFoundError(f"split: audio_row_probs not found: {probs_path}")
        accepted, routing_summary = load_accepted_rows(
            probs_path, text_label, audio_label, threshold
        )
        routing_summary["source"] = str(probs_path)
        ctx.log(
            f"routing: {routing_summary['rows_accepted']:,} accepted "
            f"of {routing_summary['rows_scanned']:,} scored row(s)"
        )

    if output_root.exists() and spec.get("overwrite", True):
        if output_root.is_symlink():
            raise SystemExit(f"Refusing to overwrite a symlinked output root: {output_root}")
        shutil.rmtree(output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    leaves = spec.require("leaves")
    writers: dict[tuple[str, str], RollingShardWriter] = {}
    counts: dict[str, dict[str, int]] = defaultdict(lambda: defaultdict(int))

    def writer_for(split: str, leaf_path: str) -> RollingShardWriter:
        key = (split, leaf_path)
        if key not in writers:
            out_dir = output_root / split / leaf_path
            prefix = leaf_path.replace("/", "_")
            writers[key] = RollingShardWriter(
                out_dir, prefix, rows_per_shard=rows_per_shard, compression=compression
            )
        return writers[key]

    try:
        for leaf in leaves:
            name = leaf["name"]
            binary = bool(leaf.get("binary", False))
            matched: list[Path] = []
            for pattern in leaf.get("include", ["*.parquet"]):
                matched.extend(sorted(data_root.glob(pattern)))
            matched = [p for p in dict.fromkeys(matched) if p.is_file()]

            if not matched:
                if leaf.get("optional", False):
                    ctx.log(f"leaf {name}: no files matched, skipped (optional)")
                    continue
                raise FileNotFoundError(
                    f"split: leaf {name!r} matched no files under {data_root} "
                    f"(include={leaf.get('include')})"
                )

            # A leaf routed by speaker names the source its keys were recorded
            # under and the column holding the key. Without both, the leaf keeps
            # the legacy hash assignment.
            speaker_source = leaf.get("speaker_source")
            key_column = leaf.get("speaker_key_column")
            by_speaker = bool(speakers) and speaker_source is not None and key_column is not None

            ctx.log(
                f"leaf {name}: {len(matched)} shard(s), binary={binary}, "
                f"routing={'speaker' if by_speaker else 'hash'}"
            )
            for path in matched:
                resolved = canonical(path)
                lev_rows = accepted.get(resolved, set())
                row_idx = -1
                for table in iter_table_batches(path, batch_size=8192):
                    if by_speaker and key_column not in table.column_names:
                        raise SystemExit(
                            f"split: leaf {name!r} is routed by speaker but column "
                            f"{key_column!r} is missing from {path} "
                            f"(has {table.column_names})"
                        )
                    key_values = table[key_column].to_pylist() if by_speaker else None

                    # Bucket the batch's rows by (split, leaf_path) then write once.
                    buckets: dict[tuple[str, str], list[int]] = defaultdict(list)
                    for i in range(table.num_rows):
                        row_idx += 1
                        if binary:
                            leaf_path = f"{name}/lev" if row_idx in lev_rows else f"{name}/non_lev"
                        else:
                            leaf_path = name

                        if by_speaker:
                            key = key_values[i]
                            split = speakers.get((speaker_source, str(key))) if key is not None else None
                            if split is None:
                                counts[leaf_path]["unassigned"] += 1
                                if unassigned_split == "drop":
                                    counts[leaf_path]["dropped"] += 1
                                    continue
                                split = (
                                    assign_split(seed, leaf_path, resolved, row_idx, ratios)
                                    if unassigned_split == "hash"
                                    else unassigned_split
                                )
                        else:
                            split = assign_split(seed, leaf_path, resolved, row_idx, ratios)

                        buckets[(split, leaf_path)].append(i)

                    for (split, leaf_path), indices in buckets.items():
                        subset = table.take(pa.array(indices, type=pa.int32()))
                        writer_for(split, leaf_path).append(subset)
                        counts[leaf_path][split] += len(indices)
                        counts[leaf_path]["total"] += len(indices)
    finally:
        for writer in writers.values():
            writer.close()

    summary = {
        "input_root": str(data_root),
        "output_root": str(output_root),
        "threshold": threshold,
        "seed": seed,
        "ratios": ratios,
        "text_label": text_label,
        "audio_label": audio_label,
        "routing": routing_summary,
        "speaker_routing": speaker_summary,
        "unassigned_split": unassigned_split,
        "per_leaf": {k: dict(v) for k, v in counts.items()},
        "totals": {
            split: sum(v.get(split, 0) for v in counts.values()) for split in SPLITS
        },
    }
    summary["totals"]["all"] = sum(summary["totals"][s] for s in SPLITS)
    ctx.write_json(output_root / "reports" / "summary.json", summary)
    ctx.write_json(ctx.stage_reports_dir(spec.id) / "summary.json", summary)

    result.counts = {"per_leaf": summary["per_leaf"], "totals": summary["totals"]}
    result.outputs["root"] = str(output_root)
    ctx.log(f"split done: {summary['totals']}")
    return result
