"""The zero-shot eval targets, as they exist on this box.

`datasets.discover_eval_targets()` globs `data/clean` and
`processed_layla_shards_v1`, which is the layout of the original GPU box. Here
the eval material was fetched separately -- Casablanca from the official
HuggingFace release, the rest out of R2 -- so this module names those files
directly instead.

    casablanca_palestinian / casablanca_jordanian
        UBC-NLP/Casablanca, the official `test` split, downloaded from
        HuggingFace (public, ungated). `validation` is present on disk too but
        is deliberately not an eval target.
    layla / omnilingual_apc
        curated_corpus/test/{layla,omni} out of R2 -- the cleaned, segmented
        versions that went into the corpus.
    masc_qasr_custom_test
        the speaker-disjoint MASC+QASR test block (see
        SPEAKER_DISJOINT_SELECTION.md), audio extracted from
        data_lev_custom_split_v1 by speaker.

Audio column shapes differ by source and `audio_io.decode_audio_cell` handles
all three: Casablanca and Layla store WAV, omni stores FLAC, both inside an
HF-style `{bytes, path}` struct; the custom test block stores raw PCM16 in
`audio_bytes` with a separate `sampling_rate`.
"""
from __future__ import annotations

import os
from pathlib import Path

EVAL_SETS_ROOT = Path(os.environ.get(
    "EVAL_SETS_ROOT", "/workspace/asr/Palestinian-ASR/eval_sets"))

# dataset key -> (glob root, patterns). Order is the order they are evaluated in.
SPEC: dict[str, tuple[str, list[str]]] = {
    "masc_qasr_custom_test": ("custom_test", ["masc_qasr_test.parquet"]),
    "casablanca_palestinian": ("casablanca_hf/Palestine", ["test-*.parquet"]),
    "casablanca_jordanian": ("casablanca_hf/Jordan", ["test-*.parquet"]),
    "layla": ("curated_corpus_test", ["layla_test.parquet"]),
    "omnilingual_apc": ("curated_corpus_test", ["omni_test.parquet"]),
}


def discover_eval_targets(root: Path | None = None) -> dict[str, list[Path]]:
    """key -> shard paths, skipping any target whose files are not present."""
    base = Path(root) if root is not None else EVAL_SETS_ROOT
    out: dict[str, list[Path]] = {}
    for key, (subdir, patterns) in SPEC.items():
        files: list[Path] = []
        for pattern in patterns:
            files.extend(sorted((base / subdir).glob(pattern)))
        files = [f for f in files if f.is_file()]
        if files:
            out[key] = files
    return out


if __name__ == "__main__":
    import pyarrow.parquet as pq

    total_rows = 0
    total_hours = 0.0
    for key, files in discover_eval_targets().items():
        rows = 0
        hours = 0.0
        for f in files:
            pf = pq.ParquetFile(f)
            rows += pf.metadata.num_rows
            col = "duration" if "duration" in pf.schema_arrow.names else None
            if col:
                hours += sum(x for x in pq.read_table(f, columns=[col]).to_pydict()[col] if x) / 3600
        total_rows += rows
        total_hours += hours
        print(f"{key:24} {len(files)} shard(s)  {rows:6,} rows  {hours:6.2f} h")
    print(f"{'TOTAL':24} {'':9}  {total_rows:6,} rows  {total_hours:6.2f} h")
