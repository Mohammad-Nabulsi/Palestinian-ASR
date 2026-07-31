"""Discover the zero-shot eval targets:
- data/clean: every *test* split (masc_c_only, casablanca_jordanian, casablanca_palestinian)
- data/clean: the entire omnilingual_apc (omni) dataset, all shards (clean + recovered_clean)
- processed_layla_shards_v1/layla: the entire Layla (Jordanian) dataset, all shards
"""
from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
DATA_CLEAN_DIR = REPO_ROOT / "data" / "clean"
LAYLA_DIR = REPO_ROOT / "processed_layla_shards_v1" / "layla"


def discover_eval_targets() -> dict[str, list[Path]]:
    groups: dict[str, list[Path]] = {}
    for f in sorted(DATA_CLEAN_DIR.glob("*clean.parquet")):
        name = f.name
        is_omni = name.startswith("omnilingual_apc")
        is_test = "test" in name.lower()
        if not (is_omni or is_test):
            continue
        key = "omnilingual_apc_full" if is_omni else name.split("__", 1)[0]
        groups.setdefault(key, []).append(f)

    layla_files = sorted(LAYLA_DIR.glob("*.parquet"))
    if layla_files:
        groups["layla"] = layla_files

    return {k: sorted(v) for k, v in groups.items()}


if __name__ == "__main__":
    for key, files in discover_eval_targets().items():
        print(f"{key}: {len(files)} shard(s)")
        for f in files:
            print(f"  {f.name}")
