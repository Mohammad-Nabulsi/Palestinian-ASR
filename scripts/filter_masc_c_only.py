from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.parquet as pq


def filter_shard(input_path: Path, output_path: Path) -> tuple[int, int]:
    table = pq.read_table(input_path)
    total_rows = table.num_rows
    filtered = table.filter(pc.equal(table.column("type"), pa.scalar("c")))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    pq.write_table(filtered, output_path)
    return total_rows, filtered.num_rows


def copy_metadata_files(source_root: Path, output_root: Path) -> None:
    for name in ["README.md", ".gitattributes"]:
        src = source_root / name
        if src.exists():
            shutil.copy2(src, output_root / name)



def main() -> None:
    parser = argparse.ArgumentParser(description="Filter MASC parquet shards to only keep rows where type == 'c'.")
    parser.add_argument(
        "--source",
        type=Path,
        default=Path("MASC-Arabic2"),
        help="Source MASC dataset root containing the data/ directory.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("data") / "masc_c_only",
        help="Output directory for the filtered dataset.",
    )
    args = parser.parse_args()

    source_root = args.source.resolve()
    output_root = args.output.resolve()
    source_data_dir = source_root / "data"
    output_data_dir = output_root / "data"

    if not source_data_dir.exists():
        raise SystemExit(f"Missing source data directory: {source_data_dir}")

    output_root.mkdir(parents=True, exist_ok=True)
    output_data_dir.mkdir(parents=True, exist_ok=True)
    copy_metadata_files(source_root, output_root)

    total_rows = 0
    kept_rows = 0
    shard_count = 0

    for input_path in sorted(source_data_dir.glob("*.parquet")):
        output_path = output_data_dir / input_path.name
        shard_total, shard_kept = filter_shard(input_path, output_path)
        total_rows += shard_total
        kept_rows += shard_kept
        shard_count += 1
        print(f"{input_path.name}: kept {shard_kept}/{shard_total} rows")

    print(f"Filtered {shard_count} shards")
    print(f"Kept {kept_rows}/{total_rows} rows where type == 'c'")


if __name__ == "__main__":
    main()
