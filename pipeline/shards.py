"""Shard-level IO helpers shared by every pipeline stage.

Consolidates the parquet/arrow/jsonl readers, the lazy multi-shard parquet writer,
and the ``stable_file_id`` naming scheme that the three fast-cleaning scripts each
carried their own copy of.
"""
from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any, Iterator, Sequence

import pyarrow as pa
import pyarrow.parquet as pq

SUPPORTED_SUFFIXES = (".parquet", ".arrow", ".jsonl")


def find_first_column(schema_names: Sequence[str], candidates: Sequence[str]) -> str | None:
    names = set(schema_names)
    for col in candidates:
        if col in names:
            return col
    return None


def safe_float(x: Any) -> float | None:
    if x is None:
        return None
    try:
        return float(x)
    except (TypeError, ValueError):
        return None


def stable_file_id(path: Path, root: Path) -> str:
    """Deterministic, collision-resistant output name for one input shard.

    The md5 is taken over the path *relative to root*, so two different input roots
    produce disjoint ids even for identically named shards. That is what let the
    QASR part-2 run be merged next to the original QASR run without collisions.
    """
    try:
        rel = str(path.relative_to(root))
    except ValueError:
        rel = str(path)
    h = hashlib.md5(rel.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel.replace("/", "__"))
    return f"{slug}__{h}"


def dataset_from_file(path: Path, root: Path) -> str:
    """Fallback dataset name: the first path component under the input root."""
    try:
        rel = path.relative_to(root)
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except ValueError:
        pass
    return path.parent.name or "unknown"


def count_rows(path: Path) -> int:
    if path.suffix == ".parquet":
        return pq.ParquetFile(path).metadata.num_rows
    return sum(table.num_rows for table in iter_table_batches(path))


def iter_parquet_batches(path: Path, batch_size: int) -> Iterator[pa.Table]:
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=batch_size):
        yield pa.Table.from_batches([batch])


def iter_arrow_batches(path: Path, batch_size: int) -> Iterator[pa.Table]:
    """Read Hugging Face / Apache Arrow IPC files without decoding audio.

    Tries the random-access file format first, then falls back to the stream format.
    """
    try:
        with pa.memory_map(str(path), "r") as source:
            reader = pa.ipc.open_file(source)
            for i in range(reader.num_record_batches):
                yield pa.Table.from_batches([reader.get_batch(i)])
        return
    except pa.ArrowInvalid:
        pass

    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_stream(source)
        for batch in reader:
            yield pa.Table.from_batches([batch])


def iter_jsonl_batches(path: Path, batch_size: int) -> Iterator[pa.Table]:
    rows: list[dict] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= batch_size:
                yield pa.Table.from_pylist(rows)
                rows = []
    if rows:
        yield pa.Table.from_pylist(rows)


def iter_table_batches(path: Path, batch_size: int = 20_000) -> Iterator[pa.Table]:
    """Dispatch on suffix and yield the file's contents as Arrow tables."""
    if path.suffix == ".parquet":
        yield from iter_parquet_batches(path, batch_size)
    elif path.suffix == ".arrow":
        yield from iter_arrow_batches(path, batch_size)
    elif path.suffix == ".jsonl":
        yield from iter_jsonl_batches(path, batch_size)
    else:
        raise ValueError(f"Unsupported input file type: {path}")


def add_or_replace_column(
    table: pa.Table,
    name: str,
    values: Sequence[Any],
    pa_type: pa.DataType = pa.string(),
) -> pa.Table:
    arr = pa.array(values, type=pa_type)
    if name in table.column_names:
        idx = table.column_names.index(name)
        return table.set_column(idx, name, arr)
    return table.append_column(name, arr)


def align_table_to_schema(table: pa.Table, schema: pa.Schema) -> pa.Table:
    """Reorder/backfill columns so heterogeneous shards can share one writer."""
    arrays = []
    for field in schema:
        if field.name in table.column_names:
            arrays.append(table.column(field.name).cast(field.type))
        else:
            arrays.append(pa.nulls(table.num_rows, type=field.type))
    return pa.Table.from_arrays(arrays, schema=schema)


def unify_schema(tables_or_paths: Sequence[Path]) -> pa.Schema | None:
    """Union the schemas of several shards, first-seen field type wins."""
    fields: list[pa.Field] = []
    seen: set[str] = set()
    for path in tables_or_paths:
        if path.suffix == ".parquet":
            schema = pq.ParquetFile(path).schema_arrow
        else:
            first = next(iter_table_batches(path, batch_size=1), None)
            if first is None:
                continue
            schema = first.schema
        for field in schema:
            if field.name not in seen:
                seen.add(field.name)
                fields.append(field)
    return pa.schema(fields) if fields else None


class TableShardWriter:
    """Lazily opens one ``ParquetWriter`` per output path and appends tables."""

    def __init__(self, compression: str = "zstd") -> None:
        self.compression = compression
        self.writers: dict[str, pq.ParquetWriter] = {}

    def write(self, path: Path, table: pa.Table | None) -> None:
        if table is None or table.num_rows == 0:
            return

        path.parent.mkdir(parents=True, exist_ok=True)
        key = str(path)
        if key not in self.writers:
            self.writers[key] = pq.ParquetWriter(
                where=path,
                schema=table.schema,
                compression=self.compression,
                use_dictionary=True,
            )
        self.writers[key].write_table(table)

    def close(self) -> None:
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()

    def __enter__(self) -> "TableShardWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()


class RollingShardWriter:
    """Writes rows into size-capped shards named ``{prefix}-{n:05d}.parquet``."""

    def __init__(
        self,
        output_dir: Path,
        prefix: str,
        rows_per_shard: int = 50_000,
        compression: str = "zstd",
    ) -> None:
        self.output_dir = output_dir
        self.prefix = prefix
        self.rows_per_shard = rows_per_shard
        self.compression = compression
        self.shard_index = 0
        self.rows_in_shard = 0
        self.total_rows = 0
        self.written_paths: list[Path] = []
        self._writer: pq.ParquetWriter | None = None
        self._schema: pa.Schema | None = None

    def _open(self, schema: pa.Schema) -> None:
        self.output_dir.mkdir(parents=True, exist_ok=True)
        path = self.output_dir / f"{self.prefix}-{self.shard_index:05d}.parquet"
        self._writer = pq.ParquetWriter(path, schema, compression=self.compression)
        self._schema = schema
        self.written_paths.append(path)
        self.rows_in_shard = 0

    def append(self, table: pa.Table) -> None:
        if table.num_rows == 0:
            return
        if self._writer is None:
            self._open(table.schema)
        elif self.rows_in_shard >= self.rows_per_shard:
            self._writer.close()
            self.shard_index += 1
            self._open(self._schema or table.schema)

        assert self._writer is not None and self._schema is not None
        self._writer.write_table(align_table_to_schema(table, self._schema))
        self.rows_in_shard += table.num_rows
        self.total_rows += table.num_rows

    def close(self) -> None:
        if self._writer is not None:
            self._writer.close()
            self._writer = None

    def __enter__(self) -> "RollingShardWriter":
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
