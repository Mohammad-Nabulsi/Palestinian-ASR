from pathlib import Path
import json
import re
import unicodedata
import hashlib
import shutil
import time
from collections import Counter, defaultdict

import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from tqdm.auto import tqdm

print("Imports ready.")


# =========================
# Main paths
# =========================

INPUT_ROOT = Path("/workspace/asr/Palestinian-ASR/data")
OUTPUT_ROOT = Path("/workspace/asr/Palestinian-ASR/data_cleaned_text_qasr_casablanca_omni_v1")

# Target only the requested datasets.
# This keeps the cleaning logic identical while avoiding obvious
# non-training artifacts like QASR index JSONL files and HF cache Arrow files.
DISCOVERY_GLOBS = [
    "processed_qasr_segments/train/*.arrow",
    "casablanca_jordanian/*.parquet",
    "casablanca_palestinian/*.parquet",
    "omnilingual_apc/data-*.arrow",
]

# =========================
# Cleaning behavior
# =========================

MIN_DURATION_SEC = 0.5
BATCH_SIZE = 20_000
LOG_EVERY_ROWS = 50_000
COMPRESSION = "zstd"

# If True, output folder is deleted before running.
OVERWRITE_OUTPUT = True

# If set to an integer, only process the first N discovered files.
# Use this for a tiny test first, e.g. DRY_RUN_MAX_FILES = 2
DRY_RUN_MAX_FILES = None

# If duration column is missing, we cannot apply the <0.5s rule without decoding.
# Recommended for fast pass: keep rows with missing duration but report them.
DROP_MISSING_DURATION = False

TEXT_COLUMN_CANDIDATES = [
    "transcript",
    "transcription",
    "text",
    "raw_text",
    "sentence",
    "normalized_text",
]

DURATION_COLUMN_CANDIDATES = [
    "duration",
    "duration_sec",
    "duration_seconds",
    "audio_duration",
    "length_sec",
]

DATASET_COLUMN_CANDIDATES = [
    "dataset",
    "source",
    "corpus",
    "config",
    "dialect",
    "language",
]

print("INPUT_ROOT:", INPUT_ROOT)
print("OUTPUT_ROOT:", OUTPUT_ROOT)
print("DISCOVERY_GLOBS:")
for pat in DISCOVERY_GLOBS:
    print(" -", pat)


ENGLISH_RE = re.compile(r"[A-Za-z]")
NUMBER_RE = re.compile(r"[0-9٠-٩۰-۹]")
BRACKET_RE = re.compile(r"(\[[^\]]+\]|<[^>]+>)")

DIACRITICS_RE = re.compile(
    r"[\u0610-\u061A\u064B-\u065F\u0670\u06D6-\u06ED]"
)

TATWEEL = "\u0640"

ALEF_VARIANTS = {
    "أ": "ا",
    "إ": "ا",
    "آ": "ا",
    "ٱ": "ا",
}

ARABIC_PUNCT_EXTRA = "،؛؟"


def normalize_arabic_transcript(text) -> str:
    """
    Manual ASR transcript normalization:
    - NFKC Unicode normalization
    - Alef variants -> bare alef
    - remove diacritics
    - remove punctuation
    - remove tatweel
    - collapse whitespace
    """
    if text is None:
        return ""

    text = str(text)
    text = unicodedata.normalize("NFKC", text)

    for src, dst in ALEF_VARIANTS.items():
        text = text.replace(src, dst)

    text = text.replace(TATWEEL, "")
    text = DIACRITICS_RE.sub("", text)

    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith("P") or ch in ARABIC_PUNCT_EXTRA:
            out.append(" ")
        else:
            out.append(ch)

    text = "".join(out)
    text = re.sub(r"\s+", " ", text).strip()
    return text


def has_english(text) -> bool:
    return bool(ENGLISH_RE.search("" if text is None else str(text)))


def has_number(text) -> bool:
    return bool(NUMBER_RE.search("" if text is None else str(text)))


def has_bracket_token(text) -> bool:
    return bool(BRACKET_RE.search("" if text is None else str(text)))


# Tiny sanity check
examples = [
    "أكلتُ التفاحةَ، وقال: آهـــ!",
    "مرحبا [laugh] كيفك؟",
    "test عربي",
    "عندي ٣ كتب",
]

for x in examples:
    print("RAW :", x)
    print("NORM:", normalize_arabic_transcript(x))
    print("english:", has_english(x), "number:", has_number(x), "bracket:", has_bracket_token(x))
    print()


def log(msg: str):
    ts = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line)
    if "REPORTS_DIR" in globals():
        REPORTS_DIR.mkdir(parents=True, exist_ok=True)
        with (REPORTS_DIR / "run.log").open("a", encoding="utf-8") as f:
            f.write(line + "\n")


def find_first_column(schema_names, candidates):
    names = set(schema_names)
    for col in candidates:
        if col in names:
            return col
    return None


def safe_float(x):
    if x is None:
        return None
    try:
        if pd.isna(x):
            return None
    except Exception:
        pass
    try:
        return float(x)
    except Exception:
        return None


def dataset_from_file(path: Path) -> str:
    """
    Fallback dataset name when no dataset/source/corpus/config column exists.
    Uses the closest parent folder name.
    """
    try:
        rel = path.relative_to(INPUT_ROOT)
        if len(rel.parts) >= 2:
            return rel.parts[0]
    except Exception:
        pass
    return path.parent.name or "unknown"


def stable_file_id(path: Path) -> str:
    try:
        rel = str(path.relative_to(INPUT_ROOT))
    except Exception:
        rel = str(path)
    h = hashlib.md5(rel.encode("utf-8")).hexdigest()[:10]
    slug = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel.replace("/", "__"))
    return f"{slug}__{h}"


class TableShardWriter:
    """
    Lazily opens ParquetWriter objects and appends tables.
    Keeps one writer per shard path.
    """
    def __init__(self, compression="zstd"):
        self.compression = compression
        self.writers = {}

    def write(self, path: Path, table: pa.Table):
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

    def close(self):
        for writer in self.writers.values():
            writer.close()
        self.writers.clear()


def add_or_replace_column(table: pa.Table, name: str, values, pa_type=pa.string()):
    arr = pa.array(values, type=pa_type)
    if name in table.column_names:
        idx = table.column_names.index(name)
        return table.set_column(idx, name, arr)
    return table.append_column(name, arr)


def counter_to_percent(n, d):
    return float(n) / float(d) * 100.0 if d else 0.0


pattern_matches = {}
input_files = []

for pattern in DISCOVERY_GLOBS:
    matches = sorted(INPUT_ROOT.glob(pattern))
    pattern_matches[pattern] = matches
    input_files.extend(matches)

# Preserve discovery order while removing accidental duplicates.
input_files = list(dict.fromkeys(input_files))

if DRY_RUN_MAX_FILES is not None:
    input_files = input_files[:DRY_RUN_MAX_FILES]

parquet_files = [p for p in input_files if p.suffix == ".parquet"]
arrow_files = [p for p in input_files if p.suffix == ".arrow"]
jsonl_files = [p for p in input_files if p.suffix == ".jsonl"]

print("Discovery summary:")
for pattern, matches in pattern_matches.items():
    print(f"- {pattern}: {len(matches):,} file(s)")

print(f"Found parquet files: {len(parquet_files):,}")
print(f"Found arrow files  : {len(arrow_files):,}")
print(f"Found jsonl files  : {len(jsonl_files):,}")
print(f"Will process files : {len(input_files):,}")

for p in input_files[:10]:
    print("-", p)

if not input_files:
    raise FileNotFoundError(
        "No .parquet, .arrow, or .jsonl files matched DISCOVERY_GLOBS under "
        f"{INPUT_ROOT}"
    )


def process_table_batch(
    table: pa.Table,
    source_file: Path,
    clean_path: Path,
    dropped_base_dir: Path,
    writer: TableShardWriter,
    totals: Counter,
    per_dataset: dict,
):
    schema_names = table.column_names

    text_col = find_first_column(schema_names, TEXT_COLUMN_CANDIDATES)
    duration_col = find_first_column(schema_names, DURATION_COLUMN_CANDIDATES)
    dataset_col = find_first_column(schema_names, DATASET_COLUMN_CANDIDATES)

    n = table.num_rows
    if n == 0:
        return

    if text_col is None:
        # No transcript means unusable for ASR fine-tuning.
        reason = "missing_transcript"
        source_file_col = [str(source_file)] * n
        table2 = add_or_replace_column(table, "source_file", source_file_col)
        writer.write(dropped_base_dir / reason / f"{stable_file_id(source_file)}__dropped.parquet", table2)

        totals["total"] += n
        totals[f"dropped_{reason}"] += n
        fallback_dataset = dataset_from_file(source_file)
        per_dataset[fallback_dataset]["total"] += n
        per_dataset[fallback_dataset][f"dropped_{reason}"] += n
        return

    text_values = table[text_col].to_pylist()

    if duration_col is not None:
        duration_values = table[duration_col].to_pylist()
    else:
        duration_values = [None] * n

    if dataset_col is not None:
        dataset_values = table[dataset_col].to_pylist()
        dataset_values = [
            str(x) if x is not None and str(x).strip() else dataset_from_file(source_file)
            for x in dataset_values
        ]
    else:
        fallback_dataset = dataset_from_file(source_file)
        dataset_values = [fallback_dataset] * n

    normalized = []
    english_flags = []
    number_flags = []
    bracket_flags = []
    too_short_flags = []
    missing_duration_flags = []
    drop_reasons = []
    keep_mask = []

    for text, dur_raw in zip(text_values, duration_values):
        norm = normalize_arabic_transcript(text)
        eng = has_english(text)
        num = has_number(text)
        bracket = has_bracket_token(text)

        dur = safe_float(dur_raw)
        missing_dur = dur is None
        too_short = False if missing_dur else dur < MIN_DURATION_SEC

        reason = None
        if eng:
            reason = "contains_english"
        elif num:
            reason = "contains_number"
        elif too_short:
            reason = "audio_too_short"
        elif missing_dur and DROP_MISSING_DURATION:
            reason = "missing_duration"

        normalized.append(norm)
        english_flags.append(eng)
        number_flags.append(num)
        bracket_flags.append(bracket)
        too_short_flags.append(too_short)
        missing_duration_flags.append(missing_dur)
        drop_reasons.append(reason)
        keep_mask.append(reason is None)

    table = add_or_replace_column(table, "manual_normalized_transcript", normalized, pa.string())
    table = add_or_replace_column(table, "source_file", [str(source_file)] * n, pa.string())

    # Optional useful audit flags.
    table = add_or_replace_column(table, "flag_contains_bracket_token", bracket_flags, pa.bool_())
    table = add_or_replace_column(table, "flag_contains_english", english_flags, pa.bool_())
    table = add_or_replace_column(table, "flag_contains_number", number_flags, pa.bool_())
    table = add_or_replace_column(table, "flag_audio_too_short", too_short_flags, pa.bool_())
    table = add_or_replace_column(table, "flag_missing_duration", missing_duration_flags, pa.bool_())

    # Counters
    for ds, eng, num, bracket, short, missing_dur, reason in zip(
        dataset_values,
        english_flags,
        number_flags,
        bracket_flags,
        too_short_flags,
        missing_duration_flags,
        drop_reasons,
    ):
        totals["total"] += 1
        per_dataset[ds]["total"] += 1

        if eng:
            totals["contains_english"] += 1
            per_dataset[ds]["contains_english"] += 1

        if num:
            totals["contains_number"] += 1
            per_dataset[ds]["contains_number"] += 1

        if bracket:
            totals["contains_bracket_token"] += 1
            per_dataset[ds]["contains_bracket_token"] += 1

        if short:
            totals["audio_too_short"] += 1
            per_dataset[ds]["audio_too_short"] += 1

        if missing_dur:
            totals["missing_duration"] += 1
            per_dataset[ds]["missing_duration"] += 1

        if reason is None:
            totals["kept"] += 1
            per_dataset[ds]["kept"] += 1
        else:
            totals[f"dropped_{reason}"] += 1
            per_dataset[ds][f"dropped_{reason}"] += 1

    # Write kept rows.
    clean_table = table.filter(pa.array(keep_mask, type=pa.bool_()))
    writer.write(clean_path, clean_table)

    # Write dropped rows by reason.
    unique_reasons = sorted({r for r in drop_reasons if r is not None})
    for reason in unique_reasons:
        mask = [r == reason for r in drop_reasons]
        dropped_table = table.filter(pa.array(mask, type=pa.bool_()))
        dropped_path = dropped_base_dir / reason / f"{stable_file_id(source_file)}__dropped.parquet"
        writer.write(dropped_path, dropped_table)


def iter_parquet_batches(path: Path):
    pf = pq.ParquetFile(path)
    for batch in pf.iter_batches(batch_size=BATCH_SIZE):
        yield pa.Table.from_batches([batch])


def iter_arrow_batches(path: Path):
    """
    Reads Hugging Face / Apache Arrow IPC files.

    Tries file format first, then stream format.
    This avoids decoding audio and only reads the Arrow rows/batches.
    """
    try:
        with pa.memory_map(str(path), "r") as source:
            reader = pa.ipc.open_file(source)
            for i in range(reader.num_record_batches):
                yield pa.Table.from_batches([reader.get_batch(i)])
        return
    except Exception:
        pass

    with pa.memory_map(str(path), "r") as source:
        reader = pa.ipc.open_stream(source)
        for batch in reader:
            yield pa.Table.from_batches([batch])


def iter_jsonl_batches(path: Path):
    # JSONL support is included, but parquet/arrow is preferred for your large ASR data.
    rows = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            if line.strip():
                rows.append(json.loads(line))
            if len(rows) >= BATCH_SIZE:
                yield pa.Table.from_pylist(rows)
                rows = []
    if rows:
        yield pa.Table.from_pylist(rows)


CLEAN_DIR = OUTPUT_ROOT / "clean"
DROPPED_DIR = OUTPUT_ROOT / "dropped"
REPORTS_DIR = OUTPUT_ROOT / "reports"

if OVERWRITE_OUTPUT and OUTPUT_ROOT.exists():
    shutil.rmtree(OUTPUT_ROOT)

CLEAN_DIR.mkdir(parents=True, exist_ok=True)
DROPPED_DIR.mkdir(parents=True, exist_ok=True)
REPORTS_DIR.mkdir(parents=True, exist_ok=True)

config = {
    "input_root": str(INPUT_ROOT),
    "discovery_globs": DISCOVERY_GLOBS,
    "output_root": str(OUTPUT_ROOT),
    "min_duration_sec": MIN_DURATION_SEC,
    "batch_size": BATCH_SIZE,
    "compression": COMPRESSION,
    "drop_missing_duration": DROP_MISSING_DURATION,
    "text_column_candidates": TEXT_COLUMN_CANDIDATES,
    "duration_column_candidates": DURATION_COLUMN_CANDIDATES,
    "dataset_column_candidates": DATASET_COLUMN_CANDIDATES,
    "skipped_audio_decoding": True,
    "skipped_rms_check": True,
    "skipped_resampling": True,
    "skipped_audio_normalization": True,
}
(REPORTS_DIR / "config.json").write_text(json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8")

totals = Counter()
per_dataset = defaultdict(Counter)

writer = TableShardWriter(compression=COMPRESSION)

processed_since_log = 0
last_total = 0
start_time = time.time()

log("START cleaning")
log(f"Input files: {len(input_files):,}")
log(f"Output root: {OUTPUT_ROOT}")

try:
    for file_idx, path in enumerate(tqdm(input_files, desc="Files"), start=1):
        file_id = stable_file_id(path)
        clean_path = CLEAN_DIR / f"{file_id}__clean.parquet"

        log(f"FILE {file_idx:,}/{len(input_files):,}: {path}")

        if path.suffix == ".parquet":
            batch_iter = iter_parquet_batches(path)
        elif path.suffix == ".arrow":
            batch_iter = iter_arrow_batches(path)
        elif path.suffix == ".jsonl":
            batch_iter = iter_jsonl_batches(path)
        else:
            continue

        for table in batch_iter:
            process_table_batch(
                table=table,
                source_file=path,
                clean_path=clean_path,
                dropped_base_dir=DROPPED_DIR,
                writer=writer,
                totals=totals,
                per_dataset=per_dataset,
            )

            processed_since_log = totals["total"] - last_total
            if processed_since_log >= LOG_EVERY_ROWS:
                elapsed = time.time() - start_time
                rows_per_sec = totals["total"] / max(elapsed, 1e-9)
                dropped = totals["total"] - totals["kept"]
                log(
                    f"PROGRESS rows={totals['total']:,} "
                    f"kept={totals['kept']:,} "
                    f"dropped={dropped:,} "
                    f"rows/sec={rows_per_sec:,.1f}"
                )
                last_total = totals["total"]

finally:
    writer.close()

elapsed = time.time() - start_time
log(f"DONE rows={totals['total']:,} elapsed_min={elapsed/60:.2f}")
print(dict(totals))


def build_dataset_report_rows(per_dataset):
    rows = []
    for ds, cnt in sorted(per_dataset.items()):
        total = cnt["total"]
        dropped_total = total - cnt["kept"]
        row = {
            "dataset": ds,
            "total": total,
            "kept": cnt["kept"],
            "dropped_total": dropped_total,
            "pct_kept": counter_to_percent(cnt["kept"], total),
            "pct_dropped": counter_to_percent(dropped_total, total),

            "contains_english": cnt["contains_english"],
            "pct_contains_english": counter_to_percent(cnt["contains_english"], total),

            "contains_number": cnt["contains_number"],
            "pct_contains_number": counter_to_percent(cnt["contains_number"], total),

            "contains_bracket_token": cnt["contains_bracket_token"],
            "pct_contains_bracket_token": counter_to_percent(cnt["contains_bracket_token"], total),

            "audio_too_short": cnt["audio_too_short"],
            "pct_audio_too_short": counter_to_percent(cnt["audio_too_short"], total),

            "missing_duration": cnt["missing_duration"],
            "pct_missing_duration": counter_to_percent(cnt["missing_duration"], total),
        }

        # Include all explicit dropped reason counts.
        for k, v in sorted(cnt.items()):
            if k.startswith("dropped_"):
                row[k] = v
                row[f"pct_{k}"] = counter_to_percent(v, total)

        rows.append(row)
    return rows


per_dataset_rows = build_dataset_report_rows(per_dataset)
per_dataset_df = pd.DataFrame(per_dataset_rows)

report = {
    "config": config,
    "totals": dict(totals),
    "elapsed_seconds": elapsed,
    "elapsed_minutes": elapsed / 60,
    "per_dataset": per_dataset_rows,
}

(REPORTS_DIR / "cleaning_report.json").write_text(
    json.dumps(report, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

per_dataset_df.to_csv(REPORTS_DIR / "per_dataset_report.csv", index=False)

manifest = {
    "input_files": [str(p) for p in input_files],
    "clean_dir": str(CLEAN_DIR),
    "dropped_dir": str(DROPPED_DIR),
    "reports_dir": str(REPORTS_DIR),
    "clean_shards": [str(p) for p in sorted(CLEAN_DIR.glob("*.parquet"))],
    "dropped_reason_dirs": [str(p) for p in sorted(DROPPED_DIR.glob("*")) if p.is_dir()],
}

(REPORTS_DIR / "run_manifest.json").write_text(
    json.dumps(manifest, indent=2, ensure_ascii=False),
    encoding="utf-8",
)

print("Reports written to:", REPORTS_DIR)
print(per_dataset_df)


print("Clean shards:", len(list(CLEAN_DIR.glob("*.parquet"))))

print("\nDropped folders:")
for reason_dir in sorted(DROPPED_DIR.glob("*")):
    if reason_dir.is_dir():
        shards = list(reason_dir.glob("*.parquet"))
        print(f"- {reason_dir.name}: {len(shards):,} shard(s)")

print("\nReport files:")
for p in sorted(REPORTS_DIR.glob("*")):
    print("-", p)


clean_shards = sorted(CLEAN_DIR.glob("*.parquet"))

if clean_shards:
    sample_df = pd.read_parquet(clean_shards[0]).head(10)
    cols = [
        c for c in [
            "transcript",
            "transcription",
            "text",
            "raw_text",
            "manual_normalized_transcript",
            "duration",
            "source_file",
            "flag_contains_bracket_token",
        ]
        if c in sample_df.columns
    ]
    print(sample_df[cols])
else:
    print("No clean shards found.")

