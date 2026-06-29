#!/usr/bin/env python3
from __future__ import annotations

import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq
from datasets import Dataset

INPUT_ROOT = Path('/home/MohammadNabulsi/whisper/.intermediate_data/omnilingual_selected/apc_north_levantine_all_splits')
OUTPUT_ROOT = Path('/home/MohammadNabulsi/whisper/data_cleaned_text_omnilingual_v2')
DISCOVERY_GLOB = 'data-*.arrow'
MIN_DURATION_SEC = 0.5
DROP_MISSING_DURATION = False
TEXT_COLUMN = 'raw_text'
DURATION_COLUMN = 'duration'
DATASET_NAME = 'omnilingual_apc'
PLACEHOLDER_TERMS = (
    'hesitation',
    'noise',
    'unintelligible',
    'unintelligable',
    'unitlegable',
)

ENGLISH_RE = re.compile(r'[A-Za-z]')
NUMBER_RE = re.compile(r'[0-9٠-٩۰-۹]')
BRACKET_RE = re.compile(r'(\[[^\]]+\]|<[^>]+>)')
DIACRITICS_RE = re.compile(r'[ؐ-ًؚ-ٰٟۖ-ۭ]')
TATWEEL = 'ـ'
ALEF_VARIANTS = {'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا'}
ARABIC_PUNCT_EXTRA = '،؛؟'
PLACEHOLDER_RE = re.compile(
    r'(?i)(?:<|\[)?\s*(?:' + '|'.join(re.escape(t) for t in PLACEHOLDER_TERMS) + r')\s*(?:>|\])?'
)
LONE_BRACKETS_RE = re.compile(r'[\[\]<>]')
SPACE_RE = re.compile(r'\s+')


def ensure_dir(path: Path) -> None:
    path.mkdir(parents=True, exist_ok=True)


def normalize_arabic_transcript(text: Any) -> str:
    if text is None:
        return ''
    text = str(text)
    text = unicodedata.normalize('NFKC', text)
    for src, dst in ALEF_VARIANTS.items():
        text = text.replace(src, dst)
    text = text.replace(TATWEEL, '')
    text = DIACRITICS_RE.sub('', text)
    out = []
    for ch in text:
        cat = unicodedata.category(ch)
        if cat.startswith('P') or ch in ARABIC_PUNCT_EXTRA:
            out.append(' ')
        else:
            out.append(ch)
    text = ''.join(out)
    return SPACE_RE.sub(' ', text).strip()


def has_english(text: Any) -> bool:
    return bool(ENGLISH_RE.search('' if text is None else str(text)))


def has_number(text: Any) -> bool:
    return bool(NUMBER_RE.search('' if text is None else str(text)))


def has_bracket_token(text: Any) -> bool:
    return bool(BRACKET_RE.search('' if text is None else str(text)))


def safe_float(x: Any):
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def stable_file_id(path: Path) -> str:
    rel = path.name
    h = hashlib.md5(rel.encode('utf-8')).hexdigest()[:10]
    slug = re.sub(r'[^A-Za-z0-9_.-]+', '_', rel)
    return f'{slug}__{h}'


def strip_placeholders_before_checks(text: Any) -> tuple[str, bool]:
    text = '' if text is None else str(text)
    stripped = PLACEHOLDER_RE.sub(' ', text)
    removed_any = stripped != text
    stripped = LONE_BRACKETS_RE.sub(' ', stripped)
    stripped = SPACE_RE.sub(' ', stripped).strip()
    return stripped, removed_any


def write_table(path: Path, table: pa.Table) -> None:
    ensure_dir(path.parent)
    pq.write_table(table, path, compression='zstd')


def main() -> None:
    ensure_dir(OUTPUT_ROOT)
    clean_dir = OUTPUT_ROOT / 'clean'
    dropped_dir = OUTPUT_ROOT / 'dropped'
    reports_dir = OUTPUT_ROOT / 'reports'
    ensure_dir(clean_dir)
    ensure_dir(dropped_dir)
    ensure_dir(reports_dir)

    shard_paths = sorted(INPUT_ROOT.glob(DISCOVERY_GLOB))
    if not shard_paths:
        raise SystemExit(f'No input files found under {INPUT_ROOT} with {DISCOVERY_GLOB}')

    totals = Counter()
    per_shard = []
    per_dataset = defaultdict(Counter)

    for shard_path in shard_paths:
        ds = Dataset.from_file(str(shard_path))
        rows = [dict(row) for row in ds]
        if not rows:
            continue

        clean_rows = []
        dropped_rows_by_reason = defaultdict(list)
        shard_counts = Counter(total=len(rows))

        for row in rows:
            original_text = row.get(TEXT_COLUMN)
            precheck_text, removed_placeholder = strip_placeholders_before_checks(original_text)
            normalized = normalize_arabic_transcript(precheck_text)
            eng = has_english(precheck_text)
            num = has_number(precheck_text)
            bracket_original = has_bracket_token(original_text)
            bracket_after = has_bracket_token(precheck_text)
            dur = safe_float(row.get(DURATION_COLUMN))
            missing_dur = dur is None
            too_short = False if missing_dur else dur < MIN_DURATION_SEC

            reason = None
            if eng:
                reason = 'contains_english'
            elif num:
                reason = 'contains_number'
            elif too_short:
                reason = 'audio_too_short'
            elif missing_dur and DROP_MISSING_DURATION:
                reason = 'missing_duration'

            out_row = dict(row)
            out_row['source_file'] = str(shard_path)
            out_row['precheck_text_v2'] = precheck_text
            out_row['manual_normalized_transcript'] = normalized
            out_row['flag_removed_placeholder_token_v2'] = removed_placeholder
            out_row['flag_contains_english'] = eng
            out_row['flag_contains_number'] = num
            out_row['flag_contains_bracket_token'] = bracket_after
            out_row['flag_contains_bracket_token_original'] = bracket_original
            out_row['flag_audio_too_short'] = too_short
            out_row['flag_missing_duration'] = missing_dur

            if reason is None:
                clean_rows.append(out_row)
                shard_counts['kept'] += 1
                per_dataset[DATASET_NAME]['kept'] += 1
            else:
                dropped_rows_by_reason[reason].append(out_row)
                shard_counts[f'dropped_{reason}'] += 1
                per_dataset[DATASET_NAME][f'dropped_{reason}'] += 1

            shard_counts['contains_english'] += int(eng)
            shard_counts['contains_number'] += int(num)
            shard_counts['contains_bracket_token_original'] += int(bracket_original)
            shard_counts['contains_bracket_token_after_precheck'] += int(bracket_after)
            shard_counts['removed_placeholder_token_v2'] += int(removed_placeholder)
            shard_counts['audio_too_short'] += int(too_short)
            shard_counts['missing_duration'] += int(missing_dur)

        file_id = stable_file_id(shard_path)
        if clean_rows:
            write_table(clean_dir / f'{file_id}__clean.parquet', pa.Table.from_pylist(clean_rows))
        for reason, reason_rows in dropped_rows_by_reason.items():
            write_table(dropped_dir / reason / f'{file_id}__dropped.parquet', pa.Table.from_pylist(reason_rows))

        totals.update(shard_counts)
        per_dataset[DATASET_NAME]['total'] += len(rows)
        per_shard.append({'shard_path': str(shard_path), **dict(shard_counts)})

    report = {
        'input_root': str(INPUT_ROOT),
        'output_root': str(OUTPUT_ROOT),
        'discovery_glob': DISCOVERY_GLOB,
        'min_duration_sec': MIN_DURATION_SEC,
        'drop_missing_duration': DROP_MISSING_DURATION,
        'text_column': TEXT_COLUMN,
        'placeholder_terms_removed_before_checks': list(PLACEHOLDER_TERMS),
        'removed_surrounding_bracket_chars_before_checks': True,
        'totals': dict(totals),
        'per_shard': per_shard,
        'per_dataset': {k: dict(v) for k, v in per_dataset.items()},
    }
    (reports_dir / 'cleaning_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    config = {
        'input_root': str(INPUT_ROOT),
        'output_root': str(OUTPUT_ROOT),
        'placeholder_terms_removed_before_checks': list(PLACEHOLDER_TERMS),
        'removed_surrounding_bracket_chars_before_checks': True,
        'text_column': TEXT_COLUMN,
        'duration_column': DURATION_COLUMN,
        'min_duration_sec': MIN_DURATION_SEC,
    }
    (reports_dir / 'config.json').write_text(json.dumps(config, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    with (reports_dir / 'per_dataset_report.csv').open('w', encoding='utf-8', newline='') as f:
        fieldnames = ['dataset', 'total', 'kept', 'dropped_contains_english', 'dropped_contains_number', 'dropped_audio_too_short', 'removed_placeholder_token_v2', 'contains_bracket_token_original', 'contains_bracket_token_after_precheck']
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for dataset, cnt in sorted(per_dataset.items()):
            writer.writerow({
                'dataset': dataset,
                'total': cnt.get('total', 0),
                'kept': cnt.get('kept', 0),
                'dropped_contains_english': cnt.get('dropped_contains_english', 0),
                'dropped_contains_number': cnt.get('dropped_contains_number', 0),
                'dropped_audio_too_short': cnt.get('dropped_audio_too_short', 0),
                'removed_placeholder_token_v2': totals.get('removed_placeholder_token_v2', 0),
                'contains_bracket_token_original': totals.get('contains_bracket_token_original', 0),
                'contains_bracket_token_after_precheck': totals.get('contains_bracket_token_after_precheck', 0),
            })
    manifest = {
        'input_files': [str(p) for p in shard_paths],
        'clean_shards': [str(p) for p in sorted(clean_dir.glob('*.parquet'))],
        'dropped_reason_dirs': [str(p) for p in sorted(dropped_dir.glob('*')) if p.is_dir()],
    }
    (reports_dir / 'run_manifest.json').write_text(json.dumps(manifest, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    summary_lines = [
        f'Input root: {INPUT_ROOT}',
        f'Output root: {OUTPUT_ROOT}',
        f'Total rows: {totals.get("total", 0)}',
        f'Kept rows: {totals.get("kept", 0)}',
        f'Dropped contains_english: {totals.get("dropped_contains_english", 0)}',
        f'Dropped contains_number: {totals.get("dropped_contains_number", 0)}',
        f'Dropped audio_too_short: {totals.get("dropped_audio_too_short", 0)}',
        f'Rows where placeholders were stripped before checks: {totals.get("removed_placeholder_token_v2", 0)}',
        f'Rows with bracket tokens in original text: {totals.get("contains_bracket_token_original", 0)}',
        f'Rows with bracket tokens after precheck stripping: {totals.get("contains_bracket_token_after_precheck", 0)}',
    ]
    (reports_dir / 'summary.txt').write_text('\n'.join(summary_lines) + '\n', encoding='utf-8')
    print('\n'.join(summary_lines))


if __name__ == '__main__':
    main()
