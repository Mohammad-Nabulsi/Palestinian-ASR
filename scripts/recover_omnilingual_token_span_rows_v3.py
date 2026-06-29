#!/usr/bin/env python3
from __future__ import annotations

import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.parquet as pq

INPUT_DIR = Path('/home/MohammadNabulsi/whisper/data_cleaned_text_omnilingual_v2/dropped/contains_english')
OUTPUT_ROOT = Path('/home/MohammadNabulsi/whisper/data_cleaned_text_omnilingual_v3_recovered_from_v2')
ENGLISH_RE = re.compile(r'[A-Za-z]')
NUMBER_RE = re.compile(r'[0-9٠-٩۰-۹]')
DIACRITICS_RE = re.compile(r'[ؐ-ًؚ-ٰٟۖ-ۭ]')
TATWEEL = 'ـ'
ALEF_VARIANTS = {'أ': 'ا', 'إ': 'ا', 'آ': 'ا', 'ٱ': 'ا'}
ARABIC_PUNCT_EXTRA = '،؛؟'
PAREN_SPAN_RE = re.compile(r'\([^)]*\)')
BRACKET_SPAN_RE = re.compile(r'\[[^\]]*\]')
ANGLE_SPAN_RE = re.compile(r'<[^>]*>')
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
    return SPACE_RE.sub(' ', ''.join(out)).strip()


def strip_token_spans(text: Any) -> str:
    text = '' if text is None else str(text)
    text = ANGLE_SPAN_RE.sub(' ', text)
    text = BRACKET_SPAN_RE.sub(' ', text)
    text = PAREN_SPAN_RE.sub(' ', text)
    return SPACE_RE.sub(' ', text).strip()


def has_english(text: Any) -> bool:
    return bool(ENGLISH_RE.search('' if text is None else str(text)))


def has_number(text: Any) -> bool:
    return bool(NUMBER_RE.search('' if text is None else str(text)))


def main() -> None:
    recovered_dir = OUTPUT_ROOT / 'recovered_clean'
    still_dropped_dir = OUTPUT_ROOT / 'still_contains_english'
    reports_dir = OUTPUT_ROOT / 'reports'
    ensure_dir(recovered_dir)
    ensure_dir(still_dropped_dir)
    ensure_dir(reports_dir)

    totals = Counter()
    examples = []

    for path in sorted(INPUT_DIR.glob('*.parquet')):
        pf = pq.ParquetFile(path)
        recovered_rows = []
        still_dropped_rows = []
        for batch in pf.iter_batches(batch_size=64):
            for row in batch.to_pylist():
                totals['input_rows'] += 1
                raw_text = row.get('raw_text')
                cleaned = strip_token_spans(raw_text)
                eng_after = has_english(cleaned)
                num_after = has_number(cleaned)
                out = dict(row)
                out['precheck_text_v3'] = cleaned
                out['manual_normalized_transcript_v3'] = normalize_arabic_transcript(cleaned)
                out['flag_contains_english_v3'] = eng_after
                out['flag_contains_number_v3'] = num_after
                out['flag_recovered_from_v2_contains_english_v3'] = (not eng_after)
                if not eng_after:
                    recovered_rows.append(out)
                    totals['recovered_rows'] += 1
                    if len(examples) < 20:
                        examples.append({
                            'file': path.name,
                            'segment_id': row.get('segment_id'),
                            'raw_text': raw_text,
                            'precheck_text_v3': cleaned,
                        })
                else:
                    still_dropped_rows.append(out)
                    totals['still_contains_english'] += 1
        if recovered_rows:
            pq.write_table(pa.Table.from_pylist(recovered_rows), recovered_dir / path.name, compression='zstd')
        if still_dropped_rows:
            pq.write_table(pa.Table.from_pylist(still_dropped_rows), still_dropped_dir / path.name, compression='zstd')

    report = {
        'input_dir': str(INPUT_DIR),
        'output_root': str(OUTPUT_ROOT),
        'rule': 'remove full (...) / [...] / <...> spans before English check',
        'totals': dict(totals),
        'recovered_examples': examples,
    }
    (reports_dir / 'recovery_report.json').write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    lines = [
        f"Input dropped-English rows from v2: {totals.get('input_rows', 0)}",
        f"Recovered rows after removing full token spans: {totals.get('recovered_rows', 0)}",
        f"Still containing English after span removal: {totals.get('still_contains_english', 0)}",
    ]
    (reports_dir / 'summary.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    print('\n'.join(lines))


if __name__ == '__main__':
    main()
