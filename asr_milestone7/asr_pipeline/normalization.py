"""Text normalization utilities for ASR metric computation.

The rules here are intentionally conservative for Arabic dialect ASR. They clean
spacing and punctuation for metric variants, but they do not convert dialectal
forms into MSA and do not globally normalize Arabic letters such as hamza forms,
maqṣūra, or tāʾ marbūṭa.
"""

from __future__ import annotations

import re
import unicodedata

# Common punctuation removed by the normalized metric variant. This includes
# Arabic punctuation but deliberately excludes Arabic letters and digits.
BASIC_PUNCTUATION = """.,!?;:،؛؟"'`“”‘’()[]{}<>/\\|*_+=~^%@#$&…-–—"""
BASIC_PUNCTUATION_RE = re.compile("[" + re.escape(BASIC_PUNCTUATION) + "]")
WHITESPACE_RE = re.compile(r"\s+")

# Arabic marks commonly used as diacritics. Tatweel is handled separately.
ARABIC_DIACRITIC_RE = re.compile(
    "["
    "\u0610-\u061A"
    "\u064B-\u065F"
    "\u0670"
    "\u06D6-\u06ED"
    "]"
)
TATWEEL = "\u0640"


def normalize_whitespace(text: str) -> str:
    """Trim leading/trailing whitespace and collapse internal whitespace."""
    return WHITESPACE_RE.sub(" ", str(text).strip())


def basic_normalize(text: str) -> str:
    """Apply basic ASR metric normalization.

    Operations:
    - trim whitespace
    - collapse repeated whitespace
    - remove common Arabic/Latin punctuation

    This does not convert dialect to MSA and does not globally convert Arabic
    letters such as ``أ/إ/آ`` to ``ا``, ``ى`` to ``ي``, or ``ه`` to ``ة``.
    """
    text = normalize_whitespace(text)
    text = BASIC_PUNCTUATION_RE.sub(" ", text)
    return normalize_whitespace(text)


def _remove_all_unicode_punctuation(text: str) -> str:
    """Remove any Unicode character whose category starts with P."""
    return "".join(" " if unicodedata.category(ch).startswith("P") else ch for ch in text)


def loose_normalize(
    text: str,
    *,
    remove_tatweel: bool = True,
    remove_arabic_diacritics: bool = True,
) -> str:
    """Apply a stronger punctuation/mark cleanup for loose metrics.

    Operations:
    - basic whitespace cleanup
    - remove all Unicode punctuation
    - optionally remove Arabic tatweel
    - optionally remove Arabic diacritics

    This remains orthography-preserving for dialect words: it does not rewrite
    dialect into MSA and does not globally normalize hamza forms, ``ى``, ``ي``,
    ``ه``, or ``ة``.
    """
    text = normalize_whitespace(text)
    text = _remove_all_unicode_punctuation(text)
    if remove_tatweel:
        text = text.replace(TATWEEL, "")
    if remove_arabic_diacritics:
        text = ARABIC_DIACRITIC_RE.sub("", text)
    return normalize_whitespace(text)
