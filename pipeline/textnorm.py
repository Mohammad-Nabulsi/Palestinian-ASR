"""Shared Arabic text normalization and content detectors.

This module is the single source of truth for logic that was previously copy-pasted
into every cleaning script/notebook in this repo:

- ``.logs/clean_broad_v1.py``
- ``.logs/clean_targeted_qasr_casablanca_omni_v1.py``
- ``.logs/clean_qasr_part2.py``
- ``scripts/reclean_omnilingual_v2.py``
- ``scripts/reclean_omnilingual_v3.py``
- ``scripts/recover_omnilingual_token_span_rows_v3.py``

The normalization is byte-for-byte equivalent to the version those files carried,
so re-running the unified pipeline reproduces the historical outputs.
"""
from __future__ import annotations

import re
import unicodedata
from typing import Any, Iterable, Sequence

# WARNING: do not retype the Arabic literals below by hand. Editing them through a
# bidirectional-text-rendering editor can silently reorder the codepoints inside a
# character class; one such reorder turns DIACRITICS_RE's ranges into
# `[U+0610-U+064B ...]`, which swallows the entire Arabic letter block U+0621-U+064A
# and normalizes every real transcript to the empty string -- with no error raised.
# `tests/test_textnorm.py` pins these codepoint-for-codepoint against the original
# scripts; run it after touching this block.
ENGLISH_RE = re.compile(r"[A-Za-z]")
NUMBER_RE = re.compile(r"[0-9٠-٩۰-۹]")  # ASCII + Arabic-Indic + Persian
BRACKET_RE = re.compile(r"(\[[^\]]+\]|<[^>]+>)")  # a whole [token] / <token>, not a lone char
DIACRITICS_RE = re.compile(r"[ؐ-ًؚ-ٰٟۖ-ۭ]")
SPACE_RE = re.compile(r"\s+")

PAREN_SPAN_RE = re.compile(r"\([^)]*\)")
BRACKET_SPAN_RE = re.compile(r"\[[^\]]*\]")
ANGLE_SPAN_RE = re.compile(r"<[^>]*>")

TATWEEL = "ـ"
ALEF_VARIANTS = {
    "أ": "ا",  # alef with hamza above
    "إ": "ا",  # alef with hamza below
    "آ": "ا",  # alef with madda above
    "ٱ": "ا",  # alef wasla
}
ARABIC_PUNCT_EXTRA = "،؛؟"  # comma, semicolon, question mark

# Placeholder words the Omnilingual APC transcripts use for non-speech events.
# Removing them before the English check is what separated the v2 reclean from
# the original fast pass, which dropped those rows as "contains_english".
DEFAULT_PLACEHOLDER_TERMS = (
    "hesitation",
    "noise",
    "unintelligible",
    "unintelligable",
    "unitlegable",
)

LONE_BRACKETS_RE = re.compile(r"[\[\]<>]")


def _placeholder_regex(terms: Sequence[str]) -> re.Pattern[str]:
    """Match a placeholder term plus any bracket that directly encloses it.

    Kept identical to ``scripts/reclean_omnilingual_v2.py``: the optional
    ``(?:<|\\[)?`` / ``(?:>|\\])?`` wrappers mean ``[noise]`` is consumed whole
    rather than leaving stray brackets behind.
    """
    alternatives = "|".join(re.escape(term) for term in terms)
    return re.compile(rf"(?i)(?:<|\[)?\s*(?:{alternatives})\s*(?:>|\])?")


DEFAULT_PLACEHOLDER_RE = _placeholder_regex(DEFAULT_PLACEHOLDER_TERMS)


def normalize_arabic_transcript(text: Any) -> str:
    """Manual ASR transcript normalization.

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

    return SPACE_RE.sub(" ", "".join(out)).strip()


def has_english(text: Any) -> bool:
    return bool(ENGLISH_RE.search("" if text is None else str(text)))


def has_number(text: Any) -> bool:
    return bool(NUMBER_RE.search("" if text is None else str(text)))


def has_bracket_token(text: Any) -> bool:
    return bool(BRACKET_RE.search("" if text is None else str(text)))


def strip_placeholder_terms(
    text: Any,
    placeholder_re: re.Pattern[str] = DEFAULT_PLACEHOLDER_RE,
) -> str:
    """Remove placeholder words and then lone bracket characters (v2 behavior)."""
    if text is None:
        return ""
    stripped = placeholder_re.sub(" ", str(text))
    stripped = LONE_BRACKETS_RE.sub(" ", stripped)
    return SPACE_RE.sub(" ", stripped).strip()


def strip_token_spans(text: Any) -> str:
    """Remove whole ``<...>``, ``[...]`` and ``(...)`` spans (v3 behavior)."""
    if text is None:
        return ""
    stripped = ANGLE_SPAN_RE.sub(" ", str(text))
    stripped = BRACKET_SPAN_RE.sub(" ", stripped)
    stripped = PAREN_SPAN_RE.sub(" ", stripped)
    return SPACE_RE.sub(" ", stripped).strip()


#: Named pre-check strategies, applied in listed order before the drop rules run.
#: ``[]``                        -> historical fast pass (broad / targeted / part2)
#: ``["placeholders"]``          -> historical Omnilingual reclean v2
#: ``["spans", "placeholders"]`` -> historical Omnilingual v3 / token-span recovery
PRECHECK_STRATEGIES = {
    "placeholders": strip_placeholder_terms,
    "spans": strip_token_spans,
}


def apply_precheck(text: Any, strategies: Iterable[str]) -> str:
    """Run the named pre-check strategies in order and return the resulting text."""
    current = "" if text is None else str(text)
    for name in strategies:
        try:
            fn = PRECHECK_STRATEGIES[name]
        except KeyError:
            raise ValueError(
                f"Unknown precheck strategy {name!r}. "
                f"Known: {sorted(PRECHECK_STRATEGIES)}"
            ) from None
        current = fn(current)
    return current
