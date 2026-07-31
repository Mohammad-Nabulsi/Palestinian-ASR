"""Guard the unified normalizer against the original per-dataset scripts.

Two kinds of check:

1. **Codepoint pinning.** The Arabic regex literals are extracted (textually, not by
   importing -- the originals are flat scripts with side effects) from
   ``scripts/reclean_omnilingual_v2.py`` and compared codepoint-for-codepoint with
   ``pipeline.textnorm``. This exists because a bidi-reordered edit of
   ``DIACRITICS_RE`` once turned its ranges into ``[U+0610-U+064B ...]``, which
   swallows the whole Arabic letter block and normalizes every transcript to "" --
   silently, with no exception and no empty-output warning anywhere downstream.

2. **Behavioral goldens.** Real Arabic must survive normalization non-empty, and the
   v2/v3 pre-check strategies must reproduce the documented v2 vs v3 difference.

Run:  python -m pytest tests/test_textnorm.py -q
      (or:  python tests/test_textnorm.py  for a dependency-free run)
"""
from __future__ import annotations

import re
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT))

from pipeline import textnorm  # noqa: E402

REFERENCE_SCRIPT = REPO_ROOT / "scripts" / "reclean_omnilingual_v2.py"

# name in pipeline.textnorm -> assignment prefix in the reference script
PINNED_PATTERNS = {
    "ENGLISH_RE": "ENGLISH_RE",
    "NUMBER_RE": "NUMBER_RE",
    "BRACKET_RE": "BRACKET_RE",
    "DIACRITICS_RE": "DIACRITICS_RE",
}


def _extract_assignment(source: str, name: str) -> str:
    for line in source.splitlines():
        stripped = line.strip()
        if stripped.startswith(f"{name} ") or stripped.startswith(f"{name}="):
            return stripped.split("=", 1)[1].strip()
    raise AssertionError(f"{name} not found in reference script")


def _pattern_literal(assignment: str) -> str:
    match = re.search(r"re\.compile\(\s*r?['\"](.*)['\"]\s*\)", assignment)
    assert match, f"could not parse pattern from: {assignment!r}"
    return match.group(1)


def test_regex_literals_match_original_codepoints() -> None:
    source = REFERENCE_SCRIPT.read_text(encoding="utf-8")
    for ours, theirs in PINNED_PATTERNS.items():
        expected = _pattern_literal(_extract_assignment(source, theirs))
        actual = getattr(textnorm, ours).pattern
        assert [ord(c) for c in actual] == [ord(c) for c in expected], (
            f"{ours} drifted from {REFERENCE_SCRIPT.name}:\n"
            f"  ours  : {[hex(ord(c)) for c in actual]}\n"
            f"  theirs: {[hex(ord(c)) for c in expected]}"
        )


def test_diacritics_class_does_not_cover_arabic_letters() -> None:
    """The specific failure mode: letters must never match the diacritics class."""
    for codepoint in range(0x0621, 0x064B):  # Arabic letters hamza..yeh
        char = chr(codepoint)
        assert not textnorm.DIACRITICS_RE.match(char), (
            f"DIACRITICS_RE matches Arabic letter U+{codepoint:04X} ({char}) -- "
            "the character class ranges are corrupted"
        )


def test_alef_and_punctuation_constants() -> None:
    assert textnorm.TATWEEL == "ـ"
    assert set(textnorm.ARABIC_PUNCT_EXTRA) == {"،", "؛", "؟"}
    assert textnorm.ALEF_VARIANTS == {
        "أ": "ا",
        "إ": "ا",
        "آ": "ا",
        "ٱ": "ا",
    }


def test_normalization_preserves_arabic() -> None:
    cases = {
        # (raw, expected)
        "مرحبا كيف حالك":
            "مرحبا كيف حالك",
        # diacritics stripped, alef variants folded, punctuation -> space
        "أَكَلتُ،":
            "اكلت",
        # tatweel removed
        "آهـــ":
            "اه",
    }
    for raw, expected in cases.items():
        actual = textnorm.normalize_arabic_transcript(raw)
        assert actual == expected, f"{raw!r} -> {actual!r}, expected {expected!r}"
        assert actual, "normalization produced an empty string for Arabic input"


def test_detectors() -> None:
    arabic = "مرحبا"
    assert textnorm.has_english(f"{arabic} hello")
    assert not textnorm.has_english(arabic)
    assert textnorm.has_number(f"{arabic} ٣")  # Arabic-Indic three
    assert textnorm.has_number(f"{arabic} 3")
    assert not textnorm.has_number(arabic)
    # BRACKET_RE matches whole tokens, not lone bracket characters.
    assert textnorm.has_bracket_token(f"{arabic} [laugh]")
    assert textnorm.has_bracket_token(f"{arabic} <noise>")
    assert not textnorm.has_bracket_token(f"{arabic} [")


def test_precheck_reproduces_v2_vs_v3_difference() -> None:
    arabic = "مرحبا"  # مرحبا
    text = f"{arabic} [background noise] {arabic}"

    # v2: placeholder terms + lone brackets. "noise" goes, "background" stays,
    # so the row still trips the English rule.
    v2 = textnorm.apply_precheck(text, ["placeholders"])
    assert textnorm.has_english(v2), f"expected leftover English in v2 output: {v2!r}"

    # v3: whole spans removed first, so nothing English survives -- this is exactly
    # the row class that recover_omnilingual_token_span_rows_v3.py rescued.
    v3 = textnorm.apply_precheck(text, ["spans", "placeholders"])
    assert not textnorm.has_english(v3), f"expected no English in v3 output: {v3!r}"
    assert arabic in v3

    # A fully-bracketed placeholder is consumed whole by v2 (no stray brackets).
    assert textnorm.apply_precheck(f"{arabic} [noise]", ["placeholders"]) == arabic


def test_unknown_precheck_strategy_raises() -> None:
    try:
        textnorm.apply_precheck("x", ["nope"])
    except ValueError as exc:
        assert "nope" in str(exc)
    else:
        raise AssertionError("expected ValueError for unknown strategy")


if __name__ == "__main__":
    failures = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except AssertionError as exc:
                failures += 1
                print(f"FAIL {name}: {exc}")
    print("ok" if not failures else f"{failures} failure(s)")
    sys.exit(1 if failures else 0)
