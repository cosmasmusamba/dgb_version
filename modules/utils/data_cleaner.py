"""
modules/utils/data_cleaner.py
==============================
Enterprise-grade text cleaning pipeline for Wikipedia dump ingestion.

Cleaning stages (in order):
  1. Unicode normalisation (NFC)
  2. Strip XML/HTML artifacts and MediaWiki markup
  3. Remove structural Wikipedia boilerplate (categories, navboxes, etc.)
  4. Filter by line length constraints
  5. Sentence-level quality heuristics (alpha ratio, symbol density, repetition)
  6. Within-batch exact deduplication via bloom-filter-style set hashing
"""
from __future__ import annotations

import logging
import re
import unicodedata
from typing import Generator, Iterable, List, Optional, Set, Tuple

logger = logging.getLogger(__name__)

# ── Compiled regexes (module-level for performance) ───────────────────────────
_RE_XML_TAG     = re.compile(r"<[^>]+>")
_RE_WIKILINK    = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")
_RE_TEMPLATE    = re.compile(r"\{\{[^}]*\}\}")
_RE_EXT_LINK    = re.compile(r"\[https?://[^\s\]]+(?:\s[^\]]+)?\]")
_RE_HEADING     = re.compile(r"^={1,6}.+={1,6}\s*$")
_RE_TABLE_ROW   = re.compile(r"^\s*[|!{]")
_RE_CATEGORY    = re.compile(r"^\s*(?:Category|File|Image|Media|Template):", re.IGNORECASE)
_RE_REDIRECT    = re.compile(r"^\s*#redirect", re.IGNORECASE)
_RE_BULLETS     = re.compile(r"^[\*#:;]+\s*")
_RE_MULTI_SPACE = re.compile(r" {2,}")
_RE_MULTI_NL    = re.compile(r"\n{3,}")
_RE_URL         = re.compile(r"https?://\S+")
_RE_REF         = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_RE_CONTROL     = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_CITATION    = re.compile(r"\[\d+\]")
_RE_REPEAT_CHAR = re.compile(r"(.)\1{4,}")   # 5+ repeated chars = noise


def _normalise_unicode(text: str) -> str:
    """NFC normalisation — collapses combining characters."""
    return unicodedata.normalize("NFC", text)


def _strip_markup(line: str) -> str:
    """Remove MediaWiki markup from a single line."""
    line = _RE_REF.sub("", line)
    line = _RE_XML_TAG.sub("", line)
    line = _RE_TEMPLATE.sub("", line)
    line = _RE_EXT_LINK.sub(r"", line)
    line = _RE_WIKILINK.sub(r"\1", line)
    line = _RE_URL.sub("", line)
    line = _RE_CITATION.sub("", line)
    line = _RE_BULLETS.sub("", line)
    line = _RE_CONTROL.sub("", line)
    return _RE_MULTI_SPACE.sub(" ", line).strip()


def _is_structural(line: str) -> bool:
    """Returns True if the line is Wikipedia structural noise to discard."""
    return bool(
        _RE_HEADING.match(line) or
        _RE_TABLE_ROW.match(line) or
        _RE_CATEGORY.match(line) or
        _RE_REDIRECT.match(line)
    )


def _quality_ok(
    line:     str,
    min_len:  int,
    max_len:  int,
) -> bool:
    """
    Multi-heuristic quality filter.

    Rejects lines that are:
    - Too short or too long
    - Insufficient alphabetic content (< 55%)
    - High symbol density (punctuation/symbols > 20%)
    - Repetitive characters (likely encoding artifacts)
    - All uppercase (likely headings or metadata that slipped through)
    """
    n = len(line)
    if n < min_len or n > max_len:
        return False

    alpha  = sum(1 for c in line if c.isalpha())
    punct  = sum(1 for c in line if not c.isalnum() and not c.isspace())
    digit  = sum(1 for c in line if c.isdigit())
    spaces = sum(1 for c in line if c.isspace())

    alpha_ratio = alpha / n
    punct_ratio = punct / n

    if alpha_ratio < 0.50:   # less than half the characters are letters
        return False
    if punct_ratio > 0.25:   # more than a quarter are punctuation/symbols
        return False
    if digit / n > 0.30:     # mostly numbers (tables, coordinate data)
        return False
    if _RE_REPEAT_CHAR.search(line):  # aaaaaaaaa type artifacts
        return False
    if line == line.upper() and alpha > 5:  # all caps
        return False

    return True


def clean_lines(
    lines:    Iterable[str],
    min_len:  int  = 20,
    max_len:  int  = 2_000,
    dedup:    bool = True,
) -> Generator[str, None, None]:
    """
    Clean a batch of raw text lines from a Wikipedia dump.

    Applies:
      1. Unicode normalisation
      2. Markup stripping
      3. Structural filtering
      4. Quality heuristics
      5. Exact deduplication (within this batch)

    Yields cleaned lines that pass all filters.
    """
    seen: Set[int] = set()

    for raw in lines:
        if not raw or not raw.strip():
            continue

        line = _normalise_unicode(raw.strip())

        if _is_structural(line):
            continue

        line = _strip_markup(line)
        if not line:
            continue

        if not _quality_ok(line, min_len, max_len):
            continue

        if dedup:
            h = hash(line.casefold())
            if h in seen:
                continue
            seen.add(h)

        yield line


def cleaning_stats(raw_count: int, cleaned_count: int) -> dict:
    """Return a summary dict of cleaning statistics."""
    removed     = raw_count - cleaned_count
    removed_pct = round((removed / max(raw_count, 1)) * 100, 2)
    return {
        "raw_lines":     raw_count,
        "cleaned_lines": cleaned_count,
        "removed_lines": removed,
        "removed_pct":   removed_pct,
    }
