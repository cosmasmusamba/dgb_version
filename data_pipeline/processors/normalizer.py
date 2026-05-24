"""
data_pipeline/processors/normalizer.py
========================================
Stage 3: Text normalisation.

Operations (all configurable via runtime_config.json):
  - Unicode NFC normalisation
  - Whitespace collapse (tabs → space, multi-space → single)
  - Line-ending standardisation (CRLF → LF)
  - Control-character removal
  - URL and email scrubbing (configurable: remove / placeholder)
  - HTML entity decoding
  - Curly-quote → straight-quote substitution
  - Zero-width character removal
  - Byte-order-mark removal
  - Configurable max / min document length enforcement
"""
from __future__ import annotations

import html
import logging
import re
import unicodedata
from typing import Optional

from data_pipeline.core.document import Document, ProcessingStage

logger = logging.getLogger(__name__)

# ── Compiled regexes ──────────────────────────────────────────────────────────
_RE_CONTROL   = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
_RE_ZW        = re.compile(r"[\u200b\u200c\u200d\u2060\ufeff\u00ad]")
_RE_MULTI_NL  = re.compile(r"\n{3,}")
_RE_MULTI_SP  = re.compile(r"[ \t]{2,}")
_RE_URL       = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
_RE_EMAIL     = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9.\-]+\.[A-Za-z]{2,}")
_RE_CURLY_Q   = str.maketrans({
    "\u2018": "'", "\u2019": "'", "\u201c": '"', "\u201d": '"',
    "\u2013": "-", "\u2014": "--", "\u2026": "...",
})


class TextNormalizer:
    """
    Configurable text normalisation pipeline.

    Parameters
    ----------
    min_chars:          Reject documents shorter than this after normalisation.
    max_chars:          Truncate documents longer than this (0 = unlimited).
    remove_urls:        Replace URLs with __URL__ placeholder.
    remove_emails:      Replace emails with __EMAIL__ placeholder.
    fix_curly_quotes:   Substitute typographic quotes with ASCII equivalents.
    decode_html_entities: Decode &amp; &lt; etc.
    collapse_whitespace: Reduce multi-space and multi-newline runs.
    """

    def __init__(
        self,
        min_chars:              int   = 50,
        max_chars:              int   = 0,
        remove_urls:            bool  = False,
        remove_emails:          bool  = True,
        fix_curly_quotes:       bool  = True,
        decode_html_entities:   bool  = True,
        collapse_whitespace:    bool  = True,
    ) -> None:
        self._min_chars   = min_chars
        self._max_chars   = max_chars
        self._rem_urls    = remove_urls
        self._rem_emails  = remove_emails
        self._fix_quotes  = fix_curly_quotes
        self._decode_html = decode_html_entities
        self._collapse_ws = collapse_whitespace

    @classmethod
    def from_cfg(cls, cfg) -> "TextNormalizer":
        nc_raw = getattr(cfg, "normalizer", None) or {}
        if hasattr(nc_raw, "model_dump"):
            nc = nc_raw.model_dump()
        elif isinstance(nc_raw, dict):
            nc = nc_raw
        elif hasattr(nc_raw, "__dict__"):
            nc = dict(nc_raw.__dict__)
        else:
            nc = {}
        return cls(
            min_chars=nc.get("min_chars", 50),
            max_chars=nc.get("max_chars", 0),
            remove_urls=nc.get("remove_urls", False),
            remove_emails=nc.get("remove_emails", True),
            fix_curly_quotes=nc.get("fix_curly_quotes", True),
            decode_html_entities=nc.get("decode_html_entities", True),
            collapse_whitespace=nc.get("collapse_whitespace", True),
        )

    def process(self, doc: Document) -> Optional[Document]:
        """
        Normalise document text in-place.
        Returns None if the document should be rejected (too short).
        """
        text = doc.text

        # BOM removal
        text = text.lstrip("\ufeff")

        # CRLF → LF
        text = text.replace("\r\n", "\n").replace("\r", "\n")

        # HTML entity decoding
        if self._decode_html:
            text = html.unescape(text)

        # Unicode NFC
        text = unicodedata.normalize("NFC", text)

        # Zero-width / invisible characters
        text = _RE_ZW.sub("", text)

        # Control characters (keep \n and \t)
        text = _RE_CONTROL.sub("", text)

        # Curly quotes → ASCII
        if self._fix_quotes:
            text = text.translate(_RE_CURLY_Q)

        # URL and email scrubbing
        if self._rem_urls:
            text = _RE_URL.sub(" __URL__ ", text)
        if self._rem_emails:
            text = _RE_EMAIL.sub(" __EMAIL__ ", text)

        # Whitespace collapse
        if self._collapse_ws:
            text = _RE_MULTI_SP.sub(" ", text)
            text = _RE_MULTI_NL.sub("\n\n", text)

        text = text.strip()

        # Length enforcement
        if self._max_chars > 0 and len(text) > self._max_chars:
            text = text[: self._max_chars]

        if len(text) < self._min_chars:
            return doc.mark_rejected(
                f"too_short_after_normalisation:{len(text)}<{self._min_chars}"
            )

        doc.text          = text
        doc.char_count    = len(text)
        doc.word_count    = len(text.split())
        doc.token_estimate = len(text) // 4
        doc.stage         = ProcessingStage.NORMALIZED
        return doc

    def process_batch(self, docs: list) -> tuple:
        """Return (accepted, rejected) lists."""
        accepted, rejected = [], []
        for doc in docs:
            result = self.process(doc)
            (rejected if result is None or result.rejected else accepted).append(doc)
        return accepted, rejected
