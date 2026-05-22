"""
data_pipeline/extractors/gutenberg_extractor.py
================================================
Project Gutenberg extractor — streams from the Gutenberg catalog
and downloads individual public-domain books as plain text.

Catalog URL: https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv
Book text:   https://www.gutenberg.org/files/{id}/{id}-0.txt  (UTF-8)
             https://www.gutenberg.org/files/{id}/{id}.txt    (fallback)
"""
from __future__ import annotations

import csv
import io
import logging
import re
from typing import AsyncGenerator, List

from data_pipeline.core.document import Document, ContentDomain, LicenseType, ProcessingStage
from data_pipeline.extractors.base_extractor import BaseExtractor, ExtractorConfig

logger = logging.getLogger(__name__)

_CATALOG_URL = "https://www.gutenberg.org/cache/epub/feeds/pg_catalog.csv"
_BOOK_URL    = "https://www.gutenberg.org/files/{id}/{id}-0.txt"
_BOOK_FALLBACK = "https://www.gutenberg.org/files/{id}/{id}.txt"

# Strip Gutenberg header/footer boilerplate
_START_MARKERS = [
    "*** START OF THE PROJECT GUTENBERG",
    "***START OF THE PROJECT GUTENBERG",
    "*** START OF THIS PROJECT GUTENBERG",
]
_END_MARKERS = [
    "*** END OF THE PROJECT GUTENBERG",
    "***END OF THE PROJECT GUTENBERG",
    "*** END OF THIS PROJECT GUTENBERG",
    "End of the Project Gutenberg",
]


def _strip_gutenberg_boilerplate(text: str) -> str:
    start = 0
    for marker in _START_MARKERS:
        idx = text.find(marker)
        if idx >= 0:
            nl = text.find("\n", idx)
            start = nl + 1 if nl >= 0 else idx + len(marker)
            break
    end = len(text)
    for marker in _END_MARKERS:
        idx = text.find(marker)
        if idx >= 0:
            end = idx
            break
    return text[start:end].strip()


class GutenbergExtractor(BaseExtractor):
    """Streams public-domain books from Project Gutenberg."""

    def __init__(self, cfg: ExtractorConfig, checkpoint_mgr=None, quota_mgr=None) -> None:
        super().__init__(cfg, checkpoint_mgr, quota_mgr)
        self._max_books   = cfg.extra.get("max_books", 0)
        self._lang_filter = cfg.extra.get("language_filter", ["en"])
        self._min_len     = cfg.extra.get("min_chars", 5_000)
        self._subject_filter = cfg.extra.get("subject_filter", [])

    @classmethod
    def build(cls, cfg_dict: dict, checkpoint_mgr=None, quota_mgr=None) -> "GutenbergExtractor":
        cfg = ExtractorConfig.from_dict(cfg_dict, source_name="gutenberg")
        cfg.domain  = ContentDomain.BOOKS
        cfg.license = LicenseType.PUBLIC_DOMAIN
        return cls(cfg, checkpoint_mgr, quota_mgr)

    async def stream(self) -> AsyncGenerator[Document, None]:
        books  = await self._load_catalog()
        done   = 0
        for book in books:
            if self._stop:
                break
            if self._max_books > 0 and done >= self._max_books:
                break
            doc = await self._fetch_book(book)
            if doc:
                done += 1
                yield doc
        self._log.info("Gutenberg: %d books extracted", done)

    async def _load_catalog(self) -> List[dict]:
        """Download and parse the Gutenberg CSV catalog."""
        self._log.info("Gutenberg: loading catalog from %s", _CATALOG_URL)
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                async with s.get(_CATALOG_URL, timeout=aiohttp.ClientTimeout(total=60)) as r:
                    text = await r.text(encoding="utf-8", errors="replace")
            books = []
            reader = csv.DictReader(io.StringIO(text))
            for row in reader:
                lang = row.get("Language", "en")
                if self._lang_filter and lang not in self._lang_filter:
                    continue
                if row.get("Type", "") != "Text":
                    continue
                subj = row.get("Subjects", "")
                if self._subject_filter and not any(
                    sf.lower() in subj.lower() for sf in self._subject_filter
                ):
                    continue
                books.append(row)
            self._log.info("Gutenberg catalog: %d matching books", len(books))
            return books
        except Exception as exc:
            self._log.error("Cannot load Gutenberg catalog: %s", exc)
            return []

    async def _fetch_book(self, book: dict) -> "Optional[Document]":
        gid  = book.get("Text#", "").strip()
        if not gid:
            return None

        resume = self._resume_offset(gid)
        if resume > 0:
            return None   # already processed

        for url_tmpl in [_BOOK_URL, _BOOK_FALLBACK]:
            url = url_tmpl.format(id=gid)
            try:
                buf = b""
                async for chunk in self._stream_url(url):
                    buf += chunk
                text = buf.decode("utf-8", errors="replace")
                text = _strip_gutenberg_boilerplate(text)
                if len(text) < self._min_len:
                    continue
                doc = self._make_doc(
                    text=text,
                    title=book.get("Title", ""),
                    url=f"https://www.gutenberg.org/ebooks/{gid}",
                    metadata={
                        "gutenberg_id": gid,
                        "author":       book.get("Authors", ""),
                        "subjects":     book.get("Subjects", ""),
                        "issued":       book.get("Issued", ""),
                    },
                    source_timestamp=book.get("Issued"),
                )
                doc.stage    = ProcessingStage.EXTRACTED
                doc.category = "books"
                self._update_checkpoint(gid, 1, 1, 1)
                return doc
            except Exception as exc:
                self._log.debug("Gutenberg %s fetch error: %s", gid, exc)
                continue
        return None