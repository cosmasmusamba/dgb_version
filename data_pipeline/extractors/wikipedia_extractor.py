"""
data_pipeline/extractors/wikipedia_extractor.py
================================================
Wikipedia dump extractor — streams directly from Wikimedia dump servers
without requiring a full upfront download.

Supports:
  - Online streaming from dump index (multistream bz2)
  - Local .bz2 / .xml.bz2 / .txt file processing
  - XML article extraction with section parsing
  - MediaWiki markup stripping
  - Category, redirect, and disambiguation filtering
  - Resumable via byte-offset checkpointing
  - Configurable dump versions and language editions

Dump index URL format:
  https://dumps.wikimedia.org/{lang}wiki/{date}/
  e.g. https://dumps.wikimedia.org/enwiki/20260501/

Files processed (in order of preference):
  1. *-pages-articles-multistream.xml.bz2   (preferred — supports streaming)
  2. *-pages-articles.xml.bz2               (single-stream fallback)
"""
from __future__ import annotations

import bz2
import io
import logging
import re
import xml.etree.ElementTree as ET
from typing import AsyncGenerator, Iterator, List, Optional, Tuple

from data_pipeline.core.document import Document, ContentDomain, LicenseType, ProcessingStage
from data_pipeline.extractors.base_extractor import BaseExtractor, ExtractorConfig

logger = logging.getLogger(__name__)

# Wikimedia dump server
_DUMP_BASE  = "https://dumps.wikimedia.org"
_NAMESPACES = {"ns0": "http://www.mediawiki.org/xml/exports/0.10/"}

# Markup cleaning patterns (matches main data_cleaner but Wikipedia-specific)
_RE_TEMPLATE   = re.compile(r"\{\{[^}]*?\}\}", re.DOTALL)
_RE_TEMPLATE2  = re.compile(r"\{\{.*?\}\}", re.DOTALL)
_RE_FILE_LINK  = re.compile(r"\[\[(?:File|Image|Media):[^\]]*\]\]", re.IGNORECASE)
_RE_WIKILINK   = re.compile(r"\[\[(?:[^|\]]*\|)?([^\]]*)\]\]")
_RE_REF        = re.compile(r"<ref[^>]*>.*?</ref>", re.DOTALL | re.IGNORECASE)
_RE_COMMENT    = re.compile(r"<!--.*?-->", re.DOTALL)
_RE_TAG        = re.compile(r"<[^>]{1,80}>")
_RE_HEADING    = re.compile(r"^={1,6}\s*.+\s*={1,6}\s*$", re.MULTILINE)
_RE_MULTI_NL   = re.compile(r"\n{3,}")
_RE_MULTI_SP   = re.compile(r" {2,}")

# Redirects and non-article namespaces to skip
_SKIP_PREFIXES = (
    "Wikipedia:", "Template:", "Help:", "File:", "Category:",
    "Talk:", "User:", "Portal:", "Draft:", "Module:",
    "MediaWiki:", "TimedText:", "Special:",
)


def _clean_wikitext(text: str) -> str:
    """Strip MediaWiki markup from raw article text."""
    text = _RE_COMMENT.sub("", text)
    text = _RE_REF.sub("", text)
    text = _RE_FILE_LINK.sub("", text)
    # Iteratively remove nested templates
    for _ in range(4):
        text = _RE_TEMPLATE.sub("", text)
    text = _RE_TEMPLATE2.sub("", text)
    text = _RE_WIKILINK.sub(r"\1", text)
    text = _RE_TAG.sub("", text)
    text = _RE_HEADING.sub("", text)
    text = _RE_MULTI_NL.sub("\n\n", text)
    text = _RE_MULTI_SP.sub(" ", text)
    return text.strip()


class WikipediaExtractor(BaseExtractor):
    """
    Streams Wikipedia articles from Wikimedia dump files.

    Supports both online streaming and processing of already-downloaded
    local files in datasets/dgb1/wk_raw/ (the existing DGB format).
    """

    def __init__(self, cfg: ExtractorConfig, checkpoint_mgr=None, quota_mgr=None) -> None:
        super().__init__(cfg, checkpoint_mgr, quota_mgr)
        self._lang       = cfg.extra.get("wiki_lang", "en")
        self._dump_date  = cfg.extra.get("dump_date", "latest")
        self._min_len    = cfg.extra.get("min_article_chars", 500)
        self._max_len    = cfg.extra.get("max_article_chars", 200_000)
        self._local_dir  = cfg.extra.get("local_dir", None)

    @classmethod
    def build(cls, cfg_dict: dict, checkpoint_mgr=None, quota_mgr=None) -> "WikipediaExtractor":
        cfg = ExtractorConfig.from_dict(cfg_dict, source_name="wikipedia")
        cfg.domain  = ContentDomain.ENCYCLOPEDIC
        cfg.license = LicenseType.CC_BY_SA
        cfg.language = cfg_dict.get("language", "en")
        return cls(cfg, checkpoint_mgr, quota_mgr)

    async def stream(self) -> AsyncGenerator[Document, None]:
        """Yield Document objects from all configured Wikipedia sources."""
        # 1. Process local pre-downloaded .txt files (existing DGB wk_*.txt format)
        if self._local_dir:
            async for doc in self._stream_local_txt():
                yield doc
            return

        # 2. Stream dump URLs
        for url in self._cfg.dump_urls:
            async for doc in self._stream_dump_url(url):
                yield doc

        # 3. Auto-discover from Wikimedia dump server
        if not self._cfg.dump_urls and not self._local_dir:
            urls = await self._discover_dump_urls()
            for url in urls:
                async for doc in self._stream_dump_url(url):
                    yield doc

    # ── Local txt processing (existing DGB wk_*.txt files) ────────────

    async def _stream_local_txt(self) -> AsyncGenerator[Document, None]:
        """
        Process pre-cleaned wk_*.txt files in local_dir.
        Each line is treated as one training example.
        This is the fast path for the existing DGB dataset format.
        """
        import asyncio
        from modules.utils.file_handler import list_files
        local = self._cfg.extra.get("local_dir")
        if not local:
            return
        files = list_files(local, "*.txt")   # natural sort: wk_0, wk_1, … wk_10
        self._log.info("WikipediaExtractor: %d local files in %s", len(files), local)

        for fpath in files:
            resume_line = self._resume_offset(str(fpath))
            accepted = seen = 0
            with open(fpath, "r", encoding="utf-8", errors="replace") as fh:
                for li, line in enumerate(fh):
                    if li < resume_line:
                        continue
                    line = line.strip()
                    if not line:
                        continue
                    seen += 1
                    if len(line) < self._min_len:
                        continue
                    doc = self._make_doc(
                        text=line,
                        url=str(fpath),
                        metadata={"source_file": fpath.name, "line_idx": li},
                    )
                    doc.stage    = ProcessingStage.EXTRACTED
                    accepted += 1
                    self._update_checkpoint(str(fpath), li, accepted, seen)
                    yield doc
                    await asyncio.sleep(0)   # yield control to event loop

            self._log.info(
                "Local file done: %s  lines=%d  accepted=%d", fpath.name, seen, accepted
            )

    # ── Online streaming ───────────────────────────────────────────────

    async def _stream_dump_url(self, url: str) -> AsyncGenerator[Document, None]:
        """Stream a Wikipedia XML dump (.bz2) from a remote URL."""
        resume = self._resume_offset(url)
        buf    = b""
        bz2dec = bz2.BZ2Decompressor()
        xml_buf = io.StringIO()
        accepted = seen = 0
        byte_pos = resume

        self._log.info("WikipediaExtractor: streaming %s (resume=%d)", url, resume)

        async for chunk in self._stream_url(url, resume_offset=resume):
            byte_pos += len(chunk)
            try:
                decompressed = bz2dec.decompress(chunk)
            except Exception:
                bz2dec  = bz2.BZ2Decompressor()
                decompressed = b""

            xml_buf.write(decompressed.decode("utf-8", errors="replace"))

            # Extract complete <page> blocks from buffer
            content = xml_buf.getvalue()
            while True:
                start = content.find("<page>")
                end   = content.find("</page>")
                if start == -1 or end == -1 or end < start:
                    break
                page_xml = content[start: end + 7]
                content  = content[end + 7:]
                doc = self._parse_page(page_xml, url)
                seen += 1
                if doc:
                    accepted += 1
                    self._update_checkpoint(url, byte_pos, accepted, seen)
                    yield doc

            xml_buf = io.StringIO()
            xml_buf.write(content)

        self._log.info(
            "Wikipedia dump done: %s  seen=%d  accepted=%d", url, seen, accepted
        )

    def _parse_page(self, page_xml: str, source_url: str) -> Optional[Document]:
        """Parse one <page> XML block into a Document."""
        try:
            root = ET.fromstring(page_xml)
        except ET.ParseError:
            return None

        def find(tag: str):
            el = root.find(f".//{tag}")
            return el.text or "" if el is not None else ""

        title    = find("title")
        ns       = find("ns")
        redirect = root.find(".//redirect")

        # Skip non-article namespaces and redirects
        if ns != "0" or redirect is not None:
            return None
        if any(title.startswith(p) for p in _SKIP_PREFIXES):
            return None
        if any(marker in title.lower() for marker in
               ["disambiguation", "(disambiguation)", "list of"]):
            return None

        wikitext  = find("text")
        if not wikitext:
            return None
        clean     = _clean_wikitext(wikitext)
        if len(clean) < self._min_len:
            return None
        if len(clean) > self._max_len:
            clean = clean[: self._max_len]

        revision  = find("id")
        timestamp = find("timestamp")

        return self._make_doc(
            text=clean,
            title=title,
            url=f"https://{self._lang}.wikipedia.org/wiki/{title.replace(' ', '_')}",
            metadata={
                "revision_id": revision,
                "dump_url":    source_url,
            },
            source_timestamp=timestamp,
            dump_version=self._dump_date,
        )

    async def _discover_dump_urls(self) -> List[str]:
        """Discover the latest dump URLs from the Wikimedia dump index."""
        index_url = f"{_DUMP_BASE}/{self._lang}wiki/{self._dump_date}/"
        self._log.info("Discovering dump URLs: %s", index_url)
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                async with s.get(index_url, timeout=aiohttp.ClientTimeout(total=30)) as r:
                    text = await r.text()
            # Find multistream articles dump
            import re as _re
            matches = _re.findall(
                r'href="([^"]*pages-articles-multistream\.xml\.bz2)"', text
            )
            if matches:
                return [index_url + m for m in matches[:1]]
            # Fallback to regular articles dump
            matches = _re.findall(r'href="([^"]*pages-articles\.xml\.bz2)"', text)
            return [index_url + m for m in matches[:1]]
        except Exception as exc:
            self._log.warning("Cannot discover dump URLs: %s", exc)
            return []
