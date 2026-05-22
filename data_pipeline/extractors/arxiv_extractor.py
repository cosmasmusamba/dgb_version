"""
data_pipeline/extractors/arxiv_extractor.py
============================================
arXiv bulk dataset extractor.

Streams from Kaggle arXiv dataset JSON or the arXiv S3 bulk access:
  s3://arxiv/src/...  (requires AWS credentials)
  https://www.kaggle.com/datasets/Cornell-University/arxiv (JSON metadata + abstracts)

This extractor uses the public JSON metadata file which contains abstracts,
titles, categories, and authors for all arXiv papers — no AWS credentials needed.

JSON metadata URL (mirror):
  https://huggingface.co/datasets/Cornell-University/arxiv/resolve/main/arxiv-metadata-oai-snapshot.json

Each document is: title + abstract [+ introduction if full-text available]
"""
from __future__ import annotations

import json
import logging
from typing import AsyncGenerator, List

from data_pipeline.core.document import Document, ContentDomain, LicenseType, ProcessingStage
from data_pipeline.extractors.base_extractor import BaseExtractor, ExtractorConfig

logger = logging.getLogger(__name__)

_HF_ARXIV_URL = (
    "https://huggingface.co/datasets/Cornell-University/arxiv/resolve/main/"
    "arxiv-metadata-oai-snapshot.json"
)

# arXiv category → domain mapping
_CATEGORY_DOMAINS = {
    "cs":      "computer_science",
    "math":    "mathematics",
    "physics": "physics",
    "astro":   "astronomy",
    "bio":     "biology",
    "chem":    "chemistry",
    "econ":    "economics",
    "stat":    "statistics",
}


class ArxivExtractor(BaseExtractor):
    """Streams academic paper abstracts from the arXiv bulk metadata snapshot."""

    def __init__(self, cfg: ExtractorConfig, checkpoint_mgr=None, quota_mgr=None) -> None:
        super().__init__(cfg, checkpoint_mgr, quota_mgr)
        self._min_abstract_len = cfg.extra.get("min_abstract_chars", 200)
        self._category_filter  = set(cfg.extra.get("categories", []))  # e.g. {"cs", "math"}
        self._include_title    = cfg.extra.get("include_title", True)
        self._urls             = cfg.dump_urls or [_HF_ARXIV_URL]

    @classmethod
    def build(cls, cfg_dict: dict, checkpoint_mgr=None, quota_mgr=None) -> "ArxivExtractor":
        cfg = ExtractorConfig.from_dict(cfg_dict, source_name="arxiv")
        cfg.domain  = ContentDomain.ACADEMIC
        cfg.license = LicenseType.CC_BY
        return cls(cfg, checkpoint_mgr, quota_mgr)

    async def stream(self) -> AsyncGenerator[Document, None]:
        for url in self._urls:
            async for doc in self._stream_snapshot(url):
                yield doc

    async def _stream_snapshot(self, url: str) -> AsyncGenerator[Document, None]:
        """Stream the newline-delimited JSON snapshot."""
        resume   = self._resume_offset(url)
        buf      = b""
        accepted = seen = line_idx = 0
        byte_pos = resume

        self._log.info("arXiv: streaming %s (resume_line≈%d)", url, resume)

        async for chunk in self._stream_url(url, resume_offset=0):
            byte_pos += len(chunk)
            buf += chunk

            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line_idx += 1
                if line_idx < resume:
                    continue
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                except json.JSONDecodeError:
                    continue

                seen += 1
                doc  = self._parse_record(record, url)
                if doc:
                    accepted += 1
                    self._update_checkpoint(url, line_idx, accepted, seen)
                    yield doc

        self._log.info("arXiv snapshot done: seen=%d accepted=%d", seen, accepted)

    def _parse_record(self, record: dict, source_url: str) -> "Optional[Document]":
        abstract = (record.get("abstract") or "").strip()
        title    = (record.get("title") or "").strip()
        cats     = (record.get("categories") or "").split()

        if not abstract or len(abstract) < self._min_abstract_len:
            return None

        if self._category_filter:
            # At least one category must match
            prefixes = {c.split(".")[0] for c in cats}
            if not (prefixes & self._category_filter):
                return None

        text = f"{title}\n\n{abstract}" if self._include_title and title else abstract
        primary_cat = cats[0].split(".")[0] if cats else "unknown"
        domain_label = _CATEGORY_DOMAINS.get(primary_cat, ContentDomain.ACADEMIC)

        arxiv_id = record.get("id", "")
        doc = self._make_doc(
            text=text,
            title=title,
            url=f"https://arxiv.org/abs/{arxiv_id}",
            metadata={
                "arxiv_id":    arxiv_id,
                "categories":  cats,
                "authors":     record.get("authors", ""),
                "journal_ref": record.get("journal-ref", ""),
                "doi":         record.get("doi", ""),
                "dump_url":    source_url,
            },
            source_timestamp=record.get("update_date"),
            dump_version=record.get("update_date", ""),
        )
        doc.stage    = ProcessingStage.EXTRACTED
        doc.category = primary_cat
        doc.domain   = domain_label
        return doc