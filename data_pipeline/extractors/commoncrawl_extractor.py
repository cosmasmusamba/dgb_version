"""
data_pipeline/extractors/commoncrawl_extractor.py
===================================================
Common Crawl WARC extractor — streams WARC files from CC S3/HTTPS.
Processes gzip segments on-the-fly, never writes full WARCs to disk.
"""
from __future__ import annotations
import gzip, io, logging, re
from typing import AsyncGenerator, List, Optional
from data_pipeline.core.document import Document, ContentDomain, LicenseType, ProcessingStage
from data_pipeline.extractors.base_extractor import BaseExtractor, ExtractorConfig

logger = logging.getLogger(__name__)
_CC_BASE  = "https://data.commoncrawl.org"
_PATHS_URL = "https://data.commoncrawl.org/crawl-data/{crawl}/warc.paths.gz"

def _extract_text(html: str) -> str:
    try:
        import trafilatura
        r = trafilatura.extract(html, include_comments=False, favor_precision=True)
        if r: return r
    except Exception: pass
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for t in soup(["script","style","nav","footer","header","aside"]): t.decompose()
        return soup.get_text(separator="\n", strip=True)
    except Exception: pass
    return re.sub(r"<[^>]+>"," ", html)

class CommonCrawlExtractor(BaseExtractor):
    def __init__(self, cfg: ExtractorConfig, checkpoint_mgr=None, quota_mgr=None) -> None:
        super().__init__(cfg, checkpoint_mgr, quota_mgr)
        self._crawl      = cfg.extra.get("crawl_id", "CC-MAIN-2024-10")
        self._max_warcs  = cfg.extra.get("max_warcs", 10)
        self._min_chars  = cfg.extra.get("min_chars", 200)
        self._blocklist  = set(cfg.extra.get("domain_blocklist", []))

    @classmethod
    def build(cls, cfg_dict: dict, checkpoint_mgr=None, quota_mgr=None) -> "CommonCrawlExtractor":
        cfg = ExtractorConfig.from_dict(cfg_dict, source_name="commoncrawl")
        cfg.domain  = ContentDomain.WEB
        cfg.license = LicenseType.CC0
        return cls(cfg, checkpoint_mgr, quota_mgr)

    async def stream(self) -> AsyncGenerator[Document, None]:
        paths = await self._get_warc_paths()
        for i, path in enumerate(paths[:self._max_warcs]):
            if self._stop: break
            url = f"{_CC_BASE}/{path}"
            async for doc in self._stream_warc(url): yield doc
            self._log.info("CC WARC %d/%d done: %s", i+1, min(len(paths),self._max_warcs), path)

    async def _get_warc_paths(self) -> List[str]:
        url = _PATHS_URL.format(crawl=self._crawl)
        try:
            import aiohttp
            async with aiohttp.ClientSession() as s:
                async with s.get(url, timeout=aiohttp.ClientTimeout(total=60)) as r:
                    data = await r.read()
            paths = gzip.decompress(data).decode().splitlines()
            self._log.info("CC: %d WARC paths for %s", len(paths), self._crawl)
            return paths
        except Exception as exc:
            self._log.error("CC paths error: %s", exc)
            return self._cfg.dump_urls

    async def _stream_warc(self, url: str) -> AsyncGenerator[Document, None]:
        resume = self._resume_offset(url)
        buf = b""; accepted = seen = byte_pos = 0
        async for chunk in self._stream_url(url, resume_offset=resume):
            byte_pos += len(chunk); buf += chunk
            while True:
                # Find WARC record boundaries
                start = buf.find(b"WARC/1.0\r\n")
                if start < 0: break
                end = buf.find(b"WARC/1.0\r\n", start + 10)
                if end < 0: break
                record = buf[start:end]; buf = buf[end:]
                doc = self._parse_warc_record(record, url)
                seen += 1
                if doc:
                    accepted += 1
                    self._update_checkpoint(url, byte_pos, accepted, seen)
                    yield doc

    def _parse_warc_record(self, record: bytes, source_url: str) -> Optional[Document]:
        try:
            # Split headers from body
            header_end = record.find(b"\r\n\r\n")
            if header_end < 0: return None
            headers_raw = record[:header_end].decode("utf-8", errors="replace")
            body = record[header_end+4:]
            headers = {}
            for line in headers_raw.split("\r\n"):
                if ": " in line:
                    k, v = line.split(": ", 1)
                    headers[k.lower()] = v.strip()
            warc_type = headers.get("warc-type","")
            if warc_type != "response": return None
            target_uri = headers.get("warc-target-uri","")
            # Extract domain for blocklist check
            m = re.search(r"https?://([^/]+)", target_uri)
            if m and m.group(1) in self._blocklist: return None
            # Decompress if needed
            try:
                content = gzip.decompress(body)
            except Exception:
                content = body
            text_raw = content.decode("utf-8", errors="replace")
            # Strip HTTP response headers
            http_end = text_raw.find("\r\n\r\n")
            html = text_raw[http_end+4:] if http_end >= 0 else text_raw
            text = _extract_text(html)
            if len(text) < self._min_chars: return None
            doc = self._make_doc(
                text=text, url=target_uri,
                metadata={"warc_type": warc_type, "warc_date": headers.get("warc-date",""),
                          "dump_url": source_url, "crawl": self._crawl},
                source_timestamp=headers.get("warc-date"),
                dump_version=self._crawl,
            )
            doc.stage = ProcessingStage.EXTRACTED
            return doc
        except Exception: return None
