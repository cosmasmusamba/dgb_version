"""
data_pipeline/extractors/base_extractor.py
============================================
Abstract base class for all source extractors.

Every extractor must implement:
    stream() → AsyncGenerator[Document, None]

And may override:
    get_dump_urls()   — discover available dump files
    extract_one()     — parse a single stream chunk into Documents

The base class provides:
  - Streaming HTTP download with range-request resume
  - Exponential backoff retry on network failures
  - Checkpoint integration (byte-offset resume)
  - Rate throttling (configurable requests/sec)
  - Progress tracking
  - Memory-bounded chunk buffering
  - Graceful shutdown on SIGINT/SIGTERM
"""
from __future__ import annotations

import asyncio
import logging
import time
from abc import ABC, abstractmethod
from pathlib import Path
from typing import AsyncGenerator, Dict, Iterator, List, Optional

from data_pipeline.core.document import Document, ContentDomain, LicenseType
from data_pipeline.core.checkpoint import CheckpointManager, SourceCheckpoint

logger = logging.getLogger(__name__)


class ExtractorConfig:
    """Runtime config slice for one extractor source."""

    def __init__(
        self,
        source_name:        str,
        enabled:            bool  = True,
        dump_urls:          List[str] = None,
        max_docs:           int   = 0,       # 0 = unlimited
        max_bytes:          int   = 0,       # 0 = unlimited
        batch_size:         int   = 1000,
        download_timeout:   int   = 30,
        max_retries:        int   = 5,
        retry_backoff:      float = 2.0,
        rate_limit_rps:     float = 0.0,     # 0 = unlimited
        stream_chunk_bytes: int   = 65536,
        language:           str   = "en",
        domain:             str   = ContentDomain.UNKNOWN,
        license:            str   = LicenseType.UNKNOWN,
        extra:              Dict  = None,
    ) -> None:
        self.source_name        = source_name
        self.enabled            = enabled
        self.dump_urls          = dump_urls or []
        self.max_docs           = max_docs
        self.max_bytes          = max_bytes
        self.batch_size         = batch_size
        self.download_timeout   = download_timeout
        self.max_retries        = max_retries
        self.retry_backoff      = retry_backoff
        self.rate_limit_rps     = rate_limit_rps
        self.stream_chunk_bytes = stream_chunk_bytes
        self.language           = language
        self.domain             = domain
        self.license            = license
        self.extra              = extra or {}

    @classmethod
    def from_dict(cls, d: dict, source_name: str = "") -> "ExtractorConfig":
        return cls(
            source_name=d.get("source_name", source_name),
            enabled=d.get("enabled", True),
            dump_urls=d.get("dump_urls", []),
            max_docs=d.get("max_docs", 0),
            max_bytes=d.get("max_bytes", 0),
            batch_size=d.get("batch_size", 1000),
            download_timeout=d.get("download_timeout", 30),
            max_retries=d.get("max_retries", 5),
            retry_backoff=d.get("retry_backoff", 2.0),
            rate_limit_rps=d.get("rate_limit_rps", 0.0),
            stream_chunk_bytes=d.get("stream_chunk_bytes", 65536),
            language=d.get("language", "en"),
            domain=d.get("domain", ContentDomain.UNKNOWN),
            license=d.get("license", LicenseType.UNKNOWN),
            extra=d.get("extra", {}),
        )


class BaseExtractor(ABC):
    """
    Abstract base for all DGB data pipeline source extractors.

    Subclasses implement stream() which yields Document objects.
    The base class handles HTTP streaming, retries, checkpointing,
    and rate limiting.
    """

    def __init__(
        self,
        cfg:             ExtractorConfig,
        checkpoint_mgr:  Optional[CheckpointManager] = None,
        quota_mgr=None,
    ) -> None:
        self._cfg    = cfg
        self._ckpt   = checkpoint_mgr
        self._quota  = quota_mgr
        self._log    = logging.getLogger(f"dgb.pipeline.{cfg.source_name}")
        self._stop   = False
        self._docs_yielded = 0
        self._bytes_read   = 0

    @property
    def source_name(self) -> str:
        return self._cfg.source_name

    def stop(self) -> None:
        """Signal the extractor to stop after the current document."""
        self._stop = True

    @abstractmethod
    async def stream(self) -> AsyncGenerator[Document, None]:
        """
        Yield Document objects from this source.
        Must be implemented by every extractor subclass.
        Supports resume via self._resume_offset(url).
        """
        ...

    async def extract_batches(
        self, batch_size: Optional[int] = None
    ) -> AsyncGenerator[List[Document], None]:
        """
        Higher-level interface that yields batches of Documents.
        Handles quota checking and stop signals.
        """
        bs    = batch_size or self._cfg.batch_size
        batch: List[Document] = []

        async for doc in self.stream():
            if self._stop:
                break
            if self._cfg.max_docs > 0 and self._docs_yielded >= self._cfg.max_docs:
                self._log.info(
                    "%s: max_docs=%d reached — stopping",
                    self.source_name, self._cfg.max_docs,
                )
                break
            if self._quota and not self._quota.can_write(self.source_name):
                self._log.warning("%s: quota exceeded — stopping", self.source_name)
                break

            batch.append(doc)
            self._docs_yielded += 1

            if len(batch) >= bs:
                yield batch
                batch = []

        if batch:
            yield batch

    # ── HTTP streaming helpers ────────────────────────────────────────

    async def _stream_url(
        self, url: str, resume_offset: int = 0
    ) -> AsyncGenerator[bytes, None]:
        """
        Stream raw bytes from a URL with resume support and retries.
        Yields raw bytes chunks of stream_chunk_bytes.
        """
        try:
            import aiohttp
        except ImportError:
            raise ImportError("aiohttp required: pip install aiohttp")

        headers = {}
        if resume_offset > 0:
            headers["Range"] = f"bytes={resume_offset}-"
            self._log.info("%s: resuming %s at byte %d", self.source_name, url, resume_offset)

        delay  = 1.0
        for attempt in range(self._cfg.max_retries):
            try:
                timeout = aiohttp.ClientTimeout(total=self._cfg.download_timeout * 10,
                                                connect=self._cfg.download_timeout)
                async with aiohttp.ClientSession(timeout=timeout) as session:
                    async with session.get(url, headers=headers) as resp:
                        if resp.status not in (200, 206):
                            raise IOError(
                                f"HTTP {resp.status} for {url}"
                            )
                        content_len = resp.content_length
                        self._log.info(
                            "%s: streaming %s  size=%s",
                            self.source_name, url,
                            f"{content_len/1024**2:.1f}MB" if content_len else "unknown",
                        )
                        async for chunk in resp.content.iter_chunked(
                            self._cfg.stream_chunk_bytes
                        ):
                            if self._stop:
                                return
                            yield chunk
                            self._bytes_read += len(chunk)
                            if (self._cfg.rate_limit_rps > 0 and
                                    self._bytes_read % (1024 * 1024) == 0):
                                await asyncio.sleep(1.0 / self._cfg.rate_limit_rps)
                return   # success — exit retry loop

            except Exception as exc:
                if attempt < self._cfg.max_retries - 1:
                    self._log.warning(
                        "%s: download error attempt %d/%d: %s — retry in %.1fs",
                        self.source_name, attempt+1, self._cfg.max_retries, exc, delay,
                    )
                    await asyncio.sleep(delay)
                    delay = min(delay * self._cfg.retry_backoff, 300)
                else:
                    self._log.error(
                        "%s: download failed after %d attempts: %s",
                        self.source_name, self._cfg.max_retries, exc,
                    )
                    raise

    def _resume_offset(self, url: str) -> int:
        """Return resume byte offset from checkpoint, or 0."""
        if not self._ckpt:
            return 0
        cp = self._ckpt.get(self.source_name)
        return cp.get_stream(url).byte_offset

    def _update_checkpoint(
        self,
        url:       str,
        offset:    int,
        accepted:  int = 0,
        seen:      int = 0,
    ) -> None:
        if not self._ckpt:
            return
        self._ckpt.get(self.source_name).update_stream(
            url, offset, records_seen=seen, records_accepted=accepted
        )
        if accepted % 10_000 == 0:
            self._ckpt.save(self.source_name)

    def _make_doc(self, text: str, url: str = "", title: str = "",
                  metadata: dict = None, **kwargs) -> Document:
        """Convenience factory for extractors."""
        from data_pipeline.core.document import Document
        return Document(
            text=text,
            title=title,
            url=url,
            source_name=self._cfg.source_name,
            source_url=url,
            domain=self._cfg.domain,
            language=self._cfg.language,
            license=self._cfg.license,
            metadata=metadata or {},
            **kwargs,
        )