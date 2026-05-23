"""
data_pipeline/extractors/github_extractor.py
=============================================
GitHub code extractor — streams from GH Archive or GitHub API.
Produces documents of the form: <filename>\n\n<code> for training.
"""
from __future__ import annotations
import json, logging, re
from typing import AsyncGenerator, List, Optional
from data_pipeline.core.document import Document, ContentDomain, LicenseType, ProcessingStage
from data_pipeline.extractors.base_extractor import BaseExtractor, ExtractorConfig

logger = logging.getLogger(__name__)
_MAX_FILE_BYTES = 100_000

# Language extensions to include
_ALLOWED_EXTS = {
    ".py",".js",".ts",".java",".cpp",".c",".h",".cs",".go",
    ".rs",".rb",".php",".swift",".kt",".scala",".sh",".sql",
    ".md",".rst",".txt",
}

class GitHubExtractor(BaseExtractor):
    def __init__(self, cfg: ExtractorConfig, checkpoint_mgr=None, quota_mgr=None) -> None:
        super().__init__(cfg, checkpoint_mgr, quota_mgr)
        self._exts   = set(cfg.extra.get("extensions", list(_ALLOWED_EXTS)))
        self._min_ch = cfg.extra.get("min_chars", 100)
        self._max_ch = cfg.extra.get("max_chars", 50_000)
        self._token  = cfg.extra.get("github_token", "")

    @classmethod
    def build(cls, cfg_dict: dict, checkpoint_mgr=None, quota_mgr=None) -> "GitHubExtractor":
        cfg = ExtractorConfig.from_dict(cfg_dict, source_name="github")
        cfg.domain  = ContentDomain.CODE
        cfg.license = LicenseType.APACHE_2
        return cls(cfg, checkpoint_mgr, quota_mgr)

    async def stream(self) -> AsyncGenerator[Document, None]:
        # Stream from dump URLs (GH Archive JSONL dumps or custom file lists)
        for url in self._cfg.dump_urls:
            async for doc in self._stream_dump(url): yield doc

    async def _stream_dump(self, url: str) -> AsyncGenerator[Document, None]:
        resume = self._resume_offset(url)
        buf = b""; accepted = seen = byte_pos = 0
        async for chunk in self._stream_url(url, resume_offset=resume):
            byte_pos += len(chunk); buf += chunk
            while b"\n" in buf:
                line, buf = buf.split(b"\n", 1)
                line = line.strip()
                if not line: continue
                try:
                    record = json.loads(line)
                except Exception: continue
                seen += 1
                doc = self._parse_record(record, url)
                if doc:
                    accepted += 1
                    self._update_checkpoint(url, byte_pos, accepted, seen)
                    yield doc
        self._log.info("GitHub dump done: %s seen=%d accepted=%d", url, seen, accepted)

    def _parse_record(self, record: dict, source_url: str) -> Optional[Document]:
        fname   = record.get("path", record.get("filename", ""))
        content = record.get("content", record.get("code", ""))
        if not content or not fname: return None
        ext = "." + fname.rsplit(".",1)[-1].lower() if "." in fname else ""
        if self._exts and ext not in self._exts: return None
        if len(content) < self._min_ch: return None
        if len(content) > self._max_ch: content = content[:self._max_ch]
        lang = ext.lstrip(".")
        text = f"# {fname}\n\n{content}"
        doc = self._make_doc(
            text=text, title=fname,
            url=record.get("url", ""),
            metadata={"filename": fname, "language": lang,
                      "repo": record.get("repo_name",""), "dump_url": source_url},
        )
        doc.stage = ProcessingStage.EXTRACTED
        doc.category = lang
        return doc
