"""
data_pipeline/extractors/stackexchange_extractor.py
=====================================================
StackExchange dump extractor.

Streams from archive.org StackExchange data dumps (7z XML archives).
Extracts high-quality Q&A pairs by combining question + accepted answer
(or top-voted answer) into a single document.

Quality filters:
  - Minimum question score (default ≥ 1)
  - Minimum answer score (default ≥ 2)
  - Accepted answer preference
  - HTML stripping from post bodies
  - Code block handling (kept as-is or stripped, configurable)

Dump index: https://archive.org/details/stackexchange
Individual: https://archive.org/download/stackexchange/{site}.7z
"""
from __future__ import annotations

import html
import io
import logging
import re
import xml.etree.ElementTree as ET
from typing import AsyncGenerator, Dict, List, Optional, Tuple

from data_pipeline.core.document import Document, ContentDomain, LicenseType, ProcessingStage
from data_pipeline.extractors.base_extractor import BaseExtractor, ExtractorConfig

logger = logging.getLogger(__name__)

_DUMP_BASE = "https://archive.org/download/stackexchange"

# HTML cleaning for SE post bodies
_RE_CODE     = re.compile(r"<code>(.*?)</code>", re.DOTALL)
_RE_PRE      = re.compile(r"<pre>(.*?)</pre>",  re.DOTALL)
_RE_TAG      = re.compile(r"<[^>]+>")
_RE_MULTI_NL = re.compile(r"\n{3,}")
_RE_MULTI_SP = re.compile(r" {2,}")


def _clean_se_body(body: str, keep_code: bool = True) -> str:
    """Clean HTML from a StackExchange post body."""
    body = html.unescape(body)
    if keep_code:
        body = _RE_CODE.sub(lambda m: "\n```\n" + m.group(1).strip() + "\n```\n", body)
        body = _RE_PRE.sub(lambda m: "\n```\n" + m.group(1).strip() + "\n```\n", body)
    body = _RE_TAG.sub(" ", body)
    body = _RE_MULTI_NL.sub("\n\n", body)
    body = _RE_MULTI_SP.sub(" ", body)
    return body.strip()


class StackExchangeExtractor(BaseExtractor):
    """
    Streams Q&A documents from StackExchange XML dumps.

    Each document is a formatted Q&A pair:
        Question: <title>\n<body>\n\nAnswer:\n<body>
    """

    def __init__(self, cfg: ExtractorConfig, checkpoint_mgr=None, quota_mgr=None) -> None:
        super().__init__(cfg, checkpoint_mgr, quota_mgr)
        self._min_q_score  = cfg.extra.get("min_question_score", 1)
        self._min_a_score  = cfg.extra.get("min_answer_score", 2)
        self._keep_code    = cfg.extra.get("keep_code_blocks", True)
        self._sites        = cfg.extra.get("sites", ["stackoverflow", "superuser", "serverfault",
                                                      "askubuntu", "mathoverflow.net",
                                                      "physics.stackexchange.com",
                                                      "stats.stackexchange.com"])

    @classmethod
    def build(cls, cfg_dict: dict, checkpoint_mgr=None, quota_mgr=None) -> "StackExchangeExtractor":
        cfg = ExtractorConfig.from_dict(cfg_dict, source_name="stackexchange")
        cfg.domain  = ContentDomain.QA
        cfg.license = LicenseType.CC_BY_SA
        return cls(cfg, checkpoint_mgr, quota_mgr)

    async def stream(self) -> AsyncGenerator[Document, None]:
        for site in self._sites:
            url = f"{_DUMP_BASE}/{site}.7z"
            if self._cfg.dump_urls:
                url = self._cfg.dump_urls[0]
            async for doc in self._stream_site(site, url):
                yield doc

    async def _stream_site(self, site: str, url: str) -> AsyncGenerator[Document, None]:
        """Stream Posts.xml from a single StackExchange site dump."""
        resume   = self._resume_offset(url)
        self._log.info("StackExchange: streaming %s (resume=%d)", site, resume)

        # Buffer for XML processing — SE dumps are large XML files inside 7z
        # We process line-by-line to avoid loading full XML into memory.
        posts: Dict[str, dict] = {}   # post_id → post data
        answers: Dict[str, List[dict]] = {}   # question_id → [answer, ...]
        accepted = seen = 0

        try:
            buf = b""
            async for chunk in self._stream_url(url, resume_offset=resume):
                buf += chunk
                # Process complete <row ... /> elements
                while True:
                    start = buf.find(b"<row ")
                    end   = buf.find(b"/>", start) + 2 if start >= 0 else -1
                    if start < 0 or end < 2:
                        break
                    row_bytes = buf[start:end]
                    buf       = buf[end:]

                    post = self._parse_row(row_bytes.decode("utf-8", errors="replace"))
                    if not post:
                        continue
                    seen += 1

                    post_type = post.get("PostTypeId", "0")
                    post_id   = post.get("Id", "")
                    parent_id = post.get("ParentId", "")

                    if post_type == "1":   # Question
                        posts[post_id] = post
                    elif post_type == "2":   # Answer
                        answers.setdefault(parent_id, []).append(post)

                    # When we have a question + its accepted/top answer, emit
                    doc = self._try_emit(post_id if post_type=="1" else parent_id,
                                         posts, answers, site, url)
                    if doc:
                        accepted += 1
                        self._update_checkpoint(url, self._bytes_read, accepted, seen)
                        yield doc

        except Exception as exc:
            self._log.error("StackExchange %s error: %s", site, exc)

        self._log.info(
            "StackExchange %s done: seen=%d accepted=%d", site, seen, accepted
        )

    def _parse_row(self, row_xml: str) -> Optional[dict]:
        """Parse a <row ... /> element into a dict."""
        try:
            el = ET.fromstring(row_xml + "</row>" if not row_xml.endswith("/>") else row_xml)
            return dict(el.attrib)
        except ET.ParseError:
            return None

    def _try_emit(
        self,
        q_id:    str,
        posts:   dict,
        answers: dict,
        site:    str,
        url:     str,
    ) -> Optional[Document]:
        """Emit a Q&A document when question + answer are both available."""
        q = posts.get(q_id)
        if not q:
            return None
        q_score = int(q.get("Score", 0))
        if q_score < self._min_q_score:
            return None

        # Prefer accepted answer, then highest-scored
        ans_list = answers.get(q_id, [])
        if not ans_list:
            return None

        accepted_id = q.get("AcceptedAnswerId", "")
        best_ans    = None
        for a in ans_list:
            if a.get("Id") == accepted_id:
                best_ans = a
                break
        if best_ans is None:
            best_ans = max(ans_list, key=lambda a: int(a.get("Score", 0)))

        a_score = int(best_ans.get("Score", 0))
        if a_score < self._min_a_score:
            return None

        q_title = html.unescape(q.get("Title", ""))
        q_body  = _clean_se_body(q.get("Body", ""), self._keep_code)
        a_body  = _clean_se_body(best_ans.get("Body", ""), self._keep_code)

        text = (
            f"Question: {q_title}\n\n"
            f"{q_body}\n\n"
            f"Answer:\n\n{a_body}"
        )
        if len(text) < 100:
            return None

        tags = q.get("Tags", "").replace("><", ",").strip("<>")
        doc  = self._make_doc(
            text=text,
            title=q_title,
            url=f"https://{site}.com/questions/{q_id}",
            metadata={
                "site":          site,
                "q_id":          q_id,
                "a_id":          best_ans.get("Id", ""),
                "q_score":       q_score,
                "a_score":       a_score,
                "accepted":      best_ans.get("Id") == accepted_id,
                "tags":          tags,
                "dump_url":      url,
            },
            source_timestamp=q.get("CreationDate"),
        )
        doc.stage    = ProcessingStage.EXTRACTED
        doc.category = tags.split(",")[0].strip() if tags else ""
        return doc