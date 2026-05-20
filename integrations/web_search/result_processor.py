"""integrations/web_search/result_processor.py"""
from __future__ import annotations
import asyncio
import logging
import re
from typing import List
from urllib.parse import urlparse

from integrations.web_search.adapters.base import SearchResult

logger = logging.getLogger(__name__)


async def _fetch_text(url: str, timeout: int = 5) -> str:
    try:
        import aiohttp
        async with aiohttp.ClientSession() as s:
            async with s.get(url, timeout=aiohttp.ClientTimeout(total=timeout)) as r:
                if "text/html" not in (r.content_type or ""):
                    return ""
                html = await r.text(errors="replace")
                text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL)
                text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL)
                text = re.sub(r"<[^>]+>", " ", text)
                return re.sub(r"\s+", " ", text).strip()[:3000]
    except Exception:
        return ""


async def enrich_and_deduplicate(
    results: List[SearchResult],
    fetch_full_text: bool = True,
    max_results: int = 5,
) -> List[SearchResult]:
    seen = set()
    deduped = []
    for r in results:
        domain = urlparse(r.url).netloc
        if domain not in seen:
            seen.add(domain)
            r.source_domain = domain
            deduped.append(r)
        if len(deduped) >= max_results:
            break

    if fetch_full_text:
        try:
            texts = await asyncio.gather(*[_fetch_text(r.url) for r in deduped])
            for res, text in zip(deduped, texts):
                if text and not res.full_text:
                    res.full_text = text
        except Exception as exc:
            logger.debug("Text fetch failed: %s", exc)

    return deduped
