"""
integrations/web_search/grounded_pipeline.py
=============================================
Coordinates intent detection → query rewriting → multi-provider search
→ result enrichment → context injection for grounded inference.
"""
from __future__ import annotations

import asyncio
import logging
from typing import List, Optional, Tuple

from integrations.web_search.adapters.base import SearchResult
from integrations.web_search.intent_detector import needs_search
from integrations.web_search.result_processor import enrich_and_deduplicate
from integrations.web_search.context_builder import build_grounded_prompt, extract_citations

logger = logging.getLogger(__name__)


async def search_and_build_context(
    query:            str,
    provider:         str  = "brave",
    num_results:      int  = 5,
    fetch_full_text:  bool = True,
) -> Tuple[List[dict], str]:
    """
    Run a full search + context-build cycle.

    Returns
    -------
    (sources_as_dicts, system_prompt_string)
    """
    should_search, reason = needs_search(query)
    if not should_search:
        logger.debug("No search needed for query (reason=%s)", reason)
        return [], "You are DGB, a helpful AI assistant."

    adapter = _get_adapter(provider)
    if adapter is None:
        logger.warning("Search adapter '%s' unavailable — no keys configured", provider)
        return [], "You are DGB, a helpful AI assistant."

    try:
        raw_results: List[SearchResult] = await adapter.search(query, num_results=num_results)
        sources = await enrich_and_deduplicate(
            raw_results, fetch_full_text=fetch_full_text, max_results=num_results
        )
        system_prompt = build_grounded_prompt(query, sources)
        sources_dicts = [
            {
                "index":   i + 1,
                "title":   s.title,
                "url":     s.url,
                "domain":  s.source_domain,
                "snippet": s.snippet[:200],
            }
            for i, s in enumerate(sources)
        ]
        logger.info("Search grounding: %d sources retrieved for '%s'", len(sources), query[:60])
        return sources_dicts, system_prompt

    except Exception as exc:
        logger.warning("Grounded search failed: %s", exc)
        return [], "You are DGB, a helpful AI assistant."


def _get_adapter(provider: str):
    """Instantiate the requested search adapter from env-configured keys."""
    import os
    if provider == "brave":
        key = os.environ.get("BRAVE_API_KEY", "")
        if key:
            from integrations.web_search.adapters.brave_adapter import BraveSearchAdapter
            return BraveSearchAdapter(api_key=key)
    elif provider == "tavily":
        key = os.environ.get("TAVILY_KEY", "")
        if key:
            from integrations.web_search.adapters.tavily_adapter import TavilyAdapter
            return TavilyAdapter(api_key=key)
    elif provider == "serpapi":
        key = os.environ.get("SERPAPI_KEY", "")
        if key:
            from integrations.web_search.adapters.serpapi_adapter import SerpAPIAdapter
            return SerpAPIAdapter(api_key=key)
    return None
