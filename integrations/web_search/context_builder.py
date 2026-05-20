"""
integrations/web_search/context_builder.py
============================================
Builds the grounded system prompt and extracts [N] citations from responses.
"""
from __future__ import annotations

import re
from typing import List

from integrations.web_search.adapters.base import SearchResult


def build_grounded_prompt(query: str, results: List[SearchResult], max_chars: int = 1200) -> str:
    block = ""
    for i, r in enumerate(results, 1):
        body = (r.full_text or r.snippet)[:max_chars]
        date = f" ({r.published_date})" if r.published_date else ""
        block += f"[{i}] {r.title}{date}\nURL: {r.url}\n{body}\n\n"

    return (
        "You are DGB, a research AI. Answer using the sources below.\n"
        "Rules:\n"
        "- Cite every factual claim with [N] using the source number.\n"
        "- If sources do not contain enough information, say so.\n"
        "- Do not fabricate facts not present in sources.\n\n"
        f"SOURCES:\n{block.strip()}\n"
    )


def extract_citations(response_text: str, results: List[SearchResult]) -> List[dict]:
    cited = set(int(n) for n in re.findall(r"\[(\d+)\]", response_text))
    return [
        {
            "index": idx,
            "title": results[idx - 1].title,
            "url":   results[idx - 1].url,
            "domain": results[idx - 1].source_domain,
            "snippet": results[idx - 1].snippet[:160],
        }
        for idx in sorted(cited)
        if 1 <= idx <= len(results)
    ]
