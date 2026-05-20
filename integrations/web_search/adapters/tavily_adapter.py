"""integrations/web_search/adapters/tavily_adapter.py"""
from __future__ import annotations
import aiohttp
from typing import List
from integrations.web_search.adapters.base import BaseSearchAdapter, SearchResult

class TavilyAdapter(BaseSearchAdapter):
    BASE_URL = "https://api.tavily.com/search"
    def __init__(self, api_key: str) -> None:
        self._key = api_key
    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        async with aiohttp.ClientSession() as s:
            async with s.post(self.BASE_URL, json={"api_key": self._key, "query": query,
                    "max_results": num_results, "include_raw_content": True}) as r:
                data = await r.json()
        return [
            SearchResult(title=i.get("title",""), url=i.get("url",""),
                snippet=i.get("content","")[:400], full_text=i.get("raw_content","")[:2000],
                source_domain=i.get("url","").split("/")[2] if i.get("url","").count("/") >= 2 else "")
            for i in data.get("results",[])[:num_results]
        ]
