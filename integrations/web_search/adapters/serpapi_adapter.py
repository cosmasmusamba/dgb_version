"""integrations/web_search/adapters/serpapi_adapter.py"""
from __future__ import annotations
import aiohttp
from typing import List
from integrations.web_search.adapters.base import BaseSearchAdapter, SearchResult

class SerpAPIAdapter(BaseSearchAdapter):
    BASE_URL = "https://serpapi.com/search"
    def __init__(self, api_key: str) -> None:
        self._key = api_key
    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        async with aiohttp.ClientSession() as s:
            async with s.get(self.BASE_URL, params={"q": query, "api_key": self._key,
                    "num": num_results, "engine": "google"}) as r:
                data = await r.json()
        return [
            SearchResult(title=i.get("title",""), url=i.get("link",""),
                snippet=i.get("snippet",""), source_domain=i.get("displayed_link",""),
                published_date=i.get("date"))
            for i in data.get("organic_results",[])[:num_results]
        ]
