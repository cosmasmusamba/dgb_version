"""integrations/web_search/adapters/brave_adapter.py"""
from __future__ import annotations
import aiohttp
from typing import List
from integrations.web_search.adapters.base import BaseSearchAdapter, SearchResult

class BraveSearchAdapter(BaseSearchAdapter):
    BASE_URL = "https://api.search.brave.com/res/v1/web/search"
    def __init__(self, api_key: str) -> None:
        self._key = api_key
    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]:
        headers = {"Accept": "application/json", "X-Subscription-Token": self._key}
        async with aiohttp.ClientSession() as s:
            async with s.get(self.BASE_URL, headers=headers, params={"q": query, "count": num_results}) as r:
                data = await r.json()
        return [
            SearchResult(title=i.get("title",""), url=i.get("url",""),
                snippet=i.get("description",""), source_domain=i.get("profile",{}).get("name",""),
                published_date=i.get("page_age"))
            for i in data.get("web",{}).get("results",[])[:num_results]
        ]
