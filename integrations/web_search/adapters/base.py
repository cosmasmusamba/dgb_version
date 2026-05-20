"""
integrations/web_search/adapters/base.py
"""
from __future__ import annotations
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import List, Optional

@dataclass
class SearchResult:
    title:          str
    url:            str
    snippet:        str
    full_text:      Optional[str] = None
    published_date: Optional[str] = None
    source_domain:  str = ""

class BaseSearchAdapter(ABC):
    @abstractmethod
    async def search(self, query: str, num_results: int = 5) -> List[SearchResult]: ...
