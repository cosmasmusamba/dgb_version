"""integrations/web_search/intent_detector.py"""
from __future__ import annotations
import re
from typing import Tuple

_LIVE = re.compile(
    r"\btoday\b|\bnow\b|\bcurrent(ly)?\b|\blatest\b|\brecent(ly)?\b|"
    r"\bright now\b|\bthis (week|month|year)\b|\b202[4-9]\b|"
    r"\bprice of\b|\bstock\b|\bweather\b|\bbreaking\b|\bnews\b|"
    r"\bupdate(s)?\b|\bwhat happened\b|\bwhen did\b|"
    r"\bwho (is|are) (the )?(current|new)\b",
    re.IGNORECASE,
)

def needs_search(query: str) -> Tuple[bool, str]:
    if _LIVE.search(query):
        return True, "live_signal"
    words = query.split()
    if len(words) <= 8 and any(
        query.lower().startswith(p)
        for p in ("who is", "what is the", "where is", "when is", "how much is")
    ):
        return True, "short_factual"
    return False, "static_knowledge"
