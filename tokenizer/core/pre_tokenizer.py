"""tokenizer/core/pre_tokenizer.py"""
from __future__ import annotations
import re
from typing import List

_SPLIT_RE = re.compile(r"""'s|'t|'re|'ve|'m|'ll|'d| ?\w+| ?[^\s\w]+|\s+(?!\S)|\s+""")

class PreTokenizer:
    EOW = "</w>"
    def __init__(self, lowercase: bool = False, add_eow: bool = True):
        self._lower   = lowercase
        self._add_eow = add_eow

    def tokenize(self, text: str) -> List[str]:
        if self._lower: text = text.lower()
        tokens = []
        for m in _SPLIT_RE.finditer(text):
            w = m.group(0)
            if w.strip():
                chars = list(w)
                if self._add_eow and chars:
                    chars[-1] += self.EOW
                tokens.extend(chars)
        return tokens

    def split_words(self, text: str) -> List[str]:
        if self._lower: text = text.lower()
        return [m.group(0).strip() for m in _SPLIT_RE.finditer(text) if m.group(0).strip()]
