"""tokenizer/core/trie.py — prefix trie for fast token lookup."""
from __future__ import annotations
from typing import Dict, Optional

class TrieNode:
    __slots__ = ["children", "token_id"]
    def __init__(self):
        self.children: Dict[str, "TrieNode"] = {}
        self.token_id: Optional[int] = None

class Trie:
    def __init__(self): self.root = TrieNode()

    def build(self, token2id: dict) -> None:
        for token, tid in token2id.items():
            node = self.root
            for ch in token:
                node = node.children.setdefault(ch, TrieNode())
            node.token_id = tid

    def longest_prefix(self, text: str, start: int):
        node, last_id, last_end = self.root, None, start
        for i in range(start, len(text)):
            ch = text[i]
            if ch not in node.children: break
            node = node.children[ch]
            if node.token_id is not None:
                last_id, last_end = node.token_id, i+1
        return last_id, last_end
