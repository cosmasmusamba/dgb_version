"""tokenizer/core/bpe_processor.py"""
from __future__ import annotations
import logging
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
logger = logging.getLogger(__name__)

class BPEProcessor:
    EOW = "</w>"
    def __init__(self, num_merges: int = 7000, min_freq: int = 2):
        self._n_merges = num_merges
        self._min_freq = min_freq
        self.merges: List[Tuple[str,str]] = []
        self._merge_rank: Dict[Tuple[str,str], int] = {}

    def train(self, word_freq: Counter) -> None:
        vocab: Dict[Tuple[str,...], int] = {}
        for word, freq in word_freq.items():
            if freq < self._min_freq: continue
            chars = tuple(list(word[:-len(self.EOW)]) + [self.EOW]) if word.endswith(self.EOW) else tuple(word)
            vocab[chars] = freq

        for i in range(self._n_merges):
            pairs = self._get_pairs(vocab)
            if not pairs: break
            best = max(pairs, key=pairs.get)
            if pairs[best] < self._min_freq: break
            vocab = self._merge(vocab, best)
            self.merges.append(best)
            self._merge_rank[best] = i
            if i % 1000 == 0:
                logger.debug("BPE merge %d/%d: %s + %s", i, self._n_merges, *best)
        logger.info("BPE training done: %d merges", len(self.merges))

    def _get_pairs(self, vocab):
        pairs = Counter()
        for word, freq in vocab.items():
            for a, b in zip(word, word[1:]):
                pairs[(a, b)] += freq
        return pairs

    def _merge(self, vocab, pair):
        new = {}
        a, b = pair
        bigram = a + b
        for word, freq in vocab.items():
            out, i = [], 0
            while i < len(word):
                if i < len(word)-1 and word[i] == a and word[i+1] == b:
                    out.append(bigram); i += 2
                else:
                    out.append(word[i]); i += 1
            new[tuple(out)] = freq
        return new

    def encode_pre_tokens(self, words: List[str]) -> List[str]:
        result = []
        for word in words:
            chars = list(word)
            if not chars: continue
            for a, b in self.merges:
                i = 0
                while i < len(chars)-1:
                    if chars[i] == a and chars[i+1] == b:
                        chars[i:i+2] = [a+b]; 
                    else: i += 1
            result.extend(chars)
        return result

    def get_state(self) -> dict:
        return {"merges": [[a, b] for a, b in self.merges]}

    def load_state(self, state: dict) -> None:
        self.merges = [tuple(m) for m in state.get("merges", [])]
        self._merge_rank = {tuple(m): i for i, m in enumerate(self.merges)}
