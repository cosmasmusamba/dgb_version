"""tokenizer/core/token_manager.py"""
from __future__ import annotations
import logging
from dataclasses import dataclass, field
from typing import List, Optional
logger = logging.getLogger(__name__)

@dataclass
class EncodingResult:
    input_ids:      List[int]
    attention_mask: List[int]
    n_tokens:       int

@dataclass
class BatchEncoding:
    input_ids:      List[List[int]]
    attention_mask: List[List[int]]
    n_tokens:       List[int]

class TokenManager:
    def __init__(self, vocab, bpe, pre_tokenizer, max_seq_len=512):
        self._vocab   = vocab
        self._bpe     = bpe
        self._pre     = pre_tokenizer
        self._max     = max_seq_len
        self._pad_id  = vocab.token2id.get("<PAD>", 0)
        self._bos_id  = vocab.token2id.get("<BOS>", 2)
        self._eos_id  = vocab.token2id.get("<EOS>", 3)
        self._unk_id  = vocab.token2id.get("<UNK>", 1)

    def encode(self, text: str, *, add_special_tokens=True,
               max_length=None, padding=False, truncation=True) -> EncodingResult:
        words  = self._pre.split_words(text)
        tokens = self._bpe.encode_pre_tokens(words)
        ids    = [self._vocab.token2id.get(t, self._unk_id) for t in tokens]
        if add_special_tokens:
            ids = [self._bos_id] + ids + [self._eos_id]
        limit = max_length or self._max
        if truncation and len(ids) > limit:
            ids = ids[:limit-1] + [self._eos_id]
        mask = [1] * len(ids)
        if padding and len(ids) < limit:
            pad = limit - len(ids)
            ids  += [self._pad_id] * pad
            mask += [0] * pad
        return EncodingResult(input_ids=ids, attention_mask=mask, n_tokens=len(ids))

    def encode_batch(self, texts: List[str], **kwargs) -> BatchEncoding:
        results = [self.encode(t, **kwargs) for t in texts]
        return BatchEncoding(
            input_ids=[r.input_ids for r in results],
            attention_mask=[r.attention_mask for r in results],
            n_tokens=[r.n_tokens for r in results],
        )

    def decode(self, ids: List[int], skip_special_tokens=True) -> str:
        special = set(self._vocab.token2id.get(t,999) for t in ["<PAD>","<BOS>","<EOS>","<MASK>","<SEP>"])
        tokens = []
        for i in ids:
            if skip_special_tokens and i in special: continue
            tokens.append(self._vocab.id2token.get(i, "<UNK>"))
        text = "".join(tokens).replace("</w>", " ").strip()
        return text
