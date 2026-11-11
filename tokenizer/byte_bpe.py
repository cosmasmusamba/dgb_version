"""
tokenizer/byte_bpe.py  (batch3 / roadmap T1.1)
================================================
Byte-level BPE tokenizer.

Why: the legacy tokenizer (a) dropped every space when decoding, because it stored no whitespace
information, and (b) mapped any non-ASCII character or emoji to <UNK>. This tokenizer works on the
UTF-8 *bytes* of the text, so
    * decode(encode(text)) == text for ANY string (exact round trip, spaces/newlines/emoji included)
    * <UNK> is never produced (kept reserved at id 1 so ids of the other specials do not move)

Layout of ids (identical special ids to the legacy tokenizer, so pad_idx=0 etc. keep working):
    0 <PAD>  1 <UNK>  2 <BOS>  3 <EOS>  4 <MASK>  5 <SEP>   6..261 raw bytes 0..255   262.. merges

Pure standard library. Same public surface as DGBTokenizer where the rest of DGB uses it:
encode / encode_plus / encode_batch / decode / vocab_size / is_trained / from_pretrained / save.
"""
from __future__ import annotations

import heapq
import json
import logging
import os
import re
from collections import Counter, defaultdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

from tokenizer.core.token_manager import BatchEncoding, EncodingResult

logger = logging.getLogger(__name__)

SPECIALS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<MASK>", "<SEP>"]
N_SPECIAL = len(SPECIALS)
PAD_ID, UNK_ID, BOS_ID, EOS_ID = 0, 1, 2, 3
MARKER_FILE = "byte_bpe.json"          # its presence in a tokenizer dir selects this tokenizer
FORMAT_VERSION = 1

# Pre-tokenizer: contractions | letters | single digits | punctuation/underscore | whitespace.
# A leading space stays attached to the word that follows it (GPT-2 style), which is what lets
# decode restore spaces. Every character matches exactly one alternative, so nothing is dropped.
_SPLIT_RE = re.compile(
    r"'(?:s|t|re|ve|m|ll|d)| ?[^\W\d_]+| ?\d| ?(?:[^\s\w]|_)+|\s+(?!\S)|\s+"
)


def pre_split(text: str) -> List[str]:
    return _SPLIT_RE.findall(text)


def _pair_counts_of(word: List[int]) -> Counter:
    c: Counter = Counter()
    for a, b in zip(word, word[1:]):
        c[(a, b)] += 1
    return c


class ByteBPETokenizer:
    """Byte-level BPE. Internal symbol ids: 0..255 = bytes, 256+i = i-th merge."""

    def __init__(self, merges: Optional[List[Tuple[int, int]]] = None, max_seq_len: int = 512):
        self.max_seq_len = max_seq_len
        self.merges: List[Tuple[int, int]] = []
        self._rank: Dict[Tuple[int, int], int] = {}
        self._tok_bytes: List[bytes] = [bytes([i]) for i in range(256)]
        self._cache: Dict[str, Tuple[int, ...]] = {}
        if merges:
            self._set_merges(merges)

    # ------------------------------------------------------------------ state
    def _set_merges(self, merges: Iterable[Tuple[int, int]]) -> None:
        self.merges = [tuple(m) for m in merges]
        self._rank = {m: i for i, m in enumerate(self.merges)}
        self._tok_bytes = [bytes([i]) for i in range(256)]
        for a, b in self.merges:
            self._tok_bytes.append(self._tok_bytes[a] + self._tok_bytes[b])
        self._cache.clear()

    @property
    def is_trained(self) -> bool:
        return True

    @property
    def vocab_size(self) -> int:
        return N_SPECIAL + len(self._tok_bytes)

    pad_id, unk_id, bos_id, eos_id = PAD_ID, UNK_ID, BOS_ID, EOS_ID

    # --------------------------------------------------------------- training
    @classmethod
    def train(cls, texts: Iterable[str], vocab_size: int = 8000, min_freq: int = 2,
              max_seq_len: int = 512) -> "ByteBPETokenizer":
        """Learn merges from an iterable of text lines (streamed; only word counts are kept)."""
        word_freq: Counter = Counter()
        for t in texts:
            word_freq.update(pre_split(t))
        return cls.train_from_word_freq(word_freq, vocab_size, min_freq, max_seq_len)

    @classmethod
    def train_from_word_freq(cls, word_freq: Counter, vocab_size: int = 8000, min_freq: int = 2,
                             max_seq_len: int = 512) -> "ByteBPETokenizer":
        n_merges = vocab_size - N_SPECIAL - 256
        if n_merges < 0:
            raise ValueError(f"vocab_size must be at least {N_SPECIAL + 256} (specials + 256 bytes)")
        words: List[List[int]] = []
        freqs: List[int] = []
        for w, f in sorted(word_freq.items()):          # sorted -> deterministic training
            words.append(list(w.encode("utf-8")))
            freqs.append(f)

        pairs: Counter = Counter()
        where: Dict[Tuple[int, int], set] = defaultdict(set)
        for i, w in enumerate(words):
            for p, c in _pair_counts_of(w).items():
                pairs[p] += c * freqs[i]
                where[p].add(i)
        heap = [(-c, p) for p, c in pairs.items()]
        heapq.heapify(heap)

        merges: List[Tuple[int, int]] = []
        while len(merges) < n_merges and heap:
            negc, best = heapq.heappop(heap)
            if pairs.get(best, 0) != -negc:             # stale heap entry
                continue
            if -negc < max(min_freq, 1):
                break
            new_id = 256 + len(merges)
            merges.append(best)
            touched = set()
            for i in list(where[best]):
                w, f = words[i], freqs[i]
                for p, c in _pair_counts_of(w).items():  # remove this word's contribution
                    pairs[p] -= c * f
                    where[p].discard(i)
                    touched.add(p)
                out, j = [], 0
                while j < len(w):
                    if j < len(w) - 1 and w[j] == best[0] and w[j + 1] == best[1]:
                        out.append(new_id)
                        j += 2
                    else:
                        out.append(w[j])
                        j += 1
                words[i] = out
                for p, c in _pair_counts_of(out).items():  # add the merged word's contribution
                    pairs[p] += c * f
                    where[p].add(i)
                    touched.add(p)
            for p in touched:
                if pairs[p] > 0:
                    heapq.heappush(heap, (-pairs[p], p))
                else:
                    pairs.pop(p, None)
        logger.info("ByteBPE trained: %d merges, vocab_size=%d", len(merges), N_SPECIAL + 256 + len(merges))
        return cls(merges, max_seq_len=max_seq_len)

    # --------------------------------------------------------------- encoding
    def _encode_word(self, word: str) -> Tuple[int, ...]:
        hit = self._cache.get(word)
        if hit is not None:
            return hit
        ids = list(word.encode("utf-8"))
        rank = self._rank
        while len(ids) > 1:
            best, best_rank = None, None
            for pair in zip(ids, ids[1:]):
                r = rank.get(pair)
                if r is not None and (best_rank is None or r < best_rank):
                    best, best_rank = pair, r
            if best is None:
                break
            new_id, out, j = 256 + best_rank, [], 0
            while j < len(ids):
                if j < len(ids) - 1 and ids[j] == best[0] and ids[j + 1] == best[1]:
                    out.append(new_id)
                    j += 2
                else:
                    out.append(ids[j])
                    j += 1
            ids = out
        res = tuple(ids)
        if len(self._cache) < 200_000:
            self._cache[word] = res
        return res

    def encode_plus(self, text: str, *, add_special_tokens: bool = True,
                    max_length: Optional[int] = None, padding: bool = False,
                    truncation: bool = True) -> EncodingResult:
        ids: List[int] = []
        for w in pre_split(text):
            ids.extend(N_SPECIAL + t for t in self._encode_word(w))
        if add_special_tokens:
            ids = [BOS_ID] + ids + [EOS_ID]
        limit = max_length or self.max_seq_len
        if truncation and len(ids) > limit:
            ids = ids[:limit - 1] + [EOS_ID]
        mask = [1] * len(ids)
        if padding and len(ids) < limit:
            pad = limit - len(ids)
            ids += [PAD_ID] * pad
            mask += [0] * pad
        return EncodingResult(input_ids=ids, attention_mask=mask, n_tokens=len(ids))

    def encode(self, text: str, **kwargs) -> List[int]:
        return self.encode_plus(text, **kwargs).input_ids

    def encode_batch(self, texts: List[str], **kwargs) -> BatchEncoding:
        rs = [self.encode_plus(t, **kwargs) for t in texts]
        return BatchEncoding(input_ids=[r.input_ids for r in rs],
                             attention_mask=[r.attention_mask for r in rs],
                             n_tokens=[r.n_tokens for r in rs])

    # --------------------------------------------------------------- decoding
    def token_bytes(self, token_id: int) -> bytes:
        if token_id < N_SPECIAL or token_id - N_SPECIAL >= len(self._tok_bytes):
            return b""
        return self._tok_bytes[token_id - N_SPECIAL]

    def decode(self, ids: List[int], *, skip_special_tokens: bool = True) -> str:
        buf = bytearray()
        for i in ids:
            if i < N_SPECIAL:
                if not skip_special_tokens:
                    buf += SPECIALS[i].encode("utf-8")
            else:
                buf += self.token_bytes(i)
        return buf.decode("utf-8", errors="replace")

    def tokenize(self, text: str) -> List[str]:
        out = []
        for w in pre_split(text):
            for t in self._encode_word(w):
                out.append(self._tok_bytes[t].decode("utf-8", errors="replace"))
        return out

    # ------------------------------------------------------------ persistence
    def save(self, save_dir) -> None:
        d = Path(save_dir)
        d.mkdir(parents=True, exist_ok=True)
        payload = {"format": "dgb-byte-bpe", "version": FORMAT_VERSION, "specials": SPECIALS,
                   "vocab_size": self.vocab_size, "merges": [list(m) for m in self.merges]}
        tmp = d / (MARKER_FILE + ".tmp")
        tmp.write_text(json.dumps(payload, separators=(",", ":")), encoding="utf-8")
        os.replace(tmp, d / MARKER_FILE)
        logger.info("ByteBPE saved -> %s  (vocab_size=%d)", d / MARKER_FILE, self.vocab_size)

    @classmethod
    def from_pretrained(cls, save_dir, max_seq_len: Optional[int] = None) -> "ByteBPETokenizer":
        p = Path(save_dir) / MARKER_FILE
        if not p.exists():
            raise FileNotFoundError(f"{MARKER_FILE} not found in {save_dir}")
        data = json.loads(p.read_text(encoding="utf-8"))
        if data.get("format") != "dgb-byte-bpe" or data.get("version") != FORMAT_VERSION:
            raise ValueError(f"Unsupported tokenizer file: {p}")
        if max_seq_len is None:
            try:
                from configs.loader import get_config
                max_seq_len = get_config().transformer.max_seq_len
            except Exception:  # noqa: BLE001 - tokenizer must stay usable without the config
                max_seq_len = 512
        tok = cls([tuple(m) for m in data["merges"]], max_seq_len=max_seq_len)
        if tok.vocab_size != data["vocab_size"]:
            raise ValueError(f"Corrupt tokenizer file {p}: vocab_size mismatch")
        logger.info("ByteBPE loaded from %s  vocab_size=%d", save_dir, tok.vocab_size)
        return tok
