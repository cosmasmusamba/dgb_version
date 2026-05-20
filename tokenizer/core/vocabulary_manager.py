"""tokenizer/core/vocabulary_manager.py"""
from __future__ import annotations
import logging
from pathlib import Path
from typing import Dict, Optional
from modules.utils.safe_writer import atomic_write_json
from modules.utils.file_handler import read_json
logger = logging.getLogger(__name__)

class VocabularyManager:
    SPECIAL = ["<PAD>","<UNK>","<BOS>","<EOS>","<MASK>","<SEP>"]
    def __init__(self, vocab_size=8000, min_freq=2):
        self._max  = vocab_size
        self._min  = min_freq
        self.token2id: Dict[str,int] = {}
        self.id2token: Dict[int,str] = {}

    def build_from_bpe(self, bpe, word_freq) -> None:
        from collections import Counter
        token_freq: Counter = Counter()
        for word, freq in word_freq.items():
            chars = list(word)
            for a, b in bpe.merges:
                i = 0
                while i < len(chars)-1:
                    if chars[i]==a and chars[i+1]==b: chars[i:i+2]=[a+b]
                    else: i+=1
            for t in chars: token_freq[t] += freq

        self.token2id = {}
        for i, tok in enumerate(self.SPECIAL):
            self.token2id[tok] = i

        for tok, freq in token_freq.most_common():
            if len(self.token2id) >= self._max: break
            if freq < self._min: break
            if tok not in self.token2id:
                self.token2id[tok] = len(self.token2id)

        self.id2token = {v:k for k,v in self.token2id.items()}
        logger.info("Vocabulary built: %d tokens", len(self.token2id))

    @property
    def size(self) -> int:
        return len(self.token2id)

    def to_dict(self) -> dict:
        return {"token2id": self.token2id, "id2token": {str(k):v for k,v in self.id2token.items()}}

    def save(self, directory: Path) -> None:
        atomic_write_json(Path(directory)/"vocabulary.json", self.to_dict())

    def load(self, directory: Path) -> None:
        candidates = sorted(Path(directory).glob("*vocabulary.json"))
        if not candidates: raise FileNotFoundError(f"No vocabulary.json in {directory}")
        d = read_json(candidates[-1])
        self.token2id = d["token2id"]
        self.id2token = {int(k):v for k,v in d["id2token"].items()}
        logger.info("Vocabulary loaded: %d tokens", self.size)
