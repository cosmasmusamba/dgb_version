"""
tokenizer/dgb_tokenizer.py
===========================
Public facade that assembles all tokenizer components into a single
easy-to-use object.

FIX B1 (v3.0.0): _finalize() now reads max_seq_len from
    get_config().transformer.max_seq_len (512)
    instead of tok_cfg.vocab_size (8000).
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import List, Optional

from configs.loader import get_config
from modules.utils.error_handler import TokenizerNotTrainedError
from modules.utils.file_handler import read_json, ensure_dir
from tokenizer.core.pre_tokenizer    import PreTokenizer
from tokenizer.core.bpe_processor    import BPEProcessor
from tokenizer.core.vocabulary_manager import VocabularyManager
from tokenizer.core.token_manager    import TokenManager, EncodingResult, BatchEncoding
from tokenizer.core.trainer          import TokenizerTrainer
from tokenizer.core.trie             import Trie

logger = logging.getLogger(__name__)


class DGBTokenizer:
    """
    Unified tokenizer for the DGB platform.

    Assembles PreTokenizer + BPEProcessor + VocabularyManager +
    TokenManager + Trie into one coherent object.
    """

    def __init__(self) -> None:
        cfg           = get_config()
        tok_cfg       = cfg.tokenizer
        self._tok_cfg = tok_cfg
        self._cfg     = cfg

        self._pre_tok = PreTokenizer(lowercase=False, add_eow=True)
        self._bpe     = BPEProcessor(
            num_merges=tok_cfg.num_merges,
            min_freq=tok_cfg.min_freq,
        )
        self._vocab   = VocabularyManager(
            vocab_size=tok_cfg.vocab_size,
            min_freq=tok_cfg.min_freq,
        )
        self._token_mgr: Optional[TokenManager] = None
        self._trie:      Optional[Trie]          = None
        self._trained    = False

    def train(self, cleaned_dir: Path, save_dir: Path) -> "DGBTokenizer":
        trainer = TokenizerTrainer()
        trainer.run(cleaned_dir=cleaned_dir, save_dir=save_dir)
        self._bpe   = trainer.bpe
        self._vocab = trainer.vocab
        self._finalize()
        return self

    def save(self, save_dir: Path) -> None:
        ensure_dir(save_dir)
        self._vocab.save(save_dir)
        from modules.utils.safe_writer import atomic_write_json
        atomic_write_json(save_dir / "bpe_merges.json", self._bpe.get_state())
        logger.info("DGBTokenizer saved → %s", save_dir)

    @classmethod
    def from_pretrained(cls, save_dir: Path) -> "DGBTokenizer":
        tok      = cls()
        save_dir = Path(save_dir)
        vocab_candidates  = sorted(save_dir.glob("*vocabulary.json"))
        merges_candidates = sorted(save_dir.glob("*bpe_merges.json"))
        if not vocab_candidates:
            raise FileNotFoundError(f"No vocabulary.json found in {save_dir}")
        if not merges_candidates:
            raise FileNotFoundError(f"No bpe_merges.json found in {save_dir}")
        tok._vocab.load(save_dir)
        tok._bpe.load_state(read_json(merges_candidates[-1]))
        tok._finalize()
        logger.info(
            "DGBTokenizer loaded from %s — vocab_size=%d  merges=%d",
            save_dir, tok.vocab_size, len(tok._bpe.merges),
        )
        return tok

    def encode(
        self,
        text: str,
        *,
        add_special_tokens: bool = True,
        max_length: Optional[int] = None,
        padding: bool = False,
        truncation: bool = True,
    ) -> List[int]:
        if not self._trained:
            raise TokenizerNotTrainedError()
        result = self._token_mgr.encode(
            text,
            add_special_tokens=add_special_tokens,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
        )
        return result.input_ids

    def encode_plus(self, text: str, **kwargs) -> EncodingResult:
        if not self._trained:
            raise TokenizerNotTrainedError()
        return self._token_mgr.encode(text, **kwargs)

    def encode_batch(self, texts: List[str], **kwargs) -> BatchEncoding:
        if not self._trained:
            raise TokenizerNotTrainedError()
        return self._token_mgr.encode_batch(texts, **kwargs)

    def decode(self, ids: List[int], *, skip_special_tokens: bool = True) -> str:
        if not self._trained:
            raise TokenizerNotTrainedError()
        return self._token_mgr.decode(ids, skip_special_tokens=skip_special_tokens)

    def tokenize(self, text: str) -> List[str]:
        if not self._trained:
            raise TokenizerNotTrainedError()
        pre = self._pre_tok.tokenize(text)
        return self._bpe.encode_pre_tokens(pre)

    @property
    def is_trained(self) -> bool:
        return self._trained

    @property
    def vocab_size(self) -> int:
        return self._vocab.size

    @property
    def vocabulary(self) -> VocabularyManager:
        return self._vocab

    def _finalize(self) -> None:
        """Wire up TokenManager and Trie. FIX B1: uses max_seq_len from transformer config."""
        # FIX B1: was self._tok_cfg.vocab_size (8000) — should be transformer.max_seq_len (512)
        max_seq_len = self._cfg.transformer.max_seq_len
        self._token_mgr = TokenManager(
            vocab=self._vocab,
            bpe=self._bpe,
            pre_tokenizer=self._pre_tok,
            max_seq_len=max_seq_len,    # FIX B1
        )
        self._trie = Trie()
        self._trie.build(self._vocab.token2id)
        self._trained = True
        logger.debug("DGBTokenizer finalised: max_seq_len=%d  vocab_size=%d",
                     max_seq_len, self._vocab.size)
