"""tokenizer/core/trainer.py — BPE tokenizer trainer."""
from __future__ import annotations
import logging, time
from pathlib import Path
from collections import Counter, defaultdict
from typing import Dict, List, Tuple
logger = logging.getLogger(__name__)

class TokenizerTrainer:
    def __init__(self):
        self.bpe   = None
        self.vocab = None

    def run(self, cleaned_dir: Path, save_dir: Path, run_id: str = "") -> None:
        from tokenizer.core.pre_tokenizer import PreTokenizer
        from tokenizer.core.bpe_processor import BPEProcessor
        from tokenizer.core.vocabulary_manager import VocabularyManager
        from modules.utils.file_handler import list_files
        from configs.loader import get_config
        cfg    = get_config()
        t_cfg  = cfg.tokenizer

        pre_tok = PreTokenizer(lowercase=False, add_eow=True)
        # list_files already uses natural sort (wk_0,1,2..10,11..)
        files   = list_files(Path(cleaned_dir), "*.txt")
        if not files:
            raise FileNotFoundError(f"No cleaned files in {cleaned_dir}")

        logger.info(
            "TokenizerTrainer: %d files  order: %s … %s",
            len(files), files[0].name, files[-1].name,
        )
        word_freq: Counter = Counter()
        for fpath in files:
            with fpath.open("r", encoding="utf-8", errors="replace") as fh:
                for line in fh:
                    words = pre_tok.split_words(line.strip())
                    word_freq.update(words)

        logger.info("Unique words: %d", len(word_freq))
        bpe   = BPEProcessor(num_merges=t_cfg.num_merges, min_freq=t_cfg.min_freq)
        bpe.train(word_freq)
        self.bpe = bpe

        vocab_mgr = VocabularyManager(vocab_size=t_cfg.vocab_size, min_freq=t_cfg.min_freq)
        vocab_mgr.build_from_bpe(bpe, word_freq)
        self.vocab = vocab_mgr

        Path(save_dir).mkdir(parents=True, exist_ok=True)
        from modules.utils.safe_writer import atomic_write_json
        prefix = f"{run_id}_" if run_id else ""
        atomic_write_json(Path(save_dir) / f"{prefix}vocabulary.json",
                          vocab_mgr.to_dict())
        atomic_write_json(Path(save_dir) / f"{prefix}bpe_merges.json",
                          bpe.get_state())
        atomic_write_json(Path(save_dir) / f"{prefix}vocab_meta.json",
                          {"vocab_size": vocab_mgr.size, "merges": len(bpe.merges),
                           "run_id": run_id, "files": len(files)})
        logger.info("Tokenizer saved: vocab_size=%d merges=%d", vocab_mgr.size, len(bpe.merges))
