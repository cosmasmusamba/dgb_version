"""
trainer/core/dataset_loader.py
================================
Streaming dataset for large text corpora — never loads full file into memory.
FIX T6: num_workers=0 default; worker sharding uses line-level not file-level.

All internal file lists are sorted with natural sort so that
wk_2.txt always comes before wk_10.txt.
"""
from __future__ import annotations
import logging, re, random
from pathlib import Path
from typing import Callable, Iterable, List, Optional, Tuple
logger = logging.getLogger(__name__)

try:
    import torch
    from torch.utils.data import IterableDataset, DataLoader
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


def _natural_sort_key(path: Path):
    """Sort key for human-numeric ordering: wk_2 < wk_10 < wk_100."""
    name = path.name if isinstance(path, Path) else str(path)
    return [
        int(p) if p.isdigit() else p.lower()
        for p in re.split(r"(\d+)", name)
    ]


class StreamingTextDataset(IterableDataset if _HAS_TORCH else object):
    """Streams (src, tgt) pairs from cleaned text files for causal LM training."""

    def __init__(self, tokenizer, cleaned_dir, max_seq_len=512,
                 start_file_idx=0, start_line_idx=0, on_line=None):
        self._tok      = tokenizer
        self._dir      = Path(cleaned_dir)
        self._msl      = max_seq_len
        self._sf       = start_file_idx
        self._sl       = start_line_idx
        self._on_line  = on_line
        # Natural sort so wk_2 < wk_10 < wk_100
        self._files    = sorted(self._dir.glob("*.txt"), key=_natural_sort_key)

    def __iter__(self):
        import torch
        worker_info = torch.utils.data.get_worker_info() if _HAS_TORCH else None

        for fi, fpath in enumerate(self._files):
            if fi < self._sf: continue
            if worker_info is not None and fi % worker_info.num_workers != worker_info.id:
                continue
            try:
                with fpath.open("r", encoding="utf-8", errors="replace") as fh:
                    for li, line in enumerate(fh):
                        if fi == self._sf and li < self._sl: continue
                        line = line.strip()
                        if not line: continue
                        if self._on_line: self._on_line(fi, li)
                        ids = self._tok.encode(line, add_special_tokens=True,
                                               truncation=True, max_length=self._msl)
                        if len(ids) < 3: continue
                        ids = ids[:self._msl]
                        src = torch.tensor(ids[:-1], dtype=torch.long)
                        tgt = torch.tensor(ids[1:],  dtype=torch.long)
                        yield src, tgt
            except Exception as exc:
                logger.warning("Dataset error in %s: %s", fpath.name, exc)


def build_streaming_loader(dataset, batch_size, num_workers=0,
                           pin_memory=False, pad_id=0):
    """Build a DataLoader with padding collate."""
    import torch
    from torch.nn.utils.rnn import pad_sequence

    def _collate(batch):
        srcs, tgts = zip(*batch)
        srcs = pad_sequence(srcs, batch_first=True, padding_value=pad_id)
        tgts = pad_sequence(tgts, batch_first=True, padding_value=pad_id)
        return srcs, tgts

    return DataLoader(
        dataset, batch_size=batch_size, num_workers=num_workers,
        pin_memory=pin_memory, collate_fn=_collate, drop_last=False,
    )
