"""
finetune/core/finetune_dataset_loader.py
Streaming dataset loader for finetune JSONL files.
Uses modules.utils.streaming to stream JSONL files safely.
"""
import json
from typing import Generator, Dict, Any, Optional, Iterable
from pathlib import Path

from modules.utils.streaming import stream_jsonl
from modules.utils.path_resolver import init_path_resolver
from modules.utils.run_context import get_run_context


class FinetuneDatasetLoader:
    def __init__(self, config: Dict[str, Any], run_ctx: Optional = None):
        from configs.loader import get_config
        
        self.config = config
        self.run_ctx = run_ctx or get_run_context()
        
        cfg = get_config()
        self.path_resolver = init_path_resolver(cfg.project.model_id, cfg)
        
        # Get dataset path from config
        fin_cfg = getattr(cfg, "finetune", None)
        if fin_cfg and hasattr(fin_cfg, "dataset_path"):
            self.dataset_path = self.path_resolver.raw_dir() / fin_cfg.dataset_path
        else:
            self.dataset_path = Path("datasets/dgb1/finetune/manual/expert.jsonl")
    
    def load(self) -> Generator[Dict[str, Any], None, None]:
        """
        Stream JSONL entries from dataset_path.
        Yields parsed JSON objects converted to unified format.
        """
        from finetune.utils.schema_validator import convert_dataset_format
        
        for entry in stream_jsonl(self.dataset_path):
            yield convert_dataset_format(entry)
    
    def load_sample(self, n: int = 10) -> Iterable[Dict[str, Any]]:
        """
        Return first n entries as a list (used for lightweight validation).
        """
        sample = []
        for i, entry in enumerate(self.load()):
            sample.append(entry)
            if i + 1 >= n:
                break
        return sample
    
    def stream_batches(self, batch_size: int, start_offset: int = 0) -> Generator[list, None, None]:
        """
        Stream batches of size batch_size. Supports start_offset to resume mid-file.
        """
        batch = []
        idx = 0
        for entry in self.load():
            if idx < start_offset:
                idx += 1
                continue
            batch.append(entry)
            idx += 1
            if len(batch) >= batch_size:
                yield batch
                batch = []
        if batch:
            yield batch