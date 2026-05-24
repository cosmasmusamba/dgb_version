"""
Streaming dataset loader for finetune JSONL files.
Uses modules.utils.streaming and path_resolver to locate files and stream safely.
Provides sample loader for validation and batch streaming with resume support.
"""
import json
from typing import Generator, Iterable, Dict, Any, Optional

from modules.utils.path_resolver import PathResolver
from modules.utils.streaming import stream_jsonl
from modules.utils.file_handler import FileHandler
from modules.utils.progress_tracking import ProgressTracker
from modules.utils.run_context import RunContext

class FinetuneDatasetLoader:
    def __init__(self, config: Dict[str, Any], run_ctx: Optional[RunContext] = None):
        self.config = config
        self.run_ctx = run_ctx or RunContext.default()
        self.path_resolver = PathResolver(self.run_ctx)
        self.file_handler = FileHandler()
        self.dataset_path = self.path_resolver.resolve(config["finetune"]["dataset_path"])

    def load(self) -> Generator[Dict[str, Any], None, None]:
        """
        Stream JSONL entries from dataset_path.
        Yields parsed JSON objects one by one.
        """
        for obj in stream_jsonl(self.dataset_path):
            yield obj

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
