"""
Validate finetune JSONL dataset entries.
Performs lightweight streaming validation so it can be used on large files.
"""
from typing import Iterable, Dict, Any

def validate_dataset(iterable: Iterable[Dict[str, Any]]):
    required = {"instruction", "input", "output"}
    for i, entry in enumerate(iterable):
        if not isinstance(entry, dict):
            raise ValueError(f"Dataset entry {i} is not an object: {entry}")
        missing = required - set(entry.keys())
        if missing:
            raise ValueError(f"Dataset entry {i} missing fields: {missing}")
        if not isinstance(entry["instruction"], str):
            raise TypeError(f"Dataset entry {i} 'instruction' must be a string")
        # optional: enforce length limits if present in config (not here to keep validator pure)
