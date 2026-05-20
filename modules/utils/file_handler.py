"""
modules/utils/file_handler.py
==============================
Enterprise file I/O utilities — directory management, file discovery,
line iteration, size reporting, and JSON reading.
"""
from __future__ import annotations

import json
import logging
import os
from pathlib import Path
from typing import Any, Generator, Iterable, List, Optional

logger = logging.getLogger(__name__)


def ensure_dir(path: Path) -> Path:
    """Create directory (and parents) if it doesn't exist. Return path."""
    path = Path(path)
    path.mkdir(parents=True, exist_ok=True)
    return path


def list_files(
    directory: Path,
    glob:      str = "*.txt",
    recursive: bool = False,
) -> List[Path]:
    """Return sorted list of files matching `glob` in `directory`."""
    directory = Path(directory)
    if not directory.exists():
        logger.debug("Directory not found: %s", directory)
        return []
    pattern = f"**/{glob}" if recursive else glob
    files   = sorted(directory.glob(pattern))
    logger.debug("list_files(%s, %s) → %d files", directory.name, glob, len(files))
    return files


def iter_lines(
    path:     Path,
    encoding: str = "utf-8",
    skip:     int = 0,
) -> Generator[str, None, None]:
    """Yield stripped, non-empty lines from `path`, optionally skipping `skip` lines."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    with path.open("r", encoding=encoding, errors="replace") as fh:
        for i, line in enumerate(fh):
            if i < skip:
                continue
            stripped = line.rstrip("\n\r")
            if stripped:
                yield stripped


def read_json(path: Path, encoding: str = "utf-8") -> Any:
    """Read and parse a JSON file. Returns Python object."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"JSON file not found: {path}")
    with path.open("r", encoding=encoding) as fh:
        return json.load(fh)


def write_text(path: Path, content: str, encoding: str = "utf-8") -> None:
    """Write text to path, creating parent directories."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding=encoding) as fh:
        fh.write(content)


def file_size_mb(path: Path) -> float:
    """Return file size in megabytes."""
    return Path(path).stat().st_size / (1024 * 1024)


def file_line_count(path: Path, encoding: str = "utf-8") -> int:
    """Count non-empty lines in a text file efficiently."""
    count = 0
    with Path(path).open("r", encoding=encoding, errors="replace") as fh:
        for line in fh:
            if line.strip():
                count += 1
    return count


def latest_file_by_name(
    directory: Path,
    glob:      str = "*.json",
) -> Optional[Path]:
    """Return the lexicographically last file matching glob (datetime prefix = newest)."""
    files = list_files(directory, glob=glob)
    return files[-1] if files else None
