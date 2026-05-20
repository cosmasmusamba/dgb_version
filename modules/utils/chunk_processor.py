"""
modules/utils/chunk_processor.py
==================================
Memory-bounded file processing via configurable line-count chunks.

chunk_file_lines() reads a text file in chunks of N characters,
yielding each chunk as a list of complete lines.  Never loads the
whole file into memory — essential for multi-GB Wikipedia dumps.
"""
from __future__ import annotations

import logging
from pathlib import Path
from typing import Generator, List

logger = logging.getLogger(__name__)


def chunk_file_lines(
    path:       Path,
    chunk_size: int  = 500_000,
    encoding:   str  = "utf-8",
) -> Generator[List[str], None, None]:
    """
    Yield successive non-overlapping chunks of lines from `path`.

    Each chunk contains approximately `chunk_size` characters worth of
    lines (lines are never split across chunk boundaries).

    Parameters
    ----------
    path:        Text file to process.
    chunk_size:  Target number of characters per chunk.
    encoding:    File encoding.

    Yields
    ------
    List[str]   — a batch of raw lines (including newline-stripped text)
    """
    path  = Path(path)
    chunk: List[str] = []
    total = 0

    with path.open("r", encoding=encoding, errors="replace") as fh:
        for raw_line in fh:
            line = raw_line.rstrip("\n\r")
            chunk.append(line)
            total += len(line)
            if total >= chunk_size:
                logger.debug(
                    "chunk_file_lines: yielding %d lines (%d chars) from %s",
                    len(chunk), total, path.name,
                )
                yield chunk
                chunk = []
                total = 0

    if chunk:
        logger.debug(
            "chunk_file_lines: final chunk %d lines (%d chars) from %s",
            len(chunk), total, path.name,
        )
        yield chunk
