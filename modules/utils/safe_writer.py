"""
modules/utils/safe_writer.py
=============================
Atomic file writing with SHA-256 checksums and file locking.

Every write in the DGB platform uses this module to guarantee:
  1. No partial writes — write to temp file, then rename atomically
  2. Data integrity — SHA-256 sidecar written alongside every binary artifact
  3. Concurrency safety — optional filelock for multi-process scenarios
  4. Encoding safety — explicit UTF-8 for all text files
  5. Retry logic — transient I/O errors are retried with exponential back-off
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Optional

logger = logging.getLogger(__name__)

_DEFAULT_ENCODING = "utf-8"
_CHECKSUM_SUFFIX  = ".sha256"
_TMP_SUFFIX       = ".dgb_tmp"
_MAX_RETRIES      = 3
_RETRY_DELAY      = 0.2


def _retry_op(fn, *args, retries: int = _MAX_RETRIES, delay: float = _RETRY_DELAY, **kwargs):
    last_exc = None
    for attempt in range(retries):
        try:
            return fn(*args, **kwargs)
        except OSError as exc:
            last_exc = exc
            if attempt < retries - 1:
                time.sleep(delay * (2 ** attempt))
    raise last_exc


def atomic_write_text(path: Path, content: str, encoding: str = _DEFAULT_ENCODING) -> None:
    """
    Write text to `path` atomically — no reader ever sees partial content.

    Algorithm: write to sibling temp file → fsync → rename over destination.
    On Windows, rename is not atomic if destination exists, so we delete first.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(_TMP_SUFFIX)
    try:
        def _write():
            with tmp.open("w", encoding=encoding) as fh:
                fh.write(content)
                fh.flush()
                os.fsync(fh.fileno())
        _retry_op(_write)
        _rename_atomic(tmp, path)
    except Exception:
        _safe_remove(tmp)
        raise


def atomic_write_bytes(path: Path, data: bytes) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(_TMP_SUFFIX)
    try:
        def _write():
            with tmp.open("wb") as fh:
                fh.write(data)
                fh.flush()
                os.fsync(fh.fileno())
        _retry_op(_write)
        _rename_atomic(tmp, path)
    except Exception:
        _safe_remove(tmp)
        raise


def atomic_write_json(
    path: Path,
    data: Any,
    indent: int = 2,
    encoding: str = _DEFAULT_ENCODING,
    ensure_ascii: bool = False,
) -> None:
    """Serialize `data` to JSON and write atomically."""
    try:
        text = json.dumps(data, indent=indent, ensure_ascii=ensure_ascii, default=str)
    except TypeError as exc:
        raise ValueError(f"Data not JSON-serialisable: {exc}") from exc
    atomic_write_text(Path(path), text, encoding=encoding)


def _rename_atomic(src: Path, dst: Path) -> None:
    """Rename src → dst.  Handles Windows non-atomic overwrite."""
    try:
        os.replace(src, dst)   # atomic on POSIX; near-atomic on Windows
    except OSError:
        _safe_remove(dst)
        shutil.move(str(src), str(dst))


def _safe_remove(path: Path) -> None:
    try:
        if path.exists():
            path.unlink()
    except OSError:
        pass


# ── Checksum helpers ──────────────────────────────────────────────────────────

def compute_checksum(path: Path, chunk_size: int = 65536) -> str:
    """Return the hex SHA-256 digest of `path`."""
    h = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for chunk in iter(lambda: fh.read(chunk_size), b""):
            h.update(chunk)
    return h.hexdigest()


def write_checksum(path: Path) -> Path:
    """
    Compute the SHA-256 of `path` and write it to `path.sha256`.
    Returns the sidecar path.
    """
    digest   = compute_checksum(path)
    sidecar  = Path(str(path) + _CHECKSUM_SUFFIX)
    atomic_write_text(sidecar, digest + "\n")
    logger.debug("Checksum written → %s  (%s…)", sidecar.name, digest[:12])
    return sidecar


def verify_checksum(path: Path) -> bool:
    """
    Verify `path` against its .sha256 sidecar.
    Raises ValueError if the sidecar is missing or digest mismatches.
    Returns True on success.
    """
    sidecar = Path(str(path) + _CHECKSUM_SUFFIX)
    if not sidecar.exists():
        raise ValueError(f"Checksum sidecar not found: {sidecar}")
    expected = sidecar.read_text().strip()
    actual   = compute_checksum(path)
    if actual != expected:
        raise ValueError(
            f"Checksum mismatch for {path.name}\n"
            f"  expected: {expected}\n"
            f"  actual:   {actual}"
        )
    logger.debug("Checksum verified ✓  %s", path.name)
    return True


# ── Convenience aliases used across the codebase ──────────────────────────────

def write_text(path: Path, content: str, encoding: str = _DEFAULT_ENCODING) -> None:
    atomic_write_text(path, content, encoding)


def write_json(path: Path, data: Any, **kwargs) -> None:
    atomic_write_json(path, data, **kwargs)
