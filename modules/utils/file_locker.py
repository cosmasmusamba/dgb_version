"""
modules/utils/file_locker.py
=============================
Cross-platform file locking using filelock with timeout and context manager.
"""
from __future__ import annotations

import logging
from pathlib import Path
from contextlib import contextmanager
from typing import Generator

from configs.constants import LOCK_TIMEOUT_SEC

logger = logging.getLogger(__name__)

try:
    from filelock import FileLock, Timeout as FileLockTimeout
    _HAS_FILELOCK = True
except ImportError:
    _HAS_FILELOCK = False
    logger.warning("filelock not installed — file locking disabled")


@contextmanager
def locked_file(path: Path, timeout: float = LOCK_TIMEOUT_SEC) -> Generator[Path, None, None]:
    """
    Context manager that acquires an exclusive file lock on `path`.

    If `filelock` is not installed, yields without locking (best-effort).
    If the lock cannot be acquired within `timeout` seconds, raises FileLockError.
    """
    from modules.utils.error_handler import FileLockError

    if not _HAS_FILELOCK:
        yield Path(path)
        return

    lock_path = Path(str(path) + ".lock")
    lock      = FileLock(str(lock_path), timeout=timeout)
    try:
        lock.acquire()
        logger.debug("Lock acquired: %s", lock_path.name)
        yield Path(path)
    except FileLockTimeout:
        raise FileLockError(str(path), timeout)
    finally:
        try:
            lock.release()
        except Exception:
            pass
