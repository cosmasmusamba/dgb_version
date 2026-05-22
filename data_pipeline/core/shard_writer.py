"""
data_pipeline/core/shard_writer.py
=====================================
Rolling shard writer with configurable size limits, atomic rotation,
and append-only JSONL output.

Features
--------
- Shards are rotated when they reach max_shard_bytes
- Every shard is written to a .partial file and renamed on close
- Partial shards are recovered on startup
- Thread-safe: one writer per source, called from async worker threads
- Integrates with QuotaManager to respect per-source storage quotas
- SHA-256 sidecar written on each closed shard for integrity verification
- Shard filenames: {run_id}_{source}_{shard_idx:06d}.jsonl
"""
from __future__ import annotations

import hashlib
import json
import logging
import os
import threading
import time
from pathlib import Path
from typing import List, Optional

from data_pipeline.core.document import Document
from modules.utils.safe_writer import write_checksum

logger = logging.getLogger(__name__)

_DEFAULT_MAX_SHARD_BYTES  = 512 * 1024 * 1024   # 512 MB
_DEFAULT_MAX_SHARD_RECORDS = 1_000_000


class ShardWriter:
    """
    Append-only, size-bounded, rolling JSONL shard writer.

    Parameters
    ----------
    output_dir:         Directory for completed shards.
    source_name:        Source identifier (e.g. "wikipedia").
    run_id:             Run datetime prefix.
    max_shard_bytes:    Rotate when shard exceeds this size.
    max_shard_records:  Rotate after this many records (whichever comes first).
    quota_manager:      Optional QuotaManager for source-level quota enforcement.
    checkpoint_mgr:     CheckpointManager — updated on every rotation.
    """

    def __init__(
        self,
        output_dir:         Path,
        source_name:        str,
        run_id:             str,
        max_shard_bytes:    int   = _DEFAULT_MAX_SHARD_BYTES,
        max_shard_records:  int   = _DEFAULT_MAX_SHARD_RECORDS,
        quota_manager=None,
        checkpoint_mgr=None,
        compress:           bool  = False,
    ) -> None:
        self._dir          = Path(output_dir)
        self._source       = source_name
        self._run_id       = run_id
        self._max_bytes    = max_shard_bytes
        self._max_records  = max_shard_records
        self._quota        = quota_manager
        self._ckpt         = checkpoint_mgr
        self._compress     = compress
        self._lock         = threading.Lock()

        self._dir.mkdir(parents=True, exist_ok=True)

        self._shard_idx      = 0
        self._records        = 0
        self._bytes_written  = 0
        self._total_records  = 0
        self._total_bytes    = 0
        self._completed_shards: List[str] = []
        self._fh             = None
        self._current_path:  Optional[Path] = None
        self._partial_path:  Optional[Path] = None
        self._hasher         = hashlib.sha256()

        self._recover_partial()
        self._open_shard()
        logger.info(
            "ShardWriter[%s]: dir=%s  max=%dMB  shard_idx=%d",
            source_name, self._dir, max_shard_bytes // 1024**2, self._shard_idx,
        )

    # ── Public API ────────────────────────────────────────────────────

    def write(self, doc: Document) -> bool:
        """
        Write one Document to the current shard.
        Rotates automatically if limits are reached.
        Returns True if written, False if quota exceeded.
        """
        if self._quota and not self._quota.can_write(self._source, 1):
            logger.warning(
                "ShardWriter[%s]: quota exceeded — write rejected", self._source
            )
            return False

        line = doc.to_jsonl()
        line_bytes = line.encode("utf-8")

        with self._lock:
            if self._needs_rotation(len(line_bytes)):
                self._rotate()
            self._write_line(line, line_bytes)

        if self._quota:
            self._quota.record_write(self._source, len(line_bytes))

        return True

    def write_batch(self, docs: List[Document]) -> int:
        """Write a batch.  Returns number of docs written."""
        written = 0
        for doc in docs:
            if self.write(doc):
                written += 1
        return written

    def close(self) -> None:
        """Finalise and close the current shard."""
        with self._lock:
            self._close_shard()

    def flush(self) -> None:
        with self._lock:
            if self._fh:
                try:
                    self._fh.flush()
                    os.fsync(self._fh.fileno())
                except Exception:
                    pass

    @property
    def total_records(self) -> int:
        return self._total_records

    @property
    def total_bytes(self) -> int:
        return self._total_bytes

    @property
    def completed_shards(self) -> List[str]:
        return list(self._completed_shards)

    @property
    def current_shard_records(self) -> int:
        return self._records

    # ── Internal ──────────────────────────────────────────────────────

    def _shard_name(self, idx: int, partial: bool = False) -> str:
        suffix = ".partial" if partial else ".jsonl"
        return f"{self._run_id}_{self._source}_{idx:06d}{suffix}"

    def _needs_rotation(self, incoming_bytes: int) -> bool:
        if self._records >= self._max_records:
            return True
        if self._bytes_written + incoming_bytes >= self._max_bytes:
            return True
        return False

    def _write_line(self, line: str, line_bytes: bytes) -> None:
        if self._fh is None:
            self._open_shard()
        self._fh.write(line)
        self._hasher.update(line_bytes)
        self._records       += 1
        self._bytes_written += len(line_bytes)
        self._total_records += 1
        self._total_bytes   += len(line_bytes)

    def _open_shard(self) -> None:
        name              = self._shard_name(self._shard_idx, partial=True)
        self._partial_path = self._dir / name
        self._fh          = self._partial_path.open("a", encoding="utf-8", buffering=65536)
        self._records       = self._partial_path.stat().st_size // 200   # estimate
        self._bytes_written = self._partial_path.stat().st_size
        self._hasher        = hashlib.sha256()
        logger.debug("ShardWriter[%s]: opened shard %s", self._source, name)

    def _rotate(self) -> None:
        self._close_shard()
        self._shard_idx += 1
        self._records       = 0
        self._bytes_written = 0
        self._open_shard()

    def _close_shard(self) -> None:
        if self._fh is None:
            return
        try:
            self._fh.flush()
            os.fsync(self._fh.fileno())
            self._fh.close()
        except Exception as exc:
            logger.warning("ShardWriter[%s]: flush error: %s", self._source, exc)
        finally:
            self._fh = None

        if self._partial_path and self._partial_path.exists():
            # Rename .partial → .jsonl
            final_name = self._shard_name(self._shard_idx)
            final_path = self._dir / final_name
            try:
                os.replace(self._partial_path, final_path)
                # Write SHA-256 sidecar
                write_checksum(final_path)
                self._completed_shards.append(str(final_path))
                logger.info(
                    "ShardWriter[%s]: closed shard %s  records=%d  bytes=%s",
                    self._source, final_name, self._records,
                    f"{self._bytes_written/1024**2:.1f}MB",
                )
                # Update checkpoint
                if self._ckpt:
                    cp = self._ckpt.get(self._source)
                    cp.completed_shards.append(str(final_path))
                    cp.total_bytes_out += self._bytes_written
                    self._ckpt.save(self._source)
            except Exception as exc:
                logger.error(
                    "ShardWriter[%s]: cannot finalise shard %s: %s",
                    self._source, final_name, exc,
                )

        self._partial_path = None

    def _recover_partial(self) -> None:
        """
        On startup, find any .partial shards and determine the highest
        completed shard index to resume from.
        """
        partials  = sorted(self._dir.glob(f"{self._run_id}_{self._source}_*.partial"))
        completed = sorted(self._dir.glob(f"{self._run_id}_{self._source}_*.jsonl"))

        self._completed_shards = [str(p) for p in completed]

        # Determine next shard index
        all_idxs = []
        for p in list(partials) + list(completed):
            stem = p.stem   # e.g. 20260520_wikipedia_000012
            try:
                idx = int(stem.rsplit("_", 1)[-1])
                all_idxs.append(idx)
            except ValueError:
                pass

        if all_idxs:
            self._shard_idx = max(all_idxs)
            logger.info(
                "ShardWriter[%s]: resuming from shard index %d  "
                "(%d completed shards)",
                self._source, self._shard_idx, len(completed),
            )

    def __enter__(self) -> "ShardWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()