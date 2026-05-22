"""
data_pipeline/core/checkpoint.py
===================================
Durable, multi-level checkpoint system for the data pipeline.

Checkpoint granularity
----------------------
  source    → which sources have been started / completed
  stream    → byte offset within the current download stream
  shard     → which output shard is being written and at what line
  batch     → the last fully committed batch index
  stage     → which preprocessing stage has been reached
  dedup     → snapshot of deduplication index state
  partial   → partially written shard path (for atomic finalisation)

All state is written atomically (temp → rename) so a crash at any point
leaves the previous valid checkpoint intact.

Checkpoint files are datetime-prefixed following DGB conventions:
    checkpoints/logs/{model_id}/pipeline/{source}/{run_id}_checkpoint.json
"""
from __future__ import annotations

import json
import logging
import os
import threading
import time
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


@dataclass
class StreamOffset:
    """Resume position within a remote dump stream."""
    url:              str   = ""
    byte_offset:      int   = 0
    records_seen:     int   = 0
    records_accepted: int   = 0
    last_updated:     float = field(default_factory=time.time)


@dataclass
class ShardState:
    """State of the current output shard."""
    shard_id:      str   = ""
    shard_path:    str   = ""
    records_written: int = 0
    bytes_written:  int  = 0
    is_open:        bool = False
    last_updated:   float = field(default_factory=time.time)


@dataclass
class StageProgress:
    """Progress within a single preprocessing stage."""
    stage_name:    str   = ""
    records_in:    int   = 0
    records_out:   int   = 0
    records_rejected: int = 0
    started_at:    float = field(default_factory=time.time)
    completed:     bool  = False


@dataclass
class SourceCheckpoint:
    """
    Complete resumable state for one pipeline source.

    This is the unit of persistence — one JSON file per source per run.
    """
    source_name:     str   = ""
    run_id:          str   = ""
    started_at:      float = field(default_factory=time.time)
    updated_at:      float = field(default_factory=time.time)
    completed:       bool  = False
    failed:          bool  = False
    error:           str   = ""

    # Stream-level resume
    streams:         Dict[str, StreamOffset] = field(default_factory=dict)
    current_stream:  str = ""

    # Shard-level resume
    current_shard:   ShardState = field(default_factory=ShardState)
    completed_shards: List[str] = field(default_factory=list)

    # Batch-level resume
    last_batch_idx:  int = 0
    total_batches:   int = 0

    # Stage progress
    stages:          Dict[str, StageProgress] = field(default_factory=dict)

    # Aggregate statistics
    total_raw:       int = 0
    total_extracted: int = 0
    total_accepted:  int = 0
    total_rejected:  int = 0
    total_bytes_out: int = 0
    dedup_removed:   int = 0

    # Deduplication index snapshot path (for external index state)
    dedup_index_path: str = ""

    # ── Accessors ─────────────────────────────────────────────────────

    def get_stream(self, url: str) -> StreamOffset:
        if url not in self.streams:
            self.streams[url] = StreamOffset(url=url)
        return self.streams[url]

    def update_stream(self, url: str, byte_offset: int,
                      records_seen: int = 0, records_accepted: int = 0) -> None:
        s = self.get_stream(url)
        s.byte_offset       = byte_offset
        s.records_seen      += records_seen
        s.records_accepted  += records_accepted
        s.last_updated      = time.time()
        self.current_stream = url
        self.updated_at     = time.time()

    def get_stage(self, name: str) -> StageProgress:
        if name not in self.stages:
            self.stages[name] = StageProgress(stage_name=name)
        return self.stages[name]

    def mark_completed(self) -> None:
        self.completed  = True
        self.updated_at = time.time()

    def mark_failed(self, error: str) -> None:
        self.failed     = True
        self.error      = error
        self.updated_at = time.time()

    # ── Serialisation ─────────────────────────────────────────────────

    def to_dict(self) -> dict:
        d = asdict(self)
        return d

    @classmethod
    def from_dict(cls, d: dict) -> "SourceCheckpoint":
        streams = {
            k: StreamOffset(**v)
            for k, v in d.pop("streams", {}).items()
        }
        shard_d = d.pop("current_shard", {})
        stages  = {
            k: StageProgress(**v)
            for k, v in d.pop("stages", {}).items()
        }
        cp = cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})
        cp.streams       = streams
        cp.current_shard = ShardState(**shard_d) if shard_d else ShardState()
        cp.stages        = stages
        return cp


class CheckpointManager:
    """
    Manages checkpoint persistence for the entire pipeline.

    One CheckpointManager per pipeline run.
    Stores one SourceCheckpoint JSON per source.

    All writes are atomic (write temp → rename) and thread-safe.
    """

    def __init__(
        self,
        checkpoint_dir: Path,
        run_id:         str,
        flush_interval: float = 30.0,     # seconds between auto-flushes
    ) -> None:
        self._dir      = Path(checkpoint_dir)
        self._run_id   = run_id
        self._interval = flush_interval
        self._lock     = threading.RLock()
        self._sources: Dict[str, SourceCheckpoint] = {}
        self._dir.mkdir(parents=True, exist_ok=True)
        logger.info("CheckpointManager: dir=%s  run_id=%s", self._dir, run_id)

    # ── Source checkpoints ────────────────────────────────────────────

    def get(self, source_name: str) -> SourceCheckpoint:
        """Return (and auto-load) the checkpoint for a source."""
        with self._lock:
            if source_name not in self._sources:
                self._sources[source_name] = self._load_or_create(source_name)
            return self._sources[source_name]

    def save(self, source_name: str) -> None:
        """Atomically persist the checkpoint for a source."""
        with self._lock:
            cp   = self._sources.get(source_name)
            if cp is None:
                return
            path = self._path(source_name)
            tmp  = path.with_suffix(".tmp.json")
            try:
                tmp.write_text(
                    json.dumps(cp.to_dict(), indent=2, default=str),
                    encoding="utf-8",
                )
                os.replace(tmp, path)
                logger.debug("Checkpoint saved: %s", path.name)
            except Exception as exc:
                logger.warning("Checkpoint save failed for %s: %s", source_name, exc)
                try:
                    tmp.unlink()
                except Exception:
                    pass

    def save_all(self) -> None:
        with self._lock:
            for name in list(self._sources):
                self.save(name)

    def mark_completed(self, source_name: str) -> None:
        cp = self.get(source_name)
        cp.mark_completed()
        self.save(source_name)

    def mark_failed(self, source_name: str, error: str) -> None:
        cp = self.get(source_name)
        cp.mark_failed(error)
        self.save(source_name)

    def is_completed(self, source_name: str) -> bool:
        return self.get(source_name).completed

    def list_sources(self) -> List[str]:
        """Return names of all sources with saved checkpoints."""
        return [
            p.stem.replace(f"{self._run_id}_", "").replace("_checkpoint", "")
            for p in self._dir.glob(f"{self._run_id}_*_checkpoint.json")
        ]

    # ── Helpers ────────────────────────────────────────────────────────

    def _path(self, source_name: str) -> Path:
        return self._dir / f"{self._run_id}_{source_name}_checkpoint.json"

    def _load_or_create(self, source_name: str) -> SourceCheckpoint:
        path = self._path(source_name)
        # Also check older runs with same source
        candidates = sorted(
            self._dir.glob(f"*_{source_name}_checkpoint.json"),
            reverse=True,
        )
        for cand in candidates:
            try:
                d  = json.loads(cand.read_text(encoding="utf-8"))
                cp = SourceCheckpoint.from_dict(d)
                if not cp.completed and not cp.failed:
                    logger.info(
                        "Resuming %s from checkpoint %s "
                        "(raw=%d accepted=%d shards=%d)",
                        source_name, cand.name,
                        cp.total_raw, cp.total_accepted, len(cp.completed_shards),
                    )
                    return cp
            except Exception as exc:
                logger.debug("Cannot load checkpoint %s: %s", cand.name, exc)

        logger.info("New checkpoint for source: %s", source_name)
        return SourceCheckpoint(source_name=source_name, run_id=self._run_id)

    def summary(self) -> str:
        with self._lock:
            lines = [f"CheckpointManager run_id={self._run_id}"]
            for name, cp in self._sources.items():
                status = "done" if cp.completed else ("fail" if cp.failed else "running")
                lines.append(
                    f"  {name:<20} [{status}] "
                    f"raw={cp.total_raw} accepted={cp.total_accepted} "
                    f"shards={len(cp.completed_shards)}"
                )
            return "\n".join(lines)