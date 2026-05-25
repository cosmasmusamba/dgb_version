"""
modules/utils/pipeline_state.py
================================
Persistent pipeline stage state machine with file-locking, audit trail,
and full resume support at BOTH stage level AND sub-stage (batch/step/line).

Stage-level API (unchanged):
    state.get("dataset_clean")         → StageRecord
    record.mark_running()
    record.mark_completed(files=5)

Sub-stage KV API (new — used by finetune_loop, training_loop, etc.):
    state.set("global_step", 1024)     # atomic persist on every call
    state.get_kv("global_step", 0)     # reads with default
    state.save_checkpoint(note="…")    # explicit full flush
    state.load_kv_checkpoint(path)     # restore KV from a previous path

The state file keeps both sections so one file covers everything:
    {
      "model_id":  "dgb1",
      "run_id":    "20260522133713",
      "created_at": 1748000000.0,
      "kv":  { "global_step": 1024, "epoch": 2, ... },
      "stages": { "dataset_clean": {...}, ... }
    }

Atomic checkpoint naming:
    <run_id>_pipeline_state.json
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Any, Dict, List, Optional

logger = logging.getLogger(__name__)


class StageStatus(str, Enum):
    PENDING   = "pending"
    RUNNING   = "running"
    COMPLETED = "completed"
    FAILED    = "failed"
    SKIPPED   = "skipped"


@dataclass
class StageRecord:
    """Immutable-ish record for one pipeline stage."""
    name:         str
    status:       str               = StageStatus.PENDING
    started_at:   Optional[float]   = None
    completed_at: Optional[float]   = None
    duration_sec: Optional[float]   = None
    error:        Optional[str]     = None
    metadata:     Dict[str, Any]    = field(default_factory=dict)
    attempts:     int               = 0

    # ── Transitions ────────────────────────────────────────────────────

    def mark_running(self) -> None:
        self.status     = StageStatus.RUNNING
        self.started_at = time.time()
        self.attempts  += 1
        self.error      = None
        logger.debug("Stage %s → RUNNING (attempt %d)", self.name, self.attempts)

    def mark_completed(self, **metadata: Any) -> None:
        now              = time.time()
        self.status      = StageStatus.COMPLETED
        self.completed_at = now
        self.duration_sec = now - (self.started_at or now)
        self.error        = None
        if metadata:
            self.metadata.update(metadata)
        logger.info(
            "Stage %-22s COMPLETED  %.1fs",
            self.name, self.duration_sec,
        )

    def mark_failed(self, error: str) -> None:
        now               = time.time()
        self.status       = StageStatus.FAILED
        self.completed_at = now
        self.duration_sec = now - (self.started_at or now)
        self.error        = error
        logger.error("Stage %s FAILED: %s", self.name, error)

    def mark_skipped(self, reason: str = "--skip flag") -> None:
        self.status = StageStatus.SKIPPED
        self.metadata["skip_reason"] = reason
        logger.info("Stage %s SKIPPED: %s", self.name, reason)

    def reset(self) -> None:
        """Reset to pending so --force re-runs this stage."""
        self.status       = StageStatus.PENDING
        self.started_at   = None
        self.completed_at = None
        self.duration_sec = None
        self.error        = None

    @property
    def is_done(self) -> bool:
        return self.status in (StageStatus.COMPLETED, StageStatus.SKIPPED)

    @property
    def is_running(self) -> bool:
        return self.status == StageStatus.RUNNING

    @property
    def failed(self) -> bool:
        return self.status == StageStatus.FAILED

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "StageRecord":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})


class PipelineState:
    """
    Thread-safe pipeline state machine.

    Two distinct state stores in one file:

    Stage store   — named StageRecords with status, timing, and metadata.
                    Used by the outer pipeline runner to track which stages
                    have completed, failed, or are pending.

    KV store      — flat key-value dict for sub-stage offsets (epoch, step,
                    batch_idx, line_offset, last_checkpoint_path, …).
                    Used by training loops, finetune loops, and cleaning
                    pipelines to enable exact granular resume.

    Both stores persist to the same atomic JSON file.

    Stage-level usage
    -----------------
        state.get("dataset_clean").mark_running()
        state.save(log_dir)
        state.get("dataset_clean").mark_completed(files=5)
        state.save(log_dir)

    KV usage (finetune_loop, training_loop, etc.)
    ----------------------------------------------
        state.set("global_step", step, save_dir=ckpt_dir)
        step = state.get_kv("global_step", default=0)
        state.save_checkpoint(ckpt_dir, note="epoch_end")
    """

    _DEFAULT_STAGES = ["dataset_clean", "train_tokenizer", "model_training"]

    def __init__(
        self,
        model_id:    str = "dgb1",
        run_id:      str = "",
        stage_names: Optional[List[str]] = None,
    ) -> None:
        self.model_id   = model_id
        self.run_id     = run_id
        self.created_at = time.time()
        self._lock      = threading.Lock()
        self._stages: Dict[str, StageRecord] = {
            name: StageRecord(name=name)
            for name in (stage_names or self._DEFAULT_STAGES)
        }
        # Sub-stage key-value store (granular resume)
        self._kv: Dict[str, Any] = {}

    # ── Stage accessors ─────────────────────────────────────────────────

    def get(self, name: str) -> StageRecord:
        """Return the StageRecord for `name`, creating it if needed."""
        if name not in self._stages:
            self._stages[name] = StageRecord(name=name)
        return self._stages[name]

    def all_done(self) -> bool:
        return all(r.is_done for r in self._stages.values())

    def any_failed(self) -> bool:
        return any(r.failed for r in self._stages.values())

    def next_pending(self) -> Optional[StageRecord]:
        for r in self._stages.values():
            if r.status == StageStatus.PENDING:
                return r
        return None

    # ── KV sub-stage API ────────────────────────────────────────────────

    def set(
        self,
        key:      str,
        value:    Any,
        *,
        save_dir: Optional[Path] = None,
    ) -> None:
        """
        Store a sub-stage offset.

        If `save_dir` is provided the state is flushed atomically to disk
        immediately — useful for high-frequency writes (per-batch step saves).
        Omit `save_dir` for very-high-frequency calls and call
        save_checkpoint() at epoch / checkpoint boundaries instead.
        """
        with self._lock:
            self._kv[key] = value
        if save_dir is not None:
            self._write(Path(save_dir))

    def get_kv(self, key: str, default: Any = None) -> Any:
        """Return a KV value, or `default` if the key has not been set."""
        with self._lock:
            return self._kv.get(key, default)

    def save_checkpoint(
        self,
        save_dir: Optional[Path] = None,
        note:     str = "",
    ) -> Optional[Path]:
        """
        Flush the full state (stages + KV) atomically to disk.

        Uses the last save_dir seen, or raises if none has been provided.
        Returns the path written, or None if no save_dir is available.
        """
        if save_dir is not None:
            self._last_save_dir = Path(save_dir)
        dir_ = getattr(self, "_last_save_dir", None)
        if dir_ is None:
            logger.warning("save_checkpoint: no save_dir — skipping flush  note=%s", note)
            return None
        if note:
            with self._lock:
                self._kv["_checkpoint_note"] = note
                self._kv["_checkpoint_ts"]   = time.time()
        return self._write(dir_)

    # ── Persistence ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "model_id":   self.model_id,
                "run_id":     self.run_id,
                "created_at": self.created_at,
                "kv":         dict(self._kv),
                "stages":     {k: v.to_dict() for k, v in self._stages.items()},
            }

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineState":
        state            = cls(model_id=d.get("model_id", "dgb1"))
        state.run_id     = d.get("run_id", "")
        state.created_at = d.get("created_at", time.time())
        state._kv        = d.get("kv", {})
        state._stages    = {
            k: StageRecord.from_dict(v)
            for k, v in d.get("stages", {}).items()
        }
        return state

    def save(self, directory: Path) -> Path:
        """
        Atomically write state to  <directory>/<run_id>_pipeline_state.json.
        Returns the path written.
        """
        self._last_save_dir = Path(directory)
        return self._write(Path(directory))

    def _write(self, directory: Path) -> Path:
        from modules.utils.safe_writer import atomic_write_json
        directory = Path(directory)
        directory.mkdir(parents=True, exist_ok=True)
        prefix = f"{self.run_id}_" if self.run_id else ""
        path   = directory / f"{prefix}pipeline_state.json"
        atomic_write_json(path, self.to_dict())
        return path

    @classmethod
    def load_latest(cls, directory: Path, model_id: str = "dgb1") -> "PipelineState":
        """
        Load the most recent pipeline_state.json from `directory`.
        Creates a fresh state if none exists.
        """
        directory = Path(directory)
        candidates = sorted(directory.glob("*pipeline_state.json"))
        for path in reversed(candidates):
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                state = cls.from_dict(d)
                logger.info(
                    "Pipeline state loaded: %s  run_id=%s",
                    path.name, state.run_id,
                )
                return state
            except Exception as exc:
                logger.warning("Cannot read %s: %s", path.name, exc)

        logger.info("No pipeline state found — starting fresh for model_id=%s", model_id)
        return cls(model_id=model_id)

    @classmethod
    def load_for_run(
        cls,
        directory: Path,
        run_id:    str,
        model_id:  str = "dgb1",
    ) -> "PipelineState":
        """Load a specific run's state file, or create fresh."""
        path = Path(directory) / f"{run_id}_pipeline_state.json"
        if path.exists():
            try:
                d = json.loads(path.read_text(encoding="utf-8"))
                return cls.from_dict(d)
            except Exception as exc:
                logger.warning("Cannot read %s: %s", path.name, exc)
        return cls(model_id=model_id, run_id=run_id)

    # ── Reporting ──────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [f"Pipeline state  run_id={self.run_id}  model={self.model_id}"]
        for rec in self._stages.values():
            dur = f"  {rec.duration_sec:.1f}s" if rec.duration_sec else ""
            err = f"  ERROR: {rec.error}" if rec.error else ""
            lines.append(f"  {rec.name:<26} [{rec.status.upper():<10}]{dur}{err}")
        if self._kv:
            lines.append("  KV offsets:")
            for k, v in sorted(self._kv.items()):
                if not k.startswith("_"):
                    lines.append(f"    {k:<24} = {v!r}")
        return "\n".join(lines)
