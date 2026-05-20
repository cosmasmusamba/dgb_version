"""
modules/utils/pipeline_state.py
================================
Persistent pipeline stage state machine with file-locking, audit trail,
and full resume support.

Every pipeline run maintains a PipelineState that records the status of
each stage (dataset_clean / train_tokenizer / model_training).  The state
is written atomically after every stage transition so a crash or interruption
anywhere in the pipeline results in a clean resume on the next run without
re-running completed stages.

State files are datetime-prefixed:
    20260518130915_pipeline_state.json

PipelineState.load_latest() discovers all state files in a directory
sorted by name (= chronological) and loads the most recent one.
A --force flag resets a single stage without affecting others.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from dataclasses import dataclass, field, asdict
from enum import Enum
from pathlib import Path
from typing import Dict, Optional, Any

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

    Stages are defined at construction and persist to a JSON file after
    every transition.  The file is written atomically (write temp → rename)
    so it is never left in a partially-written state.

    Usage
    -----
    state = PipelineState.load_latest(log_dir, model_id)
    state.run_id = ctx.run_id

    rec = state.get("dataset_clean")
    rec.mark_running(); state.save(log_dir)
    rec.mark_completed(files_processed=5); state.save(log_dir)

    # Force reset one stage:
    state.get("model_training").reset()
    state.save(log_dir)
    """

    _DEFAULT_STAGES = ["dataset_clean", "train_tokenizer", "model_training"]

    def __init__(
        self,
        model_id:    str = "dgb1",
        run_id:      str = "",
        stage_names: list[str] = None,
    ) -> None:
        self.model_id   = model_id
        self.run_id     = run_id
        self.created_at = time.time()
        self._lock      = threading.Lock()
        self._stages: Dict[str, StageRecord] = {
            name: StageRecord(name=name)
            for name in (stage_names or self._DEFAULT_STAGES)
        }

    # ── Accessors ──────────────────────────────────────────────────────

    def get(self, name: str) -> StageRecord:
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

    # ── Persistence ────────────────────────────────────────────────────

    def to_dict(self) -> dict:
        with self._lock:
            return {
                "model_id":   self.model_id,
                "run_id":     self.run_id,
                "created_at": self.created_at,
                "stages":     {k: v.to_dict() for k, v in self._stages.items()},
            }

    @classmethod
    def from_dict(cls, d: dict) -> "PipelineState":
        state            = cls(model_id=d.get("model_id", "dgb1"))
        state.run_id     = d.get("run_id", "")
        state.created_at = d.get("created_at", time.time())
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

    # ── Reporting ──────────────────────────────────────────────────────

    def summary(self) -> str:
        lines = [f"Pipeline state  run_id={self.run_id}  model={self.model_id}"]
        for rec in self._stages.values():
            dur = f"  {rec.duration_sec:.1f}s" if rec.duration_sec else ""
            err = f"  ERROR: {rec.error}" if rec.error else ""
            lines.append(f"  {rec.name:<26} [{rec.status.upper():<10}]{dur}{err}")
        return "\n".join(lines)
