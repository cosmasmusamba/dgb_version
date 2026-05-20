"""
modules/utils/unified_log.py
=============================
Unified structured event log — single JSONL stream per run.

Every subsystem (trainer, cleaner, tokenizer, API) writes to one
chronological JSONL file per run:
    <run_id>_pipeline.jsonl

Each line is a self-contained JSON object with type, stage, timestamp,
and payload — making the file both human-readable and machine-parseable
for dashboards, audits, and post-mortem analysis.

The writer is thread-safe (the training loop, resource monitor, and API
may all emit events concurrently) and uses atomic line appends.
"""
from __future__ import annotations

import json
import logging
import threading
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger(__name__)


class UnifiedLogWriter:
    """
    Thread-safe JSONL event log for one pipeline run.

    Every call to batch(), epoch(), pipeline(), device() appends one
    JSON line atomically to the configured file.

    The file is opened in append mode and an exclusive threading.Lock
    guards every write so concurrent callers never interleave output.
    """

    def __init__(
        self,
        path:     Path,
        run_id:   str = "",
        model_id: str = "dgb1",
    ) -> None:
        self._path     = Path(path)
        self._run_id   = run_id
        self._model_id = model_id
        self._lock     = threading.Lock()
        self._closed   = False
        self._path.parent.mkdir(parents=True, exist_ok=True)
        # Open in append mode — survives across restarts of a resumed run
        self._fh = self._path.open("a", encoding="utf-8", buffering=1)
        logger.debug("UnifiedLogWriter opened: %s", self._path.name)

    # ── Write helpers ─────────────────────────────────────────────────

    def _emit(self, type_: str, stage: str, level: str = "INFO", **payload: Any) -> None:
        if self._closed:
            return
        record = {
            "type":      type_,
            "stage":     stage,
            "level":     level,
            "run_id":    self._run_id,
            "model_id":  self._model_id,
            "timestamp": time.time(),
            **payload,
        }
        line = json.dumps(record, ensure_ascii=False, default=str) + "\n"
        with self._lock:
            try:
                self._fh.write(line)
            except Exception as exc:
                logger.warning("UnifiedLog write failed: %s", exc)

    # ── Domain-specific emitters ──────────────────────────────────────

    def batch(
        self,
        epoch: int, batch: int, step: int,
        loss: float, lr: float, grad_norm: float,
        ema_loss: float, batch_size: int, accum: int,
        pressure: str, ram_avail: float, **extra: Any,
    ) -> None:
        self._emit(
            "batch", stage="model_training",
            epoch=epoch, batch=batch, step=step,
            loss=round(loss, 6), lr=round(lr, 8),
            grad_norm=round(grad_norm, 6), ema_loss=round(ema_loss, 6),
            batch_size=batch_size, accum=accum,
            pressure=pressure, ram_avail=round(ram_avail, 2),
            **extra,
        )

    def epoch(
        self,
        epoch: int, avg_loss: float, best_loss: float,
        duration_sec: float, n_batches: int, ema_loss: float,
        batch_size: int, val_loss: float = None, **extra: Any,
    ) -> None:
        self._emit(
            "epoch", stage="model_training",
            epoch=epoch, avg_loss=round(avg_loss, 6),
            val_loss=round(val_loss, 6) if val_loss is not None else None,
            best_loss=round(best_loss, 6), duration_sec=round(duration_sec, 2),
            n_batches=n_batches, ema_loss=round(ema_loss, 6),
            batch_size=batch_size, **extra,
        )

    def pipeline(
        self,
        message: str,
        stage:   str = "pipeline",
        level:   str = "INFO",
        **extra: Any,
    ) -> None:
        self._emit("pipeline", stage=stage, level=level, message=message, **extra)

    def device(self, snapshot: Dict[str, Any], stage: str = "pipeline") -> None:
        self._emit("device", stage=stage, **snapshot)

    def tokenizer(self, message: str, **extra: Any) -> None:
        self._emit("tokenizer", stage="train_tokenizer", message=message, **extra)

    def dataset(self, message: str, **extra: Any) -> None:
        self._emit("dataset", stage="dataset_clean", message=message, **extra)

    def error(self, message: str, stage: str = "pipeline", **extra: Any) -> None:
        self._emit("error", stage=stage, level="ERROR", message=message, **extra)

    # ── Lifecycle ─────────────────────────────────────────────────────

    def close(self) -> None:
        if not self._closed:
            self._closed = True
            with self._lock:
                try:
                    self._fh.flush()
                    self._fh.close()
                except Exception:
                    pass
            logger.debug("UnifiedLogWriter closed: %s", self._path.name)

    def __del__(self) -> None:
        self.close()


# ── Singleton management ──────────────────────────────────────────────────────

_writer: Optional[UnifiedLogWriter] = None
_init_lock = threading.Lock()


def init_unified_log(
    path:     Path,
    run_id:   str = "",
    model_id: str = "dgb1",
) -> UnifiedLogWriter:
    global _writer
    with _init_lock:
        if _writer is not None:
            _writer.close()
        _writer = UnifiedLogWriter(path=path, run_id=run_id, model_id=model_id)
    return _writer


def get_unified_log() -> Optional[UnifiedLogWriter]:
    return _writer
