"""
modules/utils/unified_log.py
=============================
Single structured JSONL event stream for every DGB pipeline stage.

Every subsystem (trainer, cleaner, tokenizer, finetune, inference, API)
writes to ONE chronological JSONL file per run:
    <run_id>_pipeline.jsonl

Each line is a self-contained JSON record — machine-parseable for
dashboards and human-readable for post-mortems.

───────────────────────────────────────────────────────────────────
UNIFIED LOG SCHEMA  (every record carries these top-level fields)
───────────────────────────────────────────────────────────────────
{
  "type":      <string>   # "batch" | "epoch" | "pipeline" | "device"
                          # "tokenizer" | "dataset" | "finetune"
                          # "inference" | "error"
  "stage":     <string>   # which pipeline stage emitted this record
  "level":     <string>   # "DEBUG" | "INFO" | "WARNING" | "ERROR"
  "run_id":    <string>   # YYYYMMDDHHMMSS timestamp of this run
  "model_id":  <string>   # e.g. "dgb1"
  "timestamp": <float>    # Unix epoch seconds (UTC)
  ... payload fields vary by type (see emitter docstrings below)
}

Type-specific payload fields:
  batch      epoch, batch, step, loss, lr, grad_norm, ema_loss,
             batch_size, accum, pressure, ram_avail
  epoch      epoch, avg_loss, val_loss, best_loss, duration_sec,
             n_batches, ema_loss, batch_size
  pipeline   message
  device     cpu_percent, ram_used_gb, ram_total_gb, gpu_util, …
  tokenizer  message  [+ arbitrary extras]
  dataset    message  [+ arbitrary extras]
  finetune   message, epoch, step, loss  [+ arbitrary extras]
  inference  prompt_tokens, output_tokens, latency_ms, method
  error      message  [+ arbitrary extras]
───────────────────────────────────────────────────────────────────

All writes are thread-safe and non-blocking (threading.Lock + append mode).
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

    Every call to batch(), epoch(), finetune(), inference(), etc.
    appends one JSON line atomically to the configured file.

    The file is opened in append mode — it survives across restarts of
    a resumed run and accumulates events chronologically.
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
        self._fh = self._path.open("a", encoding="utf-8", buffering=1)
        logger.debug("UnifiedLogWriter opened: %s", self._path.name)

    # ── Core emit ─────────────────────────────────────────────────────

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

    # ── Training emitters ─────────────────────────────────────────────

    def batch(
        self,
        epoch: int, batch: int, step: int,
        loss: float, lr: float, grad_norm: float,
        ema_loss: float, batch_size: int, accum: int,
        pressure: str, ram_avail: float, **extra: Any,
    ) -> None:
        """Emit one training batch record."""
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
        batch_size: int, val_loss: Optional[float] = None, **extra: Any,
    ) -> None:
        """Emit one training epoch record."""
        self._emit(
            "epoch", stage="model_training",
            epoch=epoch, avg_loss=round(avg_loss, 6),
            val_loss=round(val_loss, 6) if val_loss is not None else None,
            best_loss=round(best_loss, 6), duration_sec=round(duration_sec, 2),
            n_batches=n_batches, ema_loss=round(ema_loss, 6),
            batch_size=batch_size, **extra,
        )

    # ── Finetune emitters ─────────────────────────────────────────────

    def finetune(
        self,
        message:  str,
        *,
        epoch:    int   = 0,
        step:     int   = 0,
        loss:     Optional[float] = None,
        level:    str   = "INFO",
        **extra:  Any,
    ) -> None:
        """
        Emit a finetune-stage record.

        Covers: loop start/end, epoch start/end, batch events,
        checkpoint autosaves, and errors — all under stage="finetune".
        """
        payload: Dict[str, Any] = {"message": message, "epoch": epoch, "step": step}
        if loss is not None:
            payload["loss"] = round(loss, 6)
        payload.update(extra)
        self._emit("finetune", stage="finetune", level=level, **payload)

    # ── Inference emitters ────────────────────────────────────────────

    def inference(
        self,
        prompt_tokens:  int,
        output_tokens:  int,
        latency_ms:     float,
        method:         str = "greedy",
        **extra:        Any,
    ) -> None:
        """
        Emit one inference request record.

        Fields
        ------
        prompt_tokens:  Token count of the input prompt.
        output_tokens:  Token count of generated output.
        latency_ms:     Wall-clock time for the generation call.
        method:         Decoding method ("greedy", "beam", "top_p").
        """
        self._emit(
            "inference", stage="inference",
            prompt_tokens=prompt_tokens,
            output_tokens=output_tokens,
            latency_ms=round(latency_ms, 2),
            method=method,
            **extra,
        )

    # ── General pipeline emitters ─────────────────────────────────────

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

    def __enter__(self) -> "UnifiedLogWriter":
        return self

    def __exit__(self, *_) -> None:
        self.close()


# ── Singleton management ──────────────────────────────────────────────────────

_writer: Optional[UnifiedLogWriter] = None
_init_lock = threading.Lock()


def init_unified_log(
    path:     Path,
    run_id:   str = "",
    model_id: str = "dgb1",
) -> UnifiedLogWriter:
    """
    Initialise (or replace) the process-wide UnifiedLogWriter.

    Call once at pipeline startup, before any stage runs.
    All subsequent calls to get_unified_log() return the same writer.
    """
    global _writer
    with _init_lock:
        if _writer is not None:
            _writer.close()
        _writer = UnifiedLogWriter(path=path, run_id=run_id, model_id=model_id)
    return _writer


def get_unified_log() -> Optional[UnifiedLogWriter]:
    """Return the active UnifiedLogWriter, or None if not yet initialised."""
    return _writer
