"""
modules/utils/metrics_logger.py
================================
Structured training metrics logging.

Writes two datetime-prefixed JSON files:
  <run_id>_metrics_steps.json   — one entry per optimizer step
  <run_id>_metrics_epochs.json  — one entry per epoch

Both files are kept in memory and flushed atomically so the training
dashboard can read them via the /admin/metrics/history endpoint with
zero risk of partial reads.

Also computes:
  - Exponential moving average of step losses
  - Perplexity from validation loss
  - Gradient norm statistics
  - Learning rate schedule trace
"""
from __future__ import annotations

import json
import logging
import math
import threading
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

from modules.utils.safe_writer import atomic_write_json

logger = logging.getLogger(__name__)


class MetricsLogger:
    """
    Records step-level and epoch-level training metrics to JSON files.

    All writes are thread-safe (training runs in a background thread
    while the API serves reads concurrently).

    Parameters
    ----------
    log_dir:   Directory for metric files.
    run_id:    RunContext run_id (datetime prefix).
    model_id:  e.g. "dgb1".
    """

    def __init__(
        self,
        log_dir:  Path,
        run_id:   str = "",
        model_id: str = "dgb1",
    ) -> None:
        self._dir      = Path(log_dir)
        self._run_id   = run_id
        self._model_id = model_id
        self._lock     = threading.RLock()

        prefix = f"{run_id}_" if run_id else ""
        self._steps_path  = self._dir / f"{prefix}metrics_steps.json"
        self._epochs_path = self._dir / f"{prefix}metrics_epochs.json"

        self._steps:  List[Dict[str, Any]] = []
        self._epochs: List[Dict[str, Any]] = []

        # EMA state
        self._ema_alpha  = 0.1
        self._ema_loss:  Optional[float] = None
        self._min_loss:  float = float("inf")
        self._max_gnorm: float = 0.0
        self._flush_every = 50   # steps between disk writes

        self._dir.mkdir(parents=True, exist_ok=True)
        logger.debug("MetricsLogger initialised — %s", self._dir)

    # ── Step metrics ──────────────────────────────────────────────────

    def log_step(
        self,
        epoch:     int,
        step:      int,
        loss:      float,
        lr:        float,
        grad_norm: float = 0.0,
        **extra:   Any,
    ) -> None:
        """Record one optimizer step.  Flushes to disk every N steps."""
        ema = self._update_ema(loss)
        gnorm_smoothed = self._update_gnorm(grad_norm)
        self._min_loss = min(self._min_loss, loss)

        record: Dict[str, Any] = {
            "epoch":     epoch,
            "step":      step,
            "loss":      round(loss, 6),
            "ema_loss":  round(ema, 6),
            "lr":        float(f"{lr:.2e}"),
            "grad_norm": round(grad_norm, 6),
            "timestamp": time.time(),
            **extra,
        }
        with self._lock:
            self._steps.append(record)
            if len(self._steps) % self._flush_every == 0:
                self._flush_steps()

        logger.debug("step=%d  loss=%.4f  ema=%.4f  lr=%.2e  grad=%.4f",
                     step, loss, ema, lr, grad_norm)

    def _update_ema(self, loss: float) -> float:
        if self._ema_loss is None:
            self._ema_loss = loss
        else:
            self._ema_loss = self._ema_alpha * loss + (1 - self._ema_alpha) * self._ema_loss
        return self._ema_loss

    def _update_gnorm(self, gnorm: float) -> float:
        self._max_gnorm = max(self._max_gnorm, gnorm)
        return gnorm

    # ── Epoch metrics ─────────────────────────────────────────────────

    def log_epoch(
        self,
        epoch:       int,
        avg_loss:    float,
        val_loss:    Optional[float] = None,
        duration_sec: float = 0.0,
        **extra:     Any,
    ) -> None:
        """Record end-of-epoch aggregates."""
        perplexity     = math.exp(min(avg_loss, 20))
        val_perplexity = math.exp(min(val_loss, 20)) if val_loss is not None else None

        record: Dict[str, Any] = {
            "epoch":          epoch,
            "avg_loss":       round(avg_loss, 6),
            "val_loss":       round(val_loss, 6) if val_loss is not None else None,
            "perplexity":     round(perplexity, 4),
            "val_perplexity": round(val_perplexity, 4) if val_perplexity else None,
            "min_step_loss":  round(self._min_loss, 6),
            "duration_sec":   round(duration_sec, 2),
            "timestamp":      time.time(),
            **extra,
        }
        with self._lock:
            self._epochs.append(record)
            self._flush_steps()
            atomic_write_json(self._epochs_path, self._epochs)

        logger.info(
            "EPOCH %d  avg_loss=%.4f  val_loss=%s  ppl=%.2f  %.1fs",
            epoch, avg_loss,
            f"{val_loss:.4f}" if val_loss is not None else "—",
            perplexity, duration_sec,
        )

    # ── Persistence ───────────────────────────────────────────────────

    def _flush_steps(self) -> None:
        try:
            atomic_write_json(self._steps_path, self._steps)
        except Exception as exc:
            logger.warning("Metrics flush failed: %s", exc)

    def flush(self) -> None:
        """Force immediate disk write of all buffered records."""
        with self._lock:
            self._flush_steps()
            atomic_write_json(self._epochs_path, self._epochs)

    def load(self) -> None:
        """
        Reload existing metric files from disk (for resume continuity).
        Extends in-memory history with whatever was already persisted.
        """
        for path, store in [(self._steps_path, self._steps), (self._epochs_path, self._epochs)]:
            candidates = sorted(path.parent.glob(f"*{path.name.split('_', 1)[-1]}"))
            for p in reversed(candidates):
                try:
                    data = json.loads(p.read_text(encoding="utf-8"))
                    if isinstance(data, list):
                        store.extend(data)
                        logger.info("Metrics loaded: %d records from %s", len(data), p.name)
                        break
                except Exception as exc:
                    logger.debug("Cannot load %s: %s", p.name, exc)

    # ── Query interface ───────────────────────────────────────────────

    def latest(self) -> Dict[str, Any]:
        """Return the most recent step and epoch records."""
        with self._lock:
            return {
                "last_step":  self._steps[-1]  if self._steps  else {},
                "last_epoch": self._epochs[-1] if self._epochs else {},
                "total_steps":  len(self._steps),
                "total_epochs": len(self._epochs),
                "ema_loss":   round(self._ema_loss, 4) if self._ema_loss else None,
                "min_loss":   round(self._min_loss, 4) if self._min_loss < float("inf") else None,
                "max_gnorm":  round(self._max_gnorm, 4),
            }

    def step_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._steps)

    def epoch_history(self) -> List[Dict[str, Any]]:
        with self._lock:
            return list(self._epochs)

    def loss_curve(self) -> Dict[str, List[float]]:
        """Return {epochs: [...], train: [...], val: [...]} for chart rendering."""
        with self._lock:
            return {
                "epochs": [r["epoch"]    for r in self._epochs],
                "train":  [r["avg_loss"] for r in self._epochs],
                "val":    [r["val_loss"] for r in self._epochs if r.get("val_loss") is not None],
            }
