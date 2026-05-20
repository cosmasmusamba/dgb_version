"""
modules/logging_config.py
==========================
Structured, async-safe logging configuration for the DGB platform.

Features
--------
- AsyncRotatingFileHandler: non-blocking file writes via a background thread queue
- Coloured console output with level-specific ANSI codes
- LogStage context manager: tags all log records within a pipeline stage
- configure_logging(): idempotent — safe to call multiple times
- Thread-local stage tracking: each thread can have an independent stage label
"""
from __future__ import annotations

import logging
import logging.handlers
import os
import queue
import sys
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Optional

from configs.constants import LOG_FORMAT, LOG_DATE_FORMAT, LOG_MAX_BYTES, LOG_BACKUP_COUNT

# ── ANSI colour codes ─────────────────────────────────────────────────────────
_RESET   = "\033[0m"
_BOLD    = "\033[1m"
_COLOURS = {
    logging.DEBUG:    "\033[36m",    # cyan
    logging.INFO:     "\033[32m",    # green
    logging.WARNING:  "\033[33m",    # yellow
    logging.ERROR:    "\033[31m",    # red
    logging.CRITICAL: "\033[35m",    # magenta
}

# ── Thread-local stage context ────────────────────────────────────────────────
_tls = threading.local()


def set_log_stage(stage: str) -> None:
    _tls.stage = stage


def get_log_stage() -> str:
    return getattr(_tls, "stage", "pipeline")


class StageFilter(logging.Filter):
    """Injects the current thread's stage into every log record."""
    def filter(self, record: logging.LogRecord) -> bool:
        record.stage = get_log_stage()
        return True


class ColouredFormatter(logging.Formatter):
    """Console formatter with ANSI colour coding per level."""
    def __init__(self, fmt: str = LOG_FORMAT, datefmt: str = LOG_DATE_FORMAT) -> None:
        super().__init__(fmt, datefmt)

    def format(self, record: logging.LogRecord) -> str:
        colour = _COLOURS.get(record.levelno, "")
        record.levelname = f"{colour}{_BOLD}{record.levelname:<8}{_RESET}"
        return super().format(record)


class AsyncRotatingFileHandler(logging.Handler):
    """
    Non-blocking rotating file handler.

    Log records are pushed onto an internal queue and consumed by a
    dedicated daemon thread — the calling thread never blocks on file I/O.
    """
    def __init__(
        self,
        path:          Path,
        max_bytes:     int = LOG_MAX_BYTES,
        backup_count:  int = LOG_BACKUP_COUNT,
        encoding:      str = "utf-8",
        queue_size:    int = 50_000,
    ) -> None:
        super().__init__()
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._queue: queue.Queue[Optional[logging.LogRecord]] = queue.Queue(maxsize=queue_size)
        self._handler = logging.handlers.RotatingFileHandler(
            str(path), maxBytes=max_bytes, backupCount=backup_count,
            encoding=encoding, delay=True,
        )
        self._thread = threading.Thread(
            target=self._worker, daemon=True, name="dgb-log-writer"
        )
        self._thread.start()

    def emit(self, record: logging.LogRecord) -> None:
        try:
            self._queue.put_nowait(record)
        except queue.Full:
            pass   # drop rather than block

    def _worker(self) -> None:
        while True:
            record = self._queue.get()
            if record is None:
                break
            try:
                self._handler.emit(record)
                self._handler.flush()
            except Exception:
                pass

    def close(self) -> None:
        try:
            self._queue.put_nowait(None)
            self._thread.join(timeout=5)
        except Exception:
            pass
        self._handler.close()
        super().close()

    def setFormatter(self, fmt: logging.Formatter) -> None:
        super().setFormatter(fmt)
        self._handler.setFormatter(fmt)

    def flush(self) -> None:
        self._queue.join() if hasattr(self._queue, "join") else None
        self._handler.flush()


# ── Public API ────────────────────────────────────────────────────────────────
_configured = False
_config_lock = threading.Lock()


def configure_logging(
    level:       str            = "INFO",
    log_file:    Optional[Path] = None,
    async_file:  bool           = True,
    force:       bool           = False,
) -> None:
    """
    Configure root logger once.  Subsequent calls are no-ops unless force=True.

    Parameters
    ----------
    level:      Log level string (DEBUG / INFO / WARNING / ERROR / CRITICAL).
    log_file:   Path for the rotating file handler.  None = console only.
    async_file: Use AsyncRotatingFileHandler (non-blocking).
    force:      Re-configure even if already configured.
    """
    global _configured
    with _config_lock:
        if _configured and not force:
            return

        numeric = getattr(logging, level.upper(), logging.INFO)
        root    = logging.getLogger()
        root.setLevel(numeric)

        # Remove any existing handlers added during prior imports
        for h in root.handlers[:]:
            try:
                h.close()
            except Exception:
                pass
        root.handlers.clear()

        stage_filter = StageFilter()

        # Console handler
        console = logging.StreamHandler(sys.stdout)
        console.setLevel(numeric)
        no_tty = not sys.stdout.isatty() or os.environ.get("NO_COLOR")
        if no_tty:
            console.setFormatter(logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT))
        else:
            console.setFormatter(ColouredFormatter())
        console.addFilter(stage_filter)
        root.addHandler(console)

        # File handler
        if log_file:
            log_file = Path(log_file)
            log_file.parent.mkdir(parents=True, exist_ok=True)
            fmt = logging.Formatter(LOG_FORMAT, LOG_DATE_FORMAT)
            if async_file:
                fh = AsyncRotatingFileHandler(log_file)
            else:
                fh = logging.handlers.RotatingFileHandler(
                    str(log_file), maxBytes=LOG_MAX_BYTES,
                    backupCount=LOG_BACKUP_COUNT, encoding="utf-8",
                )
            fh.setLevel(numeric)
            fh.setFormatter(fmt)
            fh.addFilter(stage_filter)
            root.addHandler(fh)

        # Silence noisy third-party loggers
        for noisy in ("uvicorn.access", "asyncio", "multipart"):
            logging.getLogger(noisy).setLevel(logging.WARNING)

        _configured = True
        logging.getLogger(__name__).info(
            "Logging configured: level=%s  file=%s  async=%s",
            level, log_file or "none", async_file,
        )


class LogStage:
    """
    Context manager that sets the thread-local log stage for the duration
    of a pipeline stage, then restores the previous value.

    Usage::

        with LogStage("dataset_clean"):
            ...   # all log records carry stage="dataset_clean"
    """
    def __init__(self, stage: str) -> None:
        self._stage = stage
        self._prev: Optional[str] = None

    def __enter__(self) -> "LogStage":
        self._prev = get_log_stage()
        set_log_stage(self._stage)
        return self

    def __exit__(self, *_) -> None:
        set_log_stage(self._prev or "pipeline")


class TrainingLogger:
    """Convenience wrapper for emitting structured training log messages."""

    def __init__(self, name: str = "dgb.trainer") -> None:
        self._log = logging.getLogger(name)

    def epoch_start(self, epoch: int, total: int, lr: float, batch_size: int, accum: int) -> None:
        self._log.info(
            "EPOCH %d/%d START  lr=%.2e  batch=%d  accum=%d",
            epoch, total, lr, batch_size, accum,
        )

    def epoch_end(self, epoch: int, total: int, avg_loss: float, best_loss: float, duration: float) -> None:
        self._log.info(
            "EPOCH %d/%d END  avg_loss=%.4f  best=%.4f  %.1fs",
            epoch, total, avg_loss, best_loss, duration,
        )

    def step(self, step: int, loss: float, lr: float, grad_norm: float, ema: float) -> None:
        self._log.info(
            "step=%d  loss=%.4f  ema=%.4f  lr=%.2e  grad=%.4f",
            step, loss, ema, lr, grad_norm,
        )

    def checkpoint(self, path: Path, epoch: int, loss: float) -> None:
        self._log.info("CKPT saved → %s  epoch=%d  loss=%.4f", path.name, epoch, loss)

    def eval(self, epoch: int, val_loss: float, perplexity: float) -> None:
        self._log.info(
            "EVAL epoch=%d  val_loss=%.4f  perplexity=%.2f",
            epoch, val_loss, perplexity,
        )

    def resource(self, batch: int, accum: int, pressure: str, ram_gb: float) -> None:
        self._log.debug(
            "RES  batch=%d  accum=%d  pressure=%s  ram=%.1fGB",
            batch, accum, pressure, ram_gb,
        )

    def warning(self, msg: str, *args) -> None:
        self._log.warning(msg, *args)

    def error(self, msg: str, *args) -> None:
        self._log.error(msg, *args)
