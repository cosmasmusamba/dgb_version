"""
modules/utils/progress_tracking.py
=====================================
Fine-grained progress tracking with ETA estimation, throughput measurement,
streaming event emission, and atomic offset persistence for exact resume.

ProgressTracker
    Tracks item-level progress, computes ETA, emits StreamEvents.
    Call `tracker.save(epoch, batch, step, path)` at checkpointing boundaries
    to persist offsets atomically so the next run can resume exactly.

ProgressCheckpoint
    Data class loaded by `ProgressTracker.load_checkpoint(path)`.
    Contains epoch, batch_idx, global_step, and line_offset so every
    level of granularity (line / batch / epoch) can be restored.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Checkpoint data class
# ---------------------------------------------------------------------------

@dataclass
class ProgressCheckpoint:
    """
    Persisted sub-stage offset for exact resume.

    Fields
    ------
    epoch:        Last fully completed epoch (0-indexed).
    batch_idx:    Last fully processed batch within the current epoch.
    global_step:  Absolute training step count.
    line_offset:  Lines processed in the current file (data-cleaning resume).
    timestamp:    Unix timestamp when this checkpoint was written.
    note:         Optional human-readable description.
    """
    epoch:       int   = 0
    batch_idx:   int   = 0
    global_step: int   = 0
    line_offset: int   = 0
    timestamp:   float = 0.0
    note:        str   = ""

    def to_dict(self) -> dict:
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ProgressCheckpoint":
        return cls(**{k: v for k, v in d.items() if k in cls.__dataclass_fields__})

    @classmethod
    def load(cls, path: Path) -> "ProgressCheckpoint":
        """Load from a JSON file written by ProgressTracker.save()."""
        import json
        path = Path(path)
        if not path.exists():
            return cls()
        try:
            return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))
        except Exception as exc:
            logger.warning("ProgressCheckpoint.load failed (%s): %s — starting fresh", path.name, exc)
            return cls()


# ---------------------------------------------------------------------------
# Progress state (live snapshot)
# ---------------------------------------------------------------------------

@dataclass
class ProgressState:
    """Snapshot of progress at a point in time."""
    current:      int
    total:        int
    elapsed_sec:  float
    eta_sec:      float
    throughput:   float     # items per second
    percent:      float


# ---------------------------------------------------------------------------
# ProgressTracker
# ---------------------------------------------------------------------------

class ProgressTracker:
    """
    Tracks item-level progress with ETA and throughput calculation.

    Uses exponential moving average of per-item time to smooth ETAs
    and avoid wild swings from variable-speed processing.

    Parameters
    ----------
    total:       Total number of items to process.
    label:       Human-readable label for logging.
    log_every:   Log a line every N items. 0 = never.
    ema_alpha:   EMA smoothing for ETA (0.1 = slow, 0.5 = fast).
    """

    def __init__(
        self,
        total:     int,
        label:     str   = "",
        log_every: int   = 10,
        ema_alpha: float = 0.2,
    ) -> None:
        self._total     = total
        self._label     = label
        self._log_every = log_every
        self._alpha     = ema_alpha
        self._current   = 0
        self._start     = time.perf_counter()
        self._last_time = self._start
        self._ema_item_sec: Optional[float] = None

    def update(self, n: int = 1) -> ProgressState:
        """
        Advance the tracker by `n` items.  Returns the current ProgressState.
        Emits a StreamEvent if the training hub is available.
        """
        self._current += n
        now            = time.perf_counter()
        elapsed        = now - self._start
        item_sec       = (now - self._last_time) / max(n, 1)
        self._last_time = now

        if self._ema_item_sec is None:
            self._ema_item_sec = item_sec
        else:
            self._ema_item_sec = (
                self._alpha * item_sec + (1 - self._alpha) * self._ema_item_sec
            )

        remaining = max(0, self._total - self._current)
        eta_sec   = remaining * self._ema_item_sec if self._ema_item_sec else 0.0
        throughput = self._current / elapsed if elapsed > 0 else 0.0
        percent    = 100.0 * self._current / max(self._total, 1)

        state = ProgressState(
            current=self._current,
            total=self._total,
            elapsed_sec=round(elapsed, 2),
            eta_sec=round(eta_sec, 1),
            throughput=round(throughput, 2),
            percent=round(percent, 1),
        )

        if self._log_every > 0 and self._current % self._log_every == 0:
            logger.info(
                "%s  %d/%d (%.1f%%)  %.0f/s  ETA %.0fs",
                self._label, self._current, self._total,
                percent, throughput, eta_sec,
            )

        try:
            from modules.utils.streaming import get_training_hub, StreamEvent
            get_training_hub().publish(StreamEvent.progress(
                label=self._label,
                current=self._current,
                total=self._total,
                percent=round(percent, 1),
                eta_sec=round(eta_sec, 1),
                throughput=round(throughput, 2),
            ))
        except Exception:
            pass

        return state

    def done(self) -> None:
        elapsed = time.perf_counter() - self._start
        logger.info(
            "%s  COMPLETE  %d/%d in %.1fs (%.1f/s)",
            self._label, self._current, self._total,
            elapsed, self._current / elapsed if elapsed > 0 else 0,
        )

    # ── Granular offset persistence ────────────────────────────────────

    def save(
        self,
        path:        Path,
        *,
        epoch:       int = 0,
        batch_idx:   int = 0,
        global_step: int = 0,
        line_offset: int = 0,
        note:        str = "",
    ) -> Path:
        """
        Atomically persist sub-stage offsets to `path`.

        Call at every checkpoint boundary (batch interval, epoch end, file
        boundary) so the next run can skip already-completed work.

        Parameters
        ----------
        path:         Destination JSON file.
        epoch:        Last fully completed epoch (0-indexed).
        batch_idx:    Last batch index within the current epoch.
        global_step:  Absolute step counter.
        line_offset:  Lines processed in the current source file.
        note:         Human-readable label for the checkpoint.

        Returns
        -------
        Path — the file written.
        """
        from modules.utils.safe_writer import atomic_write_json
        ckpt = ProgressCheckpoint(
            epoch=epoch,
            batch_idx=batch_idx,
            global_step=global_step,
            line_offset=line_offset,
            timestamp=time.time(),
            note=note,
        )
        path = Path(path)
        atomic_write_json(path, ckpt.to_dict())
        logger.debug(
            "ProgressTracker.save → %s  epoch=%d  batch=%d  step=%d  line=%d",
            path.name, epoch, batch_idx, global_step, line_offset,
        )
        return path

    @staticmethod
    def load_checkpoint(path: Path) -> ProgressCheckpoint:
        """
        Restore offsets from a previously saved checkpoint file.

        Returns a zeroed ProgressCheckpoint if the file does not exist or
        is corrupted (safe default: restart from the beginning).
        """
        return ProgressCheckpoint.load(path)

    @property
    def current(self) -> int:
        return self._current

    @property
    def percent(self) -> float:
        return 100.0 * self._current / max(self._total, 1)

    @property
    def is_done(self) -> bool:
        return self._current >= self._total
