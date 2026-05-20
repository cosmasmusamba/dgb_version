"""
modules/utils/progress_tracking.py
=====================================
Fine-grained progress tracking with ETA estimation, throughput measurement,
and streaming event emission.

Designed for the data cleaning pipeline (file-level progress) and the
training loop (batch-level progress).  Emits StreamEvents so the admin
dashboard receives live progress without polling.
"""
from __future__ import annotations

import logging
import time
from dataclasses import dataclass
from typing import Optional

logger = logging.getLogger(__name__)


@dataclass
class ProgressState:
    """Snapshot of progress at a point in time."""
    current:      int
    total:        int
    elapsed_sec:  float
    eta_sec:      float
    throughput:   float     # items per second
    percent:      float


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

        # EMA smoothing
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

        # Emit streaming event (non-blocking — hub.publish is fire-and-forget)
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

    @property
    def percent(self) -> float:
        return 100.0 * self._current / max(self._total, 1)

    @property
    def is_done(self) -> bool:
        return self._current >= self._total
