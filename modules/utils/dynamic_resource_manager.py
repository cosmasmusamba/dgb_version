"""
modules/utils/dynamic_resource_manager.py
==========================================
Dynamic resource adaptation during training.

Monitors available RAM every `interval` seconds (background thread) and
adjusts batch_size, chunk_size_chars, grad_accum, and num_workers.

All current settings are exposed via the `handle` property as a
ResourceHandle snapshot so callers never hold a stale reference.

Usage (pipeline / finetune / trainer)
--------------------------------------
    rm = DynamicResourceManager(
        interval=15,
        initial_batch=profile.recommended_batch_size,
        initial_chunk=profile.recommended_chunk_size_chars,
        initial_workers=profile.recommended_num_workers,
        gpu_device=profile.recommended_device,
    )
    rm.start()
    ...
    batch = rm.handle.batch_size
    chunk = rm.handle.chunk_size_chars
    rm.stop()

Legacy API (trainer.py / from_profile)
---------------------------------------
    rm = DynamicResourceManager.from_profile(profile, cfg)
    rm.check(step)           # called every GC_INTERVAL steps
    batch = rm.batch_size    # direct property access
"""
from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass
from typing import Optional

from configs.constants import GC_INTERVAL
from modules.utils.system_detector import SystemProfile

logger = logging.getLogger(__name__)

_MIN_BATCH       = 4
_MAX_ACCUM       = 64
_MIN_CHUNK_CHARS = 50_000


# ---------------------------------------------------------------------------
# ResourceHandle  — immutable snapshot consumed by training/finetune loops
# ---------------------------------------------------------------------------

@dataclass(frozen=True)
class ResourceHandle:
    """
    Immutable snapshot of current resource settings.

    Read once per epoch/batch boundary from `DynamicResourceManager.handle`.
    """
    batch_size:       int
    chunk_size_chars: int
    grad_accum:       int
    num_workers:      int
    pin_memory:       bool
    pressure:         str       # "normal" | "high" | "critical"
    ram_available_gb: float


# ---------------------------------------------------------------------------
# DynamicResourceManager
# ---------------------------------------------------------------------------

class DynamicResourceManager:
    """
    Adapts batch size, chunk size, and gradient accumulation during training.

    Parameters (new unified signature)
    -----------------------------------
    interval:        Background check interval in seconds (0 = disabled).
    initial_batch:   Starting batch size.
    initial_chunk:   Starting chunk size in chars for data cleaning/tokenizing.
    initial_accum:   Starting gradient accumulation steps.
    initial_workers: Starting DataLoader num_workers.
    initial_pin_mem: Whether DataLoader pins memory (CUDA only).
    gpu_device:      Device string, e.g. "cuda" or "cpu".
    max_batch:       Upper bound for batch size (never exceeded).
    min_batch:       Lower bound for batch size.
    ram_available_gb: Override starting RAM estimate (for tests).

    Legacy parameters (also accepted for backward compatibility)
    -----------------------------------------------------------
    pin_memory       → alias for initial_pin_mem
    """

    def __init__(
        self,
        interval:         int   = 0,
        initial_batch:    int   = 32,
        initial_chunk:    int   = 500_000,
        initial_accum:    int   = 1,
        initial_workers:  int   = 0,
        initial_pin_mem:  bool  = False,
        gpu_device:       str   = "cpu",
        max_batch:        int   = 512,
        min_batch:        int   = _MIN_BATCH,
        ram_available_gb: float = 8.0,
        # legacy keyword aliases
        pin_memory:       bool  = False,
    ) -> None:
        self._batch    = initial_batch
        self._chunk    = initial_chunk
        self._accum    = initial_accum
        self._workers  = initial_workers
        self._pin      = initial_pin_mem or pin_memory
        self._device   = gpu_device
        self._max_batch = max_batch
        self._min_batch = min(min_batch, initial_batch)
        self._ram      = ram_available_gb
        self._pressure = "normal"

        self._interval  = interval
        self._thread:   Optional[threading.Thread] = None
        self._stop_evt  = threading.Event()
        self._lock      = threading.Lock()

    # ── Background thread ──────────────────────────────────────────────

    def start(self) -> None:
        """Start the background monitoring thread (no-op if interval == 0)."""
        if self._interval <= 0 or self._thread is not None:
            return
        self._stop_evt.clear()
        self._thread = threading.Thread(
            target=self._monitor_loop, daemon=True, name="DRM-monitor"
        )
        self._thread.start()
        logger.debug("DynamicResourceManager started (interval=%ds)", self._interval)

    def stop(self) -> None:
        """Stop the background monitoring thread."""
        self._stop_evt.set()
        if self._thread is not None:
            self._thread.join(timeout=5)
            self._thread = None
        logger.debug("DynamicResourceManager stopped")

    def _monitor_loop(self) -> None:
        while not self._stop_evt.wait(timeout=self._interval):
            self._adapt()

    # ── Handle (snapshot API used by loops) ───────────────────────────

    @property
    def handle(self) -> ResourceHandle:
        """Return an immutable snapshot of current resource settings."""
        with self._lock:
            return ResourceHandle(
                batch_size=self._batch,
                chunk_size_chars=self._chunk,
                grad_accum=self._accum,
                num_workers=self._workers,
                pin_memory=self._pin,
                pressure=self._pressure,
                ram_available_gb=self._ram,
            )

    # ── Legacy direct-property API (used by trainer.py / check()) ─────

    @property
    def batch_size(self) -> int:
        return self._batch

    @property
    def chunk_size_chars(self) -> int:
        return self._chunk

    @property
    def grad_accum(self) -> int:
        return self._accum

    @property
    def num_workers(self) -> int:
        return self._workers

    @property
    def pin_memory(self) -> bool:
        return self._pin

    @property
    def pressure(self) -> str:
        return self._pressure

    @property
    def ram_available_gb(self) -> float:
        return self._ram

    @property
    def effective_batch(self) -> int:
        return self._batch * self._accum

    # ── Adaptation (called by background thread or check()) ───────────

    def _adapt(self) -> bool:
        """Sample RAM and adjust settings. Returns True if anything changed."""
        try:
            import psutil
            vm = psutil.virtual_memory()
            ram = vm.available / 1024 ** 3
        except Exception:
            return False

        with self._lock:
            self._ram = ram
            changed   = False

            if ram < 0.5:
                self._pressure = "critical"
                if self._batch > self._min_batch:
                    old_b        = self._batch
                    self._batch  = max(self._min_batch, self._batch // 2)
                    self._accum  = min(_MAX_ACCUM, self._accum * 2)
                    self._chunk  = max(_MIN_CHUNK_CHARS, self._chunk // 2)
                    logger.warning(
                        "DRM CRITICAL: RAM=%.2fGB  batch %d→%d  chunk→%d  accum→%d",
                        ram, old_b, self._batch, self._chunk, self._accum,
                    )
                    changed = True
            elif ram < 1.0:
                self._pressure = "high"
                if self._batch > self._min_batch * 2:
                    old_b        = self._batch
                    self._batch  = max(self._min_batch, self._batch - self._min_batch)
                    self._accum  = min(_MAX_ACCUM, self._accum + 1)
                    self._chunk  = max(_MIN_CHUNK_CHARS, int(self._chunk * 0.75))
                    logger.warning(
                        "DRM HIGH: RAM=%.2fGB  batch %d→%d  chunk→%d",
                        ram, old_b, self._batch, self._chunk,
                    )
                    changed = True
            else:
                self._pressure = "normal"

        return changed

    def check(self, step: int) -> bool:
        """
        Legacy step-based check — called from training loop every GC_INTERVAL
        steps.  Returns True if any parameter changed.
        """
        if step % GC_INTERVAL != 0:
            return False
        return self._adapt()

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    def from_profile(cls, profile: "SystemProfile", cfg) -> "DynamicResourceManager":
        """Build a DRM from system profile and training config."""
        workers = min(
            getattr(cfg.training, "num_workers", 0),
            profile.recommended_num_workers,
        )
        return cls(
            initial_batch=profile.recommended_batch_size,
            initial_chunk=getattr(profile, "recommended_chunk_size_chars", 500_000),
            initial_accum=1,
            initial_workers=workers,
            initial_pin_mem=profile.recommended_pin_memory,
            gpu_device=profile.recommended_device,
            ram_available_gb=profile.ram_available_gb,
        )

    def summary(self) -> str:
        h = self.handle
        return (
            f"DRM: batch={h.batch_size}  chunk={h.chunk_size_chars:,}  "
            f"accum={h.grad_accum}  effective={self.effective_batch}  "
            f"workers={h.num_workers}  pin={h.pin_memory}  "
            f"pressure={h.pressure}  RAM={h.ram_available_gb:.1f}GB"
        )

    def __enter__(self) -> "DynamicResourceManager":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()