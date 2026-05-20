"""
modules/utils/dynamic_resource_manager.py
==========================================
Dynamic resource adaptation during training.

Monitors available RAM every step and adjusts:
  - batch_size:  reduced under pressure, never below MIN_BATCH
  - grad_accum:  increased to maintain effective batch size when shrinking
  - num_workers: reduced on Windows or low-RAM systems

All adjustments are logged and emitted as StreamEvents.
"""
from __future__ import annotations

import logging
from typing import Optional

from configs.constants import GC_INTERVAL
from modules.utils.system_detector import SystemProfile

logger = logging.getLogger(__name__)

_MIN_BATCH = 4
_MAX_ACCUM = 64


class DynamicResourceManager:
    """
    Adapts batch size and gradient accumulation during training.

    Parameters
    ----------
    initial_batch:   Starting batch size.
    initial_accum:   Starting gradient accumulation steps.
    initial_workers: Starting DataLoader num_workers.
    pin_memory:      Whether DataLoader pins memory (CUDA only).
    """

    def __init__(
        self,
        initial_batch:   int   = 32,
        initial_accum:   int   = 1,
        initial_workers: int   = 0,
        pin_memory:      bool  = False,
        ram_available_gb: float = 8.0,
    ) -> None:
        self._batch   = initial_batch
        self._accum   = initial_accum
        self._workers = initial_workers
        self._pin     = pin_memory
        self._ram     = ram_available_gb
        self._pressure = "normal"

    # ── Accessors ──────────────────────────────────────────────────────

    @property
    def batch_size(self) -> int:
        return self._batch

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

    # ── Adaptation ────────────────────────────────────────────────────

    def check(self, step: int) -> bool:
        """
        Check memory and potentially adapt resources.
        Returns True if any parameter changed.
        Called from training loop every GC_INTERVAL steps.
        """
        if step % GC_INTERVAL != 0:
            return False
        try:
            import psutil
            vm = psutil.virtual_memory()
            self._ram = vm.available / 1024 ** 3
        except Exception:
            return False

        changed = False

        if self._ram < 0.5:
            self._pressure = "critical"
            if self._batch > _MIN_BATCH:
                old = self._batch
                self._batch  = max(_MIN_BATCH, self._batch // 2)
                self._accum  = min(_MAX_ACCUM, self._accum * 2)
                logger.warning(
                    "DRM CRITICAL: RAM %.2f GB — batch %d→%d  accum %d→%d",
                    self._ram, old, self._batch, self._accum // 2, self._accum,
                )
                changed = True
        elif self._ram < 1.0:
            self._pressure = "high"
            if self._batch > _MIN_BATCH * 2:
                old = self._batch
                self._batch  = max(_MIN_BATCH, self._batch - _MIN_BATCH)
                self._accum  = min(_MAX_ACCUM, self._accum + 1)
                logger.warning(
                    "DRM HIGH: RAM %.2f GB — batch %d→%d",
                    self._ram, old, self._batch,
                )
                changed = True
        else:
            self._pressure = "normal"

        return changed

    # ── Factory ────────────────────────────────────────────────────────

    @classmethod
    def from_profile(cls, profile: SystemProfile, cfg) -> "DynamicResourceManager":
        """Build a DRM from system profile and training config."""
        workers = min(cfg.training.num_workers, profile.recommended_num_workers)
        return cls(
            initial_batch=profile.recommended_batch_size,
            initial_accum=1,
            initial_workers=workers,
            pin_memory=profile.recommended_pin_memory,
            ram_available_gb=profile.ram_available_gb,
        )

    def summary(self) -> str:
        return (
            f"DRM: batch={self._batch}  accum={self._accum}  "
            f"effective={self.effective_batch}  "
            f"workers={self._workers}  pin={self._pin}  "
            f"pressure={self._pressure}  RAM={self._ram:.1f}GB"
        )
