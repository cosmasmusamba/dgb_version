"""
modules/utils/memory_manager.py
================================
Runtime memory monitoring and automatic garbage collection.

Runs lightweight checks on every GC_INTERVAL steps during training.
If memory drops below the warning threshold, triggers Python GC and
(if CUDA) empties the cache.  If memory drops below the error threshold,
raises OutOfMemoryError to halt training cleanly before OOM kills the process.
"""
from __future__ import annotations

import gc
import logging
import os
from typing import Optional

from configs.constants import MEM_WARN_GB, MEM_ERROR_GB, GC_INTERVAL
from modules.utils.error_handler import OutOfMemoryError

logger = logging.getLogger(__name__)

try:
    import psutil
    _HAS_PSUTIL = True
except ImportError:
    _HAS_PSUTIL = False

try:
    import torch
    _HAS_TORCH = True
except ImportError:
    _HAS_TORCH = False


class MemoryManager:
    """
    Periodic memory health checks during training.

    Call `check(global_step)` after every batch. The manager throttles
    expensive checks to every GC_INTERVAL steps.

    Parameters
    ----------
    device:         Torch device string for GPU cache clearing.
    warn_gb:        RAM-available threshold for GC trigger (default 1.0 GB).
    error_gb:       RAM-available threshold for OOM error (default 0.25 GB).
    gc_interval:    Steps between checks (default from constants).
    """

    def __init__(
        self,
        device:      str   = "cpu",
        warn_gb:     float = MEM_WARN_GB,
        error_gb:    float = MEM_ERROR_GB,
        gc_interval: int   = GC_INTERVAL,
    ) -> None:
        self._device      = device
        self._warn_gb     = warn_gb
        self._error_gb    = error_gb
        self._gc_interval = gc_interval
        self._last_avail: Optional[float] = None

    def check(self, step: int) -> None:
        """
        Run a memory check at this step.
        No-ops unless step is a multiple of gc_interval.
        """
        if step % self._gc_interval != 0:
            return
        if not _HAS_PSUTIL:
            return

        import psutil
        vm       = psutil.virtual_memory()
        avail_gb = vm.available / 1024**3
        self._last_avail = avail_gb

        if avail_gb < self._error_gb:
            self._emergency_free()
            msg = (
                f"CRITICAL: Available RAM {avail_gb:.2f} GB < "
                f"error threshold {self._error_gb:.2f} GB — halting training"
            )
            logger.critical(msg)
            raise OutOfMemoryError(msg)

        if avail_gb < self._warn_gb:
            logger.warning(
                "Low RAM: %.2f GB available — triggering GC (step %d)",
                avail_gb, step,
            )
            self._free_memory()

    def _free_memory(self) -> None:
        """Trigger Python garbage collection and CUDA cache clearing."""
        gc.collect()
        if _HAS_TORCH and "cuda" in self._device:
            import torch
            try:
                torch.cuda.empty_cache()
                torch.cuda.synchronize()
            except Exception:
                pass
        logger.debug("Memory freed (GC + CUDA cache)")

    def _emergency_free(self) -> None:
        """Aggressive memory recovery before raising OOM error."""
        gc.collect()
        gc.collect()   # second pass catches cyclic garbage missed by first
        if _HAS_TORCH:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.ipc_collect()

    @property
    def available_gb(self) -> float:
        if not _HAS_PSUTIL:
            return -1.0
        import psutil
        return psutil.virtual_memory().available / 1024**3

    def summary(self) -> str:
        if not _HAS_PSUTIL:
            return "memory info unavailable (psutil not installed)"
        import psutil
        vm = psutil.virtual_memory()
        lines = [
            f"RAM: {vm.used/1024**3:.1f}/{vm.total/1024**3:.1f} GB ({vm.percent:.0f}% used)",
            f"Available: {vm.available/1024**3:.2f} GB",
        ]
        if _HAS_TORCH and torch.cuda.is_available():
            import torch
            alloc = torch.cuda.memory_allocated() / 1024**3
            reserved = torch.cuda.memory_reserved() / 1024**3
            lines.append(f"CUDA: {alloc:.2f} GB allocated / {reserved:.2f} GB reserved")
        return " | ".join(lines)
