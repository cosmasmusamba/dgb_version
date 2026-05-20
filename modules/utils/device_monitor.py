"""
modules/utils/device_monitor.py
================================
Background device health monitoring with OS sleep prevention.

DeviceMonitor: background thread that logs CPU/GPU/RAM stats every
N seconds during training — feeds the admin health dashboard.

SleepGuard: prevents OS from sleeping during long training runs
(optional display-on lock for GPU workstations).
"""
from __future__ import annotations

import logging
import platform
import threading
import time
from dataclasses import dataclass, asdict
from typing import Any, Dict, Optional

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


@dataclass
class DeviceSnapshot:
    timestamp:      float
    cpu_percent:    float
    ram_used_gb:    float
    ram_total_gb:   float
    ram_percent:    float
    swap_used_gb:   float
    gpu_util:       Optional[float]
    gpu_vram_used:  Optional[float]
    gpu_vram_total: Optional[float]

    def summary(self) -> str:
        parts = [
            f"CPU: {self.cpu_percent:.0f}%",
            f"RAM: {self.ram_used_gb:.1f}/{self.ram_total_gb:.1f} GB ({self.ram_percent:.0f}%)",
        ]
        if self.gpu_util is not None:
            parts.append(f"GPU: {self.gpu_util:.0f}%  VRAM: {self.gpu_vram_used:.1f}/{self.gpu_vram_total:.1f} GB")
        return "  |  ".join(parts)

    def to_dict(self) -> Dict[str, Any]:
        return asdict(self)


class DeviceMonitor:
    """
    Background thread that periodically samples and logs device health.

    Parameters
    ----------
    interval:  Sampling interval in seconds.
    profile:   SystemProfile for GPU device string.
    """

    def __init__(self, interval: float = 30.0, profile=None) -> None:
        self._interval    = interval
        self._device      = getattr(profile, "recommended_device", "cpu") if profile else "cpu"
        self._stop        = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self._latest: Optional[DeviceSnapshot] = None
        self._lock        = threading.Lock()

    def start(self) -> "DeviceMonitor":
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._loop, daemon=True, name="dgb-device-monitor"
        )
        self._thread.start()
        logger.debug("DeviceMonitor started (interval=%.0fs)", self._interval)
        return self

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)
        logger.debug("DeviceMonitor stopped")

    def snapshot(self) -> DeviceSnapshot:
        """Return the latest snapshot, or a fresh one if monitor hasn't run yet."""
        with self._lock:
            if self._latest:
                return self._latest
        return self._sample()

    def _loop(self) -> None:
        while not self._stop.wait(timeout=self._interval):
            try:
                snap = self._sample()
                with self._lock:
                    self._latest = snap
                logger.debug("Device: %s", snap.summary())
                from modules.utils.streaming import get_training_hub, StreamEvent
                get_training_hub().publish(StreamEvent("device", snap.to_dict()))
            except Exception as exc:
                logger.debug("DeviceMonitor error: %s", exc)

    def _sample(self) -> DeviceSnapshot:
        cpu_pct = ram_used = ram_total = ram_pct = swap = 0.0
        if _HAS_PSUTIL:
            import psutil
            cpu_pct  = psutil.cpu_percent(interval=None)
            vm       = psutil.virtual_memory()
            ram_used  = vm.used    / 1024**3
            ram_total = vm.total   / 1024**3
            ram_pct   = vm.percent
            try:
                swap = psutil.swap_memory().used / 1024**3
            except Exception:
                pass

        gpu_util = gpu_used = gpu_total = None
        if _HAS_TORCH and "cuda" in self._device:
            try:
                import torch
                props    = torch.cuda.get_device_properties(0)
                gpu_total = props.total_memory / 1024**3
                gpu_used  = torch.cuda.memory_allocated(0) / 1024**3
                gpu_util  = gpu_used / gpu_total * 100 if gpu_total else 0.0
            except Exception:
                pass

        return DeviceSnapshot(
            timestamp=time.time(),
            cpu_percent=round(cpu_pct, 1),
            ram_used_gb=round(ram_used, 2),
            ram_total_gb=round(ram_total, 2),
            ram_percent=round(ram_pct, 1),
            swap_used_gb=round(swap, 2),
            gpu_util=round(gpu_util, 1) if gpu_util is not None else None,
            gpu_vram_used=round(gpu_used, 2) if gpu_used is not None else None,
            gpu_vram_total=round(gpu_total, 2) if gpu_total is not None else None,
        )


class SleepGuard:
    """
    Prevents the OS from sleeping or dimming the display during training.
    Uses platform-specific mechanisms (Windows SetThreadExecutionState,
    macOS caffeinate, Linux caffeine/xdg-screensaver).
    """

    def __init__(self, keep_display_on: bool = False) -> None:
        self._display = keep_display_on
        self._os      = platform.system()
        self._proc    = None
        self._enabled = False

    def enable(self) -> None:
        if self._enabled:
            return
        try:
            if self._os == "Windows":
                import ctypes
                ES_CONTINUOUS         = 0x80000000
                ES_SYSTEM_REQUIRED    = 0x00000001
                ES_DISPLAY_REQUIRED   = 0x00000002
                flags = ES_CONTINUOUS | ES_SYSTEM_REQUIRED
                if self._display:
                    flags |= ES_DISPLAY_REQUIRED
                ctypes.windll.kernel32.SetThreadExecutionState(flags)
                logger.debug("SleepGuard enabled (Windows)")

            elif self._os == "Darwin":
                import subprocess
                args = ["caffeinate", "-i"]
                if self._display:
                    args.append("-d")
                self._proc = subprocess.Popen(args)
                logger.debug("SleepGuard enabled (macOS caffeinate)")

            self._enabled = True
        except Exception as exc:
            logger.debug("SleepGuard.enable failed: %s", exc)

    def disable(self) -> None:
        if not self._enabled:
            return
        try:
            if self._os == "Windows":
                import ctypes
                ctypes.windll.kernel32.SetThreadExecutionState(0x80000000)
            elif self._os == "Darwin" and self._proc:
                self._proc.terminate()
                self._proc = None
            self._enabled = False
            logger.debug("SleepGuard disabled")
        except Exception as exc:
            logger.debug("SleepGuard.disable failed: %s", exc)

    def __enter__(self) -> "SleepGuard":
        self.enable()
        return self

    def __exit__(self, *_) -> None:
        self.disable()
