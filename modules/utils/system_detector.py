"""
modules/utils/system_detector.py
==================================
Hardware capability detection and resource recommendation engine.

Produces a SystemProfile at pipeline startup that drives:
  - Initial batch size (DynamicResourceManager baseline)
  - Number of DataLoader workers
  - Device selection (CUDA > MPS > CPU)
  - Mixed precision eligibility
  - Recommended chunk size for data cleaning

Detection is non-blocking — all calls use cached psutil values.
"""
from __future__ import annotations

import logging
import os
import platform
from dataclasses import dataclass
from functools import lru_cache
from typing import Optional

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
class SystemProfile:
    """Hardware capabilities and recommended initial runtime parameters."""
    os_name:             str
    cpu_name:            str
    cpu_cores_logical:   int
    cpu_cores_physical:  int
    ram_total_gb:        float
    ram_available_gb:    float
    gpu_name:            Optional[str]
    gpu_vram_gb:         float
    has_cuda:            bool
    has_mps:             bool
    is_windows:          bool

    recommended_device:      str
    recommended_batch_size:  int
    recommended_num_workers: int
    recommended_chunk_size_chars: int
    recommended_pin_memory:  bool
    mixed_precision_eligible: bool

    def summary(self) -> str:
        lines = [
            f"OS:         {self.os_name}",
            f"CPU:        {self.cpu_name} ({self.cpu_cores_physical}p/{self.cpu_cores_logical}l cores)",
            f"RAM:        {self.ram_total_gb:.1f} GB total  {self.ram_available_gb:.1f} GB available",
            f"GPU:        {self.gpu_name or 'none'}" + (f" ({self.gpu_vram_gb:.1f} GB VRAM)" if self.gpu_name else ""),
            f"Device:     {self.recommended_device}",
            f"Batch:      {self.recommended_batch_size}",
            f"Workers:    {self.recommended_num_workers}",
            f"Chunk:      {self.recommended_chunk_size_chars:,} chars",
            f"FP16:       {'yes' if self.mixed_precision_eligible else 'no'}",
        ]
        return "\n".join(lines)


@lru_cache(maxsize=1)
def get_system_profile() -> SystemProfile:
    """
    Detect hardware capabilities and produce a SystemProfile.
    Cached — called once at pipeline startup; subsequent calls are free.
    """
    os_name = platform.system()
    is_win  = os_name == "Windows"

    # CPU
    cpu_name    = platform.processor() or "unknown"
    logical     = os.cpu_count() or 1
    physical    = logical
    if _HAS_PSUTIL:
        physical = psutil.cpu_count(logical=False) or logical

    # RAM
    ram_total_gb = ram_avail_gb = 4.0
    if _HAS_PSUTIL:
        vm = psutil.virtual_memory()
        ram_total_gb = vm.total    / 1024**3
        ram_avail_gb = vm.available / 1024**3

    # GPU
    gpu_name    = None
    gpu_vram_gb = 0.0
    has_cuda    = False
    has_mps     = False

    if _HAS_TORCH:
        has_cuda = torch.cuda.is_available()
        if has_cuda:
            try:
                gpu_name    = torch.cuda.get_device_name(0)
                gpu_vram_gb = torch.cuda.get_device_properties(0).total_memory / 1024**3
            except Exception:
                gpu_name = "CUDA device"
        has_mps = (
            hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
        )

    # Device recommendation
    if has_cuda:
        device = "cuda"
    elif has_mps:
        device = "mps"
    else:
        device = "cpu"

    # Batch size — conservative: 60% of available RAM, ~10 tokens/sample avg
    # GPU: use VRAM budget; CPU: use RAM budget
    if has_cuda and gpu_vram_gb > 0:
        budget_gb = gpu_vram_gb * 0.65
    else:
        budget_gb = ram_avail_gb * 0.40

    # Rough heuristic: 1 sample at seq_len=512 ≈ 1 MB
    batch = max(4, min(128, int(budget_gb * 1024 / 512)))
    batch = _round_to_power_of_2_or_nearby(batch)

    # Workers — 0 on Windows (spawn overhead), else cpu_count/2
    if is_win:
        workers = 0
    else:
        workers = min(4, max(0, physical // 2 - 1))

    # Chunk size for data cleaning — scale with available RAM
    chunk = int(ram_avail_gb * 0.30 * 1e6)
    chunk = max(100_000, min(5_000_000, chunk))

    # Pin memory — only with CUDA and adequate RAM
    pin_mem = has_cuda and ram_avail_gb > 4.0

    # Mixed precision — only with CUDA and fp16-capable GPU
    mp_eligible = False
    if has_cuda and _HAS_TORCH:
        try:
            cc = torch.cuda.get_device_capability(0)
            mp_eligible = cc[0] >= 7   # Volta+ supports fp16 efficiently
        except Exception:
            pass

    profile = SystemProfile(
        os_name=os_name,
        cpu_name=cpu_name,
        cpu_cores_logical=logical,
        cpu_cores_physical=physical,
        ram_total_gb=round(ram_total_gb, 2),
        ram_available_gb=round(ram_avail_gb, 2),
        gpu_name=gpu_name,
        gpu_vram_gb=round(gpu_vram_gb, 2),
        has_cuda=has_cuda,
        has_mps=has_mps,
        is_windows=is_win,
        recommended_device=device,
        recommended_batch_size=batch,
        recommended_num_workers=workers,
        recommended_chunk_size_chars=chunk,
        recommended_pin_memory=pin_mem,
        mixed_precision_eligible=mp_eligible,
    )
    logger.info("SystemProfile:\n%s", profile.summary())
    return profile


def _round_to_power_of_2_or_nearby(n: int) -> int:
    """Round n to the nearest power of 2 or a nearby practical batch size."""
    for candidate in [4, 8, 16, 24, 32, 48, 64, 96, 128]:
        if n <= candidate:
            return candidate
    return 128
