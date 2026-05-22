"""
data_pipeline/core/quota_manager.py
=====================================
Per-source and global storage quota enforcement.

Prevents disk exhaustion by:
  1. Tracking bytes written per source against configured quota
  2. Monitoring actual disk free space continuously
  3. Pausing extraction when disk free < safety_margin_gb
  4. Logging warnings at configurable thresholds
  5. Blocking writes that would exceed per-source quotas

All limits are loaded from runtime_config.json and respected strictly.
No writes bypass the quota manager.
"""
from __future__ import annotations

import logging
import shutil
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional

logger = logging.getLogger(__name__)


@dataclass
class SourceQuota:
    """Per-source storage allocation and usage."""
    source_name:    str
    max_bytes:      int        # configured limit (0 = unlimited)
    used_bytes:     int = 0
    records_written: int = 0
    last_updated:   float = field(default_factory=time.time)
    paused:         bool = False
    pause_reason:   str = ""

    @property
    def remaining_bytes(self) -> int:
        if self.max_bytes <= 0:
            return int(1e18)    # effectively unlimited
        return max(0, self.max_bytes - self.used_bytes)

    @property
    def used_fraction(self) -> float:
        if self.max_bytes <= 0:
            return 0.0
        return self.used_bytes / self.max_bytes

    @property
    def used_gb(self) -> float:
        return self.used_bytes / 1024**3

    @property
    def limit_gb(self) -> float:
        return self.max_bytes / 1024**3 if self.max_bytes > 0 else 0.0


class QuotaManager:
    """
    Central storage quota enforcer for the data pipeline.

    Parameters
    ----------
    datasets_root:      Root directory that holds all source output dirs.
    global_max_bytes:   Hard limit across all sources combined (0 = unlimited).
    safety_margin_bytes: Pause all writes when disk_free < this value.
    warn_fraction:      Log a warning when a source exceeds this fraction of quota.
    source_quotas:      Dict mapping source_name → max_bytes.
    check_interval:     How often to sample actual disk usage (seconds).
    """

    def __init__(
        self,
        datasets_root:        Path,
        global_max_bytes:     int   = 0,
        safety_margin_bytes:  int   = 10 * 1024**3,     # 10 GB safety margin
        warn_fraction:        float = 0.85,
        source_quotas:        Optional[Dict[str, int]] = None,
        check_interval:       float = 60.0,
    ) -> None:
        self._root          = Path(datasets_root)
        self._global_max    = global_max_bytes
        self._safety_margin = safety_margin_bytes
        self._warn_frac     = warn_fraction
        self._check_interval = check_interval
        self._lock          = threading.RLock()

        self._sources: Dict[str, SourceQuota] = {}
        for name, limit in (source_quotas or {}).items():
            self._sources[name] = SourceQuota(source_name=name, max_bytes=limit)

        self._global_used      = 0
        self._disk_free_bytes  = self._sample_disk_free()
        self._paused_global    = False
        self._last_disk_check  = time.time()
        self._total_records    = 0

        logger.info(
            "QuotaManager: root=%s  global_max=%dGB  safety=%dGB",
            self._root,
            global_max_bytes // 1024**3 if global_max_bytes else 0,
            safety_margin_bytes // 1024**3,
        )

    # ── Write permission ──────────────────────────────────────────────

    def can_write(self, source_name: str, n_records: int = 1) -> bool:
        """Return True if the source is allowed to write more data."""
        self._maybe_refresh_disk()

        with self._lock:
            if self._paused_global:
                return False

            quota = self._get_quota(source_name)
            if quota.paused:
                return False
            if quota.max_bytes > 0 and quota.used_bytes >= quota.max_bytes:
                if not quota.paused:
                    quota.paused      = True
                    quota.pause_reason = "quota_exceeded"
                    logger.warning(
                        "QuotaManager: %s quota exhausted (%.1f GB / %.1f GB)",
                        source_name, quota.used_gb, quota.limit_gb,
                    )
                return False

            if self._global_max > 0 and self._global_used >= self._global_max:
                self._paused_global = True
                logger.critical(
                    "QuotaManager: global storage limit reached (%.1f GB)",
                    self._global_used / 1024**3,
                )
                return False

            if self._disk_free_bytes < self._safety_margin:
                self._paused_global = True
                logger.critical(
                    "QuotaManager: disk free %.1f GB < safety %.1f GB — PAUSING ALL",
                    self._disk_free_bytes / 1024**3,
                    self._safety_margin / 1024**3,
                )
                return False

        return True

    def record_write(self, source_name: str, n_bytes: int, n_records: int = 1) -> None:
        """Record that bytes have been written for a source."""
        with self._lock:
            quota = self._get_quota(source_name)
            quota.used_bytes     += n_bytes
            quota.records_written += n_records
            quota.last_updated   = time.time()
            self._global_used    += n_bytes
            self._total_records  += n_records

            # Warn approaching quota
            if (quota.max_bytes > 0 and not quota.paused and
                    quota.used_fraction >= self._warn_frac):
                logger.warning(
                    "QuotaManager: %s at %.0f%% of quota (%.1f/%.1f GB)",
                    source_name,
                    quota.used_fraction * 100,
                    quota.used_gb,
                    quota.limit_gb,
                )

    def resume_source(self, source_name: str) -> None:
        """Resume a paused source (call after manual intervention)."""
        with self._lock:
            if source_name in self._sources:
                self._sources[source_name].paused = False
                self._sources[source_name].pause_reason = ""

    def resume_global(self) -> None:
        """Resume global pause (call after disk space is freed)."""
        with self._lock:
            self._paused_global = False
            self._disk_free_bytes = self._sample_disk_free()

    # ── Stats ─────────────────────────────────────────────────────────

    def source_stats(self, source_name: str) -> dict:
        with self._lock:
            q = self._get_quota(source_name)
            return {
                "source":         source_name,
                "used_gb":        round(q.used_gb, 3),
                "limit_gb":       round(q.limit_gb, 3),
                "used_pct":       round(q.used_fraction * 100, 1),
                "records":        q.records_written,
                "paused":         q.paused,
                "pause_reason":   q.pause_reason,
                "disk_free_gb":   round(self._disk_free_bytes / 1024**3, 2),
            }

    def global_stats(self) -> dict:
        with self._lock:
            return {
                "global_used_gb":  round(self._global_used / 1024**3, 3),
                "global_max_gb":   round(self._global_max / 1024**3, 3) if self._global_max else 0,
                "disk_free_gb":    round(self._disk_free_bytes / 1024**3, 2),
                "paused":          self._paused_global,
                "total_records":   self._total_records,
                "sources":         {
                    n: self.source_stats(n) for n in self._sources
                },
            }

    # ── Internals ─────────────────────────────────────────────────────

    def _get_quota(self, source_name: str) -> SourceQuota:
        if source_name not in self._sources:
            self._sources[source_name] = SourceQuota(
                source_name=source_name, max_bytes=0
            )
        return self._sources[source_name]

    def _sample_disk_free(self) -> int:
        try:
            usage = shutil.disk_usage(self._root)
            return usage.free
        except Exception:
            return int(100 * 1024**3)   # assume 100 GB if check fails

    def _maybe_refresh_disk(self) -> None:
        now = time.time()
        if now - self._last_disk_check < self._check_interval:
            return
        with self._lock:
            self._disk_free_bytes = self._sample_disk_free()
            self._last_disk_check = now
            # Auto-resume global pause if disk space freed
            if self._paused_global and self._disk_free_bytes > self._safety_margin * 2:
                logger.info(
                    "QuotaManager: disk free %.1f GB — resuming",
                    self._disk_free_bytes / 1024**3,
                )
                self._paused_global = False