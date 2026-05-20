"""
security/rate_limiter.py
==========================
Enterprise per-API-key rate limiting with sliding window and audit trail.
Thread-safe in-memory implementation — replace backing store with Redis
for multi-process / multi-server deployments.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from dataclasses import dataclass, field
from threading import RLock
from typing import Deque, Dict, Optional, Tuple

from modules.utils.error_handler import RateLimitError

logger = logging.getLogger(__name__)


@dataclass
class RateLimit:
    """Configuration for one rate limit tier."""
    limit:      int   # max requests
    window_sec: int   # sliding window duration


# Default tiers — override per key in _key_limits
_DEFAULT_TIER = RateLimit(limit=100, window_sec=60)
_ADMIN_TIER   = RateLimit(limit=1000, window_sec=60)
_FREE_TIER    = RateLimit(limit=20,   window_sec=60)


@dataclass
class RateLimitStats:
    key:         str
    limit:       int
    window_sec:  int
    consumed:    int
    remaining:   int
    reset_at:    float


class RateLimiter:
    """
    Thread-safe sliding-window rate limiter.

    Each key gets its own request timestamp deque.
    Old entries are purged on every check.
    """

    def __init__(self) -> None:
        self._lock:     RLock                         = RLock()
        self._buckets:  Dict[str, Deque[float]]       = defaultdict(deque)
        self._key_tiers: Dict[str, RateLimit]         = {}
        self._audit:    list                          = []
        self._max_audit = 10_000

    def set_tier(self, key: str, tier: RateLimit) -> None:
        """Assign a custom rate limit tier to a specific key."""
        with self._lock:
            self._key_tiers[key] = tier

    def check(self, key: str) -> RateLimitStats:
        """
        Check and record one request for `key`.
        Raises RateLimitError if limit exceeded.
        Returns RateLimitStats on success.
        """
        tier   = self._key_tiers.get(key, _DEFAULT_TIER)
        now    = time.time()
        cutoff = now - tier.window_sec

        with self._lock:
            bucket = self._buckets[key]
            # Evict expired entries
            while bucket and bucket[0] < cutoff:
                bucket.popleft()

            consumed   = len(bucket)
            remaining  = tier.limit - consumed
            reset_at   = (bucket[0] + tier.window_sec) if bucket else now + tier.window_sec

            if consumed >= tier.limit:
                self._audit_entry(key, "rejected", consumed, tier)
                raise RateLimitError(tier.limit, tier.window_sec)

            bucket.append(now)
            self._audit_entry(key, "allowed", consumed + 1, tier)

            return RateLimitStats(
                key=key,
                limit=tier.limit,
                window_sec=tier.window_sec,
                consumed=consumed + 1,
                remaining=remaining - 1,
                reset_at=reset_at,
            )

    def _audit_entry(self, key: str, outcome: str, consumed: int, tier: RateLimit) -> None:
        entry = {
            "ts":       time.time(),
            "key":      key,
            "outcome":  outcome,
            "consumed": consumed,
            "limit":    tier.limit,
        }
        self._audit.append(entry)
        if len(self._audit) > self._max_audit:
            self._audit = self._audit[-self._max_audit:]

    def stats(self, key: str) -> dict:
        tier   = self._key_tiers.get(key, _DEFAULT_TIER)
        now    = time.time()
        cutoff = now - tier.window_sec
        with self._lock:
            bucket = self._buckets.get(key, deque())
            consumed = sum(1 for t in bucket if t > cutoff)
        return {
            "key":       key,
            "consumed":  consumed,
            "remaining": max(0, tier.limit - consumed),
            "limit":     tier.limit,
            "window_sec": tier.window_sec,
        }

    def audit_tail(self, n: int = 100) -> list:
        with self._lock:
            return list(self._audit[-n:])


# ── Singleton ─────────────────────────────────────────────────────────────────
_limiter: Optional[RateLimiter] = None


def get_rate_limiter() -> RateLimiter:
    global _limiter
    if _limiter is None:
        _limiter = RateLimiter()
    return _limiter
