"""
api_manager/middleware/rate_limiter.py
=========================================
Sliding-window rate limiter ASGI middleware.

Per-IP and per-API-key limits applied before auth.
Uses an in-memory deque per client — replace with Redis for multi-process.
"""
from __future__ import annotations

import hashlib
import logging
import os
import time
import uuid
from collections import defaultdict, deque
from typing import Callable, Deque, Dict, Optional, Tuple

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)

try:
    import redis  # type: ignore
except Exception:  # pragma: no cover - optional dependency
    redis = None


class SlidingWindowRateLimiter(BaseHTTPMiddleware):
    """
    Sliding-window rate limiter.

    Parameters
    ----------
    limit:       Max requests per window.
    window_sec:  Window duration in seconds.
    exempt_paths: Paths that bypass rate limiting (e.g. /health, /docs).
    """

    def __init__(
        self,
        app,
        limit:        int   = 100,
        window_sec:   int   = 60,
        exempt_paths: tuple = ("/health", "/docs", "/redoc", "/openapi.json"),
    ) -> None:
        super().__init__(app)
        self._limit   = limit
        self._window  = window_sec
        self._exempt  = set(exempt_paths)
        self._buckets: Dict[str, Deque[float]] = defaultdict(deque)
        self._redis_client = self._init_redis_client()

    def _init_redis_client(self):
        if redis is None:
            return None
        uri = os.getenv("DGB_REDIS_URL")
        if not uri:
            return None
        try:
            client = redis.Redis.from_url(uri, decode_responses=True)
            client.ping()
            logger.info("Redis rate limiter connected")
            return client
        except Exception as exc:
            logger.warning("Redis rate limiter unavailable: %s", exc)
            return None

    async def dispatch(self, request: Request, call_next: Callable):
        path = request.url.path
        if any(path.startswith(e) for e in self._exempt):
            return await call_next(request)

        client_key = self._get_key(request)
        if self._redis_client is not None:
            allowed, retry_after, remaining = self._check_redis_bucket(client_key)
            if not allowed:
                logger.warning("Rate limit exceeded: %s  path=%s", client_key, path)
                return JSONResponse(
                    status_code=429,
                    content={"detail": f"Rate limit {self._limit}/{self._window}s exceeded"},
                    headers={"Retry-After": str(retry_after)},
                )
            response = await call_next(request)
            response.headers["X-RateLimit-Limit"] = str(self._limit)
            response.headers["X-RateLimit-Remaining"] = str(remaining)
            response.headers["X-RateLimit-Reset"] = str(int(time.time() + self._window))
            return response

        now = time.time()
        bucket = self._buckets[client_key]
        while bucket and bucket[0] < now - self._window:
            bucket.popleft()

        if len(bucket) >= self._limit:
            retry_after = int(self._window - (now - bucket[0])) + 1
            logger.warning("Rate limit exceeded: %s  path=%s", client_key, path)
            return JSONResponse(
                status_code=429,
                content={"detail": f"Rate limit {self._limit}/{self._window}s exceeded"},
                headers={"Retry-After": str(retry_after)},
            )

        bucket.append(now)
        response = await call_next(request)
        response.headers["X-RateLimit-Limit"] = str(self._limit)
        response.headers["X-RateLimit-Remaining"] = str(self._limit - len(bucket))
        response.headers["X-RateLimit-Reset"] = str(int(now + self._window))
        return response

    def _check_redis_bucket(self, client_key: str) -> Tuple[bool, int, int]:
        if self._redis_client is None:
            return True, 0, self._limit

        key = f"dgb:rl:{client_key}"
        now = int(time.time())
        window_start = now - self._window

        try:
            current_count = self._redis_client.zcount(key, window_start, now)
            if current_count >= self._limit:
                oldest = self._redis_client.zrange(key, 0, 0, withscores=True)
                if oldest:
                    oldest_ts = float(oldest[0][1])
                    retry_after = max(1, int(self._window - (now - oldest_ts)) + 1)
                else:
                    retry_after = self._window
                return False, retry_after, max(0, self._limit - current_count)

            member = f"{now}:{uuid.uuid4().hex}"
            self._redis_client.zadd(key, {member: now})
            self._redis_client.zremrangebyscore(key, 0, window_start)
            remaining = max(0, self._limit - (current_count + 1))
            return True, 0, remaining
        except Exception as exc:
            logger.warning("Redis rate limiter check failed for %s: %s", client_key, exc)
            return True, 0, self._limit

    def _get_key(self, request: Request) -> str:
        """Prefer API key from Authorization header, fall back to client IP."""
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:].strip()
            token_hash = hashlib.sha256(token.encode("utf-8")).hexdigest()
            return f"token:{token_hash}"
        cf_ip = request.headers.get("CF-Connecting-IP", "")
        x_fwd = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP", "")
        return f"ip:{cf_ip or x_fwd or real_ip or request.client.host}"
