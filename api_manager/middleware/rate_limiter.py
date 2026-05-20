"""
api_manager/middleware/rate_limiter.py
=========================================
Sliding-window rate limiter ASGI middleware.

Per-IP and per-API-key limits applied before auth.
Uses an in-memory deque per client — replace with Redis for multi-process.
"""
from __future__ import annotations

import logging
import time
from collections import defaultdict, deque
from typing import Callable, Deque, Dict

from starlette.middleware.base import BaseHTTPMiddleware
from starlette.requests import Request
from starlette.responses import JSONResponse

logger = logging.getLogger(__name__)


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

    async def dispatch(self, request: Request, call_next: Callable):
        path = request.url.path
        if any(path.startswith(e) for e in self._exempt):
            return await call_next(request)

        client_key = self._get_key(request)
        now        = time.time()
        bucket     = self._buckets[client_key]

        # Evict timestamps outside the window
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
        response.headers["X-RateLimit-Limit"]     = str(self._limit)
        response.headers["X-RateLimit-Remaining"] = str(self._limit - len(bucket))
        response.headers["X-RateLimit-Reset"]     = str(int(now + self._window))
        return response

    def _get_key(self, request: Request) -> str:
        """Prefer API key from Authorization header, fall back to client IP."""
        auth = request.headers.get("Authorization", "")
        if auth.startswith("Bearer "):
            token = auth[7:20]   # first 13 chars of token — not the full token
            return f"token:{token}"
        cf_ip   = request.headers.get("CF-Connecting-IP", "")
        x_fwd   = request.headers.get("X-Forwarded-For", "").split(",")[0].strip()
        real_ip = request.headers.get("X-Real-IP", "")
        return f"ip:{cf_ip or x_fwd or real_ip or request.client.host}"
