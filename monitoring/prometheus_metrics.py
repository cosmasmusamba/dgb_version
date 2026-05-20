"""
monitoring/prometheus_metrics.py
==================================
Prometheus metrics for the DGB platform.

Metrics exposed at GET /metrics (Prometheus scrape endpoint).

Covers:
  - API request counts and latency histograms
  - Training step loss and learning rate gauges
  - GPU utilisation and VRAM usage
  - Token throughput counter
  - System RAM usage gauge
"""
from __future__ import annotations

import logging
import time
from typing import Optional

logger = logging.getLogger(__name__)

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, Summary,
        generate_latest, CONTENT_TYPE_LATEST,
        CollectorRegistry, REGISTRY,
    )
    _HAS_PROM = True
except ImportError:
    _HAS_PROM = False
    logger.info("prometheus_client not installed — metrics disabled")


# ── Metric definitions ────────────────────────────────────────────────────────
if _HAS_PROM:
    # API
    dgb_requests_total = Counter(
        "dgb_api_requests_total",
        "Total API requests",
        ["method", "endpoint", "status"],
    )
    dgb_request_latency = Histogram(
        "dgb_api_request_latency_seconds",
        "API request latency",
        ["endpoint"],
        buckets=[.005, .01, .025, .05, .1, .25, .5, 1, 2.5, 5, 10],
    )
    # Training
    dgb_train_loss = Gauge("dgb_train_loss", "Current training loss")
    dgb_val_loss   = Gauge("dgb_val_loss",   "Current validation loss")
    dgb_train_step = Gauge("dgb_train_step", "Global training step")
    dgb_train_lr   = Gauge("dgb_train_lr",   "Current learning rate")
    dgb_grad_norm  = Gauge("dgb_grad_norm",  "Gradient norm")
    dgb_epoch      = Gauge("dgb_epoch",      "Current epoch")
    dgb_tokens_generated = Counter(
        "dgb_tokens_generated_total",
        "Total tokens generated across all inference requests",
    )
    # System
    dgb_ram_used_gb  = Gauge("dgb_ram_used_gb",  "RAM used (GB)")
    dgb_gpu_util_pct = Gauge("dgb_gpu_util_pct", "GPU utilisation (%)")
    dgb_gpu_vram_gb  = Gauge("dgb_gpu_vram_gb",  "GPU VRAM used (GB)")


def record_request(method: str, endpoint: str, status: int, latency: float) -> None:
    if not _HAS_PROM:
        return
    dgb_requests_total.labels(method=method, endpoint=endpoint, status=str(status)).inc()
    dgb_request_latency.labels(endpoint=endpoint).observe(latency)


def record_train_step(
    step: int, loss: float, lr: float, grad_norm: float, epoch: int
) -> None:
    if not _HAS_PROM:
        return
    dgb_train_step.set(step)
    dgb_train_loss.set(loss)
    dgb_train_lr.set(lr)
    dgb_grad_norm.set(grad_norm)
    dgb_epoch.set(epoch)


def record_val_loss(val_loss: float) -> None:
    if not _HAS_PROM:
        return
    dgb_val_loss.set(val_loss)


def record_tokens(n: int) -> None:
    if not _HAS_PROM:
        return
    dgb_tokens_generated.inc(n)


def record_system(ram_gb: float, gpu_util: Optional[float], gpu_vram: Optional[float]) -> None:
    if not _HAS_PROM:
        return
    dgb_ram_used_gb.set(ram_gb)
    if gpu_util is not None:
        dgb_gpu_util_pct.set(gpu_util)
    if gpu_vram is not None:
        dgb_gpu_vram_gb.set(gpu_vram)


def get_metrics_response():
    """Return (content, media_type) for the /metrics FastAPI endpoint."""
    if not _HAS_PROM:
        return b"# prometheus_client not installed\n", "text/plain"
    return generate_latest(), CONTENT_TYPE_LATEST


# ── FastAPI route helper ──────────────────────────────────────────────────────

def add_metrics_route(app) -> None:
    """Attach GET /metrics to a FastAPI app."""
    from fastapi import Response

    @app.get("/metrics", include_in_schema=False)
    async def metrics():
        content, ctype = get_metrics_response()
        return Response(content=content, media_type=ctype)

    logger.info("Prometheus /metrics endpoint registered")


# ── Starlette request timing middleware ───────────────────────────────────────

class PrometheusMiddleware:
    """ASGI middleware that records latency and request counts."""

    def __init__(self, app) -> None:
        self._app = app

    async def __call__(self, scope, receive, send) -> None:
        if scope["type"] != "http":
            await self._app(scope, receive, send)
            return

        path   = scope.get("path", "")
        method = scope.get("method", "")
        t0     = time.perf_counter()
        status = 500

        async def _send_wrapper(message):
            nonlocal status
            if message["type"] == "http.response.start":
                status = message.get("status", 500)
            await send(message)

        await self._app(scope, receive, _send_wrapper)
        latency = time.perf_counter() - t0
        record_request(method, path, status, latency)
