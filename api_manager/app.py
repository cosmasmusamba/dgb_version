"""
api_manager/app.py
====================
FastAPI application factory — wires together all routes, middleware,
CORS, lifespan (startup/shutdown), exception handlers, and static UI.
"""
from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from fastapi.staticfiles import StaticFiles

from configs.loader import get_config
from modules.logging_config import configure_logging, LogStage
from modules.utils.error_handler import DGBError
from api_manager.middleware.rate_limiter import SlidingWindowRateLimiter

logger = logging.getLogger(__name__)
_start_time = time.time()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan — startup → yield → shutdown."""
    cfg = get_config()
    configure_logging(level=cfg.logging.level, log_file=cfg.training_log())

    logger.info("=" * 60)
    logger.info("DGB Platform v%s starting", cfg.project.version)
    logger.info("Model: %s  Device: %s", cfg.project.model_id, cfg.training.device)
    logger.info("API: http://%s:%d", cfg.api.host, cfg.api.port)
    logger.info("=" * 60)

    # Start log broadcast server
    try:
        from api_manager.log_streamer import LogBroadcastServer
        _server = LogBroadcastServer(port=cfg.streaming.log_server_port)
        _server.start()
    except Exception as exc:
        logger.warning("Log broadcast server failed to start: %s", exc)

    # Pre-warm inference engine if model checkpoint exists
    try:
        from modules.utils.path_resolver import init_path_resolver
        res = init_path_resolver(cfg.project.model_id, cfg)
        if any(res.models_dir(create=False).glob("*.pt")):
            from api_manager.routes.inference import _get_engine
            _get_engine()
            logger.info("Inference engine pre-warmed")
    except Exception as exc:
        logger.info("Inference engine not pre-warmed: %s", exc)

    yield

    logger.info("DGB Platform shutting down")


def create_app() -> FastAPI:
    cfg = get_config()

    app = FastAPI(
        title=f"{cfg.project.name} AI Platform",
        version=cfg.project.version,
        description="DGB — fully custom AI framework built from scratch",
        docs_url=cfg.api.docs_url,
        redoc_url=cfg.api.redoc_url,
        lifespan=lifespan,
    )

    # ── CORS ──────────────────────────────────────────────────────────
    app.add_middleware(
        CORSMiddleware,
        allow_origins=cfg.api.cors_origins or ["*"],
        allow_credentials=True,
        allow_methods=cfg.api.cors_methods,
        allow_headers=cfg.api.cors_headers,
    )

    # ── Rate limiting ─────────────────────────────────────────────────
    app.add_middleware(
        SlidingWindowRateLimiter,
        limit=200,
        window_sec=60,
    )

    # ── Exception handlers ────────────────────────────────────────────
    @app.exception_handler(DGBError)
    async def dgb_error_handler(request: Request, exc: DGBError) -> JSONResponse:
        logger.error("DGBError [%s]: %s  path=%s", exc.error_code, exc.message, request.url.path)
        return JSONResponse(
            status_code=exc.http_status,
            content=exc.to_dict(),
        )

    @app.exception_handler(Exception)
    async def generic_error_handler(request: Request, exc: Exception) -> JSONResponse:
        logger.exception("Unhandled error on %s: %s", request.url.path, exc)
        return JSONResponse(
            status_code=500,
            content={"error_code": "INTERNAL_ERROR", "message": "Internal server error"},
        )

    # ── Routes ────────────────────────────────────────────────────────
    from api_manager.routes import auth, admin, inference, stream
    app.include_router(auth.router)
    app.include_router(admin.router)
    app.include_router(inference.router)
    app.include_router(stream.router)

    # ── Health endpoint ───────────────────────────────────────────────
    @app.get("/health", tags=["health"])
    async def health() -> dict:
        return {
            "status":  "ok",
            "version": cfg.project.version,
            "uptime":  round(time.time() - _start_time, 1),
        }

    # ── Static UI ─────────────────────────────────────────────────────
    ui_dist = Path(__file__).parent.parent / "ui" / "portal" / "dist"
    if ui_dist.exists():
        app.mount("/ui", StaticFiles(directory=str(ui_dist), html=True), name="ui")
        logger.info("UI mounted at /ui from %s", ui_dist)

    return app
