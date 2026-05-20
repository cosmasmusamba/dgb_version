"""
api_manager/routes/stream.py
==============================
Real-time streaming endpoints.

GET  /stream/events    — SSE: all training events (log + metric + progress)
GET  /stream/logs      — SSE: training log lines only
GET  /stream/metrics   — SSE: metric events only
WS   /stream/ws/events — WebSocket: all events
WS   /stream/ws/infer  — WebSocket: inference streaming
"""
from __future__ import annotations

import asyncio
import json
import logging
from typing import AsyncGenerator

from fastapi import APIRouter, Depends, WebSocket, WebSocketDisconnect
from fastapi.responses import StreamingResponse
from sse_starlette.sse import EventSourceResponse

from api_manager.auth.api_security import get_current_user, TokenData
from configs.constants import SSE_RETRY_MS, StreamEventType
from modules.utils.streaming import get_training_hub, StreamEvent

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/stream", tags=["stream"])


async def _event_generator(
    type_filter: str = "",
) -> AsyncGenerator[str, None]:
    """
    Yields SSE-formatted strings from the training hub.
    Sends heartbeat comments every 15s to keep connections alive.
    """
    hub = get_training_hub()

    # Replay recent history for late-joining clients
    for ev in hub.recent_history(n=50):
        if not type_filter or ev.type == type_filter:
            yield ev.to_sse()

    async with hub.subscribe() as q:
        while True:
            try:
                event: StreamEvent = await asyncio.wait_for(q.get(), timeout=15.0)
                if not type_filter or event.type == type_filter:
                    yield event.to_sse()
            except asyncio.TimeoutError:
                yield ": heartbeat\n\n"
            except Exception as exc:
                logger.debug("SSE generator error: %s", exc)
                break


@router.get("/events")
async def sse_all_events(
    user: TokenData = Depends(get_current_user),
):
    """SSE stream of all training events."""
    return StreamingResponse(
        _event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control":  "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@router.get("/logs")
async def sse_logs(
    user: TokenData = Depends(get_current_user),
):
    """SSE stream of log events only."""
    return StreamingResponse(
        _event_generator(StreamEventType.LOG),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.get("/metrics")
async def sse_metrics(
    user: TokenData = Depends(get_current_user),
):
    """SSE stream of metric events only."""
    return StreamingResponse(
        _event_generator(StreamEventType.METRIC),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


@router.websocket("/ws/events")
async def ws_events(ws: WebSocket):
    """WebSocket stream of all training events."""
    await ws.accept()
    hub = get_training_hub()
    for ev in hub.recent_history(n=50):
        try:
            await ws.send_text(json.dumps(ev.to_dict()))
        except WebSocketDisconnect:
            return

    async with hub.subscribe() as q:
        while True:
            try:
                event: StreamEvent = await asyncio.wait_for(q.get(), timeout=20.0)
                await ws.send_text(json.dumps(event.to_dict()))
            except asyncio.TimeoutError:
                try:
                    await ws.send_text(json.dumps({"type": "heartbeat"}))
                except WebSocketDisconnect:
                    break
            except WebSocketDisconnect:
                break
            except Exception as exc:
                logger.debug("WS events error: %s", exc)
                break


@router.websocket("/ws/infer")
async def ws_infer(ws: WebSocket):
    """
    WebSocket inference endpoint.
    Receives: {"prompt": "...", "max_new_tokens": 256, "temperature": 1.0}
    Sends:    {"token": "..."} × N, then {"done": true}
    """
    await ws.accept()
    try:
        while True:
            raw = await ws.receive_text()
            req = json.loads(raw)
            prompt = req.get("prompt", "")
            if not prompt:
                await ws.send_text(json.dumps({"error": "prompt required"}))
                continue

            from inference.engine.generator import get_engine
            from inference.sampling.top_p_sampler import GenerationConfig

            engine = get_engine()
            if engine is None:
                await ws.send_text(json.dumps({"error": "engine not initialised"}))
                continue

            cfg = GenerationConfig(
                max_new_tokens=req.get("max_new_tokens", 256),
                temperature=req.get("temperature", 1.0),
                top_p=req.get("top_p", 0.9),
                do_sample=req.get("do_sample", True),
            )
            async for tok in engine.stream(prompt, cfg):
                await ws.send_text(json.dumps({"token": tok}))
            await ws.send_text(json.dumps({"done": True}))

    except WebSocketDisconnect:
        logger.debug("WS inference client disconnected")
    except Exception as exc:
        logger.warning("WS inference error: %s", exc)
        try:
            await ws.send_text(json.dumps({"error": str(exc)}))
        except Exception:
            pass
