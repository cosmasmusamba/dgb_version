"""
modules/utils/streaming.py
============================
Async-safe SSE/WebSocket broadcast hub.

BroadcastHub:
  - Singleton publish/subscribe fan-out for StreamEvents
  - Multiple async subscriber queues (one per SSE connection)
  - Non-blocking publish (drops if queue full)
  - Auto-cleans disconnected subscribers

StreamEvent:
  - Typed factory methods: .log(), .metric(), .progress(), .status(), .done()
  - Serialises to SSE data: payload over a text/event-stream connection

StreamQueue:
  - Thin wrapper around asyncio.Queue used by SSE endpoint
  - Type-aliased so callers can `from modules.utils.streaming import StreamQueue`

Helpers:
  - reset_training_hub()  — replace the singleton (tests / multi-run)
  - sse_format(event)     — format a dict as a raw SSE string
"""
from __future__ import annotations

import asyncio
import json
import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Dict, Optional, Set

from configs.constants import STREAM_QUEUE_MAXSIZE, StreamEventType

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# StreamEvent
# ---------------------------------------------------------------------------

@dataclass
class StreamEvent:
    """One structured event pushed through the broadcast hub."""
    type:      str
    payload:   Dict[str, Any] = field(default_factory=dict)
    timestamp: float          = field(default_factory=time.time)

    # ── Factory methods ────────────────────────────────────────────────

    @classmethod
    def log(cls, message: str, level: str = "INFO", stage: str = "pipeline",
            **extra: Any) -> "StreamEvent":
        return cls(StreamEventType.LOG, {
            "message": message, "level": level, "stage": stage, **extra,
        })

    @classmethod
    def metric(cls, **metrics: Any) -> "StreamEvent":
        return cls(StreamEventType.METRIC, dict(metrics))

    @classmethod
    def progress(cls, **data: Any) -> "StreamEvent":
        return cls(StreamEventType.PROGRESS, dict(data))

    @classmethod
    def status(cls, status: str, stage: str = "", **extra: Any) -> "StreamEvent":
        return cls(StreamEventType.STATUS, {
            "status": status, "stage": stage, **extra,
        })

    @classmethod
    def error(cls, message: str, stage: str = "", **extra: Any) -> "StreamEvent":
        return cls(StreamEventType.ERROR, {
            "message": message, "stage": stage, **extra,
        })

    @classmethod
    def done(cls, stage: str = "") -> "StreamEvent":
        return cls(StreamEventType.DONE, {"stage": stage})

    # ── SSE serialisation ──────────────────────────────────────────────

    def to_sse(self) -> str:
        """Format as a complete SSE message block."""
        data = json.dumps({
            "type":      self.type,
            "payload":   self.payload,
            "timestamp": round(self.timestamp, 3),
        }, ensure_ascii=False, default=str)
        return f"event: {self.type}\ndata: {data}\n\n"

    def to_dict(self) -> Dict[str, Any]:
        return {
            "type":      self.type,
            "payload":   self.payload,
            "timestamp": self.timestamp,
        }


# ---------------------------------------------------------------------------
# StreamQueue  — type alias used by API routes / SSE endpoints
# ---------------------------------------------------------------------------

# Thin alias so callers can type-hint as StreamQueue without knowing asyncio
StreamQueue = asyncio.Queue


# ---------------------------------------------------------------------------
# BroadcastHub
# ---------------------------------------------------------------------------

class BroadcastHub:
    """
    Thread-safe, async-friendly fan-out message bus.

    Usage
    -----
    hub = get_training_hub()

    # Publisher (sync, from training thread):
    hub.publish(StreamEvent.metric(loss=2.3, step=100))

    # Subscriber (async, from FastAPI SSE endpoint):
    async with hub.subscribe() as queue:
        async for event in queue:
            yield event.to_sse()
    """

    def __init__(self, name: str = "training", maxsize: int = STREAM_QUEUE_MAXSIZE) -> None:
        self._name     = name
        self._maxsize  = maxsize
        self._lock     = threading.Lock()
        # Keyed by subscriber id → (loop, asyncio.Queue)
        self._subs: Dict[int, tuple] = {}
        self._next_id  = 0
        self._history: list = []          # last 500 events for late-joining SSE clients
        self._max_hist = 500

    def publish(self, event: StreamEvent) -> None:
        """
        Publish an event to all current subscribers.
        Thread-safe — called from sync training loop.
        """
        with self._lock:
            self._history.append(event)
            if len(self._history) > self._max_hist:
                self._history = self._history[-self._max_hist:]
            dead = []
            for sid, (loop, q) in list(self._subs.items()):
                try:
                    loop.call_soon_threadsafe(self._safe_put, q, event)
                except RuntimeError:
                    dead.append(sid)
            for sid in dead:
                self._subs.pop(sid, None)

    def _safe_put(self, q: asyncio.Queue, event: StreamEvent) -> None:
        try:
            q.put_nowait(event)
        except asyncio.QueueFull:
            pass

    def subscribe(self) -> "_Subscription":
        """Return an async context manager yielding an asyncio.Queue of StreamEvents."""
        return _Subscription(self)

    def _add_subscriber(self, loop: asyncio.AbstractEventLoop) -> tuple:
        q = asyncio.Queue(maxsize=self._maxsize)
        with self._lock:
            sid = self._next_id
            self._next_id += 1
            self._subs[sid] = (loop, q)
        logger.debug("BroadcastHub[%s]: subscriber %d added (%d total)",
                     self._name, sid, len(self._subs))
        return sid, q

    def _remove_subscriber(self, sid: int) -> None:
        with self._lock:
            self._subs.pop(sid, None)
        logger.debug("BroadcastHub[%s]: subscriber %d removed", self._name, sid)

    def recent_history(self, n: int = 100) -> list:
        with self._lock:
            return list(self._history[-n:])

    @property
    def subscriber_count(self) -> int:
        return len(self._subs)


class _Subscription:
    """Async context manager that manages subscriber lifecycle."""
    def __init__(self, hub: BroadcastHub) -> None:
        self._hub = hub
        self._sid: Optional[int] = None
        self._q:   Optional[asyncio.Queue] = None

    async def __aenter__(self) -> asyncio.Queue:
        loop       = asyncio.get_event_loop()
        self._sid, self._q = self._hub._add_subscriber(loop)
        return self._q

    async def __aexit__(self, *_) -> None:
        if self._sid is not None:
            self._hub._remove_subscriber(self._sid)


# ---------------------------------------------------------------------------
# Singleton management
# ---------------------------------------------------------------------------

_hubs: Dict[str, BroadcastHub] = {}
_hub_lock = threading.Lock()


def get_training_hub() -> BroadcastHub:
    """Return (or lazily create) the singleton training broadcast hub."""
    return _get_hub("training")


def _get_hub(name: str) -> BroadcastHub:
    with _hub_lock:
        if name not in _hubs:
            _hubs[name] = BroadcastHub(name=name)
        return _hubs[name]


def reset_training_hub() -> BroadcastHub:
    """
    Replace the training hub singleton with a fresh instance.

    Called between runs in the main pipeline or in tests to ensure
    subscribers from a previous run do not receive events from the next.
    """
    with _hub_lock:
        _hubs["training"] = BroadcastHub(name="training")
    logger.debug("Training hub reset")
    return _hubs["training"]


# ---------------------------------------------------------------------------
# SSE helpers
# ---------------------------------------------------------------------------

def sse_format(data: Dict[str, Any], event: str = "message") -> str:
    """
    Format a plain dict as a raw SSE string.

    Use this when you have a dict (not a StreamEvent) that you need to
    push directly to an SSE endpoint without going through BroadcastHub.

        yield sse_format({"step": 100, "loss": 2.3}, event="metric")
    """
    payload = json.dumps(data, ensure_ascii=False, default=str)
    return f"event: {event}\ndata: {payload}\n\n"