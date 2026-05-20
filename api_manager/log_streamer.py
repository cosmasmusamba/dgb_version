"""
api_manager/log_streamer.py
=============================
TCP log broadcast server.

LogBroadcastServer:
  - Runs a background thread that tails the training log file
  - Parses each new line as a structured event
  - Publishes to BroadcastHub so SSE/WS endpoints receive live logs

HubLogHandler:
  - Python logging.Handler that publishes to BroadcastHub
  - Attached to the root logger during training so all log records
    arrive in the admin dashboard with zero extra calls
"""
from __future__ import annotations

import json
import logging
import re
import threading
import time
from pathlib import Path
from typing import Optional

from modules.utils.streaming import get_training_hub, StreamEvent

logger = logging.getLogger(__name__)

# Regex to parse structured training log lines
_LOG_RE = re.compile(
    r"(?P<ts>\d{4}-\d{2}-\d{2} \d{2}:\d{2}:\d{2})"
    r"\s+\|\s+(?P<level>\w+)\s+\|\s+(?P<name>[^|]+)\s+\|\s+(?P<msg>.+)"
)


def _parse_log_line(line: str) -> Optional[dict]:
    m = _LOG_RE.match(line.strip())
    if not m:
        return None
    msg = m.group("msg").strip()
    level = m.group("level").strip()
    return {"time": m.group("ts"), "level": level, "name": m.group("name").strip(), "message": msg}


class LogBroadcastServer:
    """
    Background thread that tails the training log file and publishes
    each new line to the BroadcastHub.

    Parameters
    ----------
    log_file:  Path to the training .log or .jsonl file.
    port:      Unused — kept for API compatibility.
    """

    def __init__(
        self,
        log_file: Optional[Path] = None,
        port:     int = 5555,
    ) -> None:
        self._log_file = log_file
        self._port     = port
        self._stop     = threading.Event()
        self._thread: Optional[threading.Thread] = None

    def start(self) -> None:
        self._stop.clear()
        self._thread = threading.Thread(
            target=self._tail_loop, daemon=True, name="dgb-log-broadcast"
        )
        self._thread.start()
        logger.info("LogBroadcastServer started (port=%d)", self._port)

    def stop(self) -> None:
        self._stop.set()
        if self._thread:
            self._thread.join(timeout=5)

    def _tail_loop(self) -> None:
        hub = get_training_hub()

        # Discover log file dynamically if not given
        def _find_log() -> Optional[Path]:
            if self._log_file and Path(self._log_file).exists():
                return Path(self._log_file)
            try:
                from configs.loader import get_config
                from modules.utils.path_resolver import init_path_resolver
                cfg = get_config()
                res = init_path_resolver(cfg.project.model_id, cfg)
                p   = res.training_log()
                return p if p.exists() else None
            except Exception:
                return None

        log_path: Optional[Path] = None
        fh = None

        while not self._stop.is_set():
            if log_path is None or not log_path.exists():
                log_path = _find_log()
                if log_path is None:
                    time.sleep(2.0)
                    continue
                if fh:
                    fh.close()
                fh = log_path.open("r", encoding="utf-8", errors="replace")
                fh.seek(0, 2)   # seek to end
                logger.debug("Tailing log: %s", log_path.name)

            line = fh.readline()
            if not line:
                time.sleep(0.1)
                continue

            parsed = _parse_log_line(line)
            if parsed:
                hub.publish(StreamEvent.log(
                    message=parsed["message"],
                    level=parsed["level"],
                    stage=parsed.get("name", "training"),
                    time=parsed["time"],
                ))
            else:
                stripped = line.strip()
                if stripped:
                    hub.publish(StreamEvent.log(message=stripped, level="INFO"))


class HubLogHandler(logging.Handler):
    """
    Standard logging.Handler that forwards every log record to BroadcastHub.
    Attach to the root logger during training:

        root = logging.getLogger()
        root.addHandler(HubLogHandler())
    """

    def emit(self, record: logging.LogRecord) -> None:
        try:
            hub = get_training_hub()
            hub.publish(StreamEvent.log(
                message=self.format(record),
                level=record.levelname,
                stage=record.name,
                time=self.formatTime(record),
            ))
        except Exception:
            pass
