"""Engine, sessions and migrations.

DGB_DATABASE_URL selects the database, e.g.
    sqlite:///C:/path/dgb.sqlite3            (default: <project>/instance/dgb.sqlite3)
    postgresql+psycopg://user:pass@host/db   (Docker/production; driver psycopg[binary] is in requirements/api.txt)
The schema is created and upgraded by Alembic migrations the first time the engine is used.
"""
from __future__ import annotations

import logging
import os
import threading
from contextlib import contextmanager
from pathlib import Path
from typing import Iterator, Optional

from sqlalchemy import create_engine, event
from sqlalchemy.engine import Engine
from sqlalchemy.orm import Session

logger = logging.getLogger(__name__)

PROJECT_ROOT = Path(__file__).resolve().parent.parent
_lock = threading.Lock()
_engine: Optional[Engine] = None


def normalize_url(url: str) -> str:
    """Accept plain postgresql:// (or postgres://) URLs and select the psycopg 3 driver we ship."""
    for prefix in ("postgresql://", "postgres://"):
        if url.startswith(prefix):
            return "postgresql+psycopg://" + url[len(prefix):]
    return url


def database_url() -> str:
    url = os.environ.get("DGB_DATABASE_URL", "").strip()
    if url:
        return normalize_url(url)
    path = PROJECT_ROOT / "instance" / "dgb.sqlite3"
    path.parent.mkdir(parents=True, exist_ok=True)
    return "sqlite:///" + path.as_posix()


def _is_sqlite(url: str) -> bool:
    return url.startswith("sqlite")


def _sqlite_pragmas(dbapi_conn, _record) -> None:
    cur = dbapi_conn.cursor()
    cur.execute("PRAGMA foreign_keys=ON")
    cur.execute("PRAGMA busy_timeout=5000")
    try:
        cur.execute("PRAGMA journal_mode=WAL")      # concurrent readers while writing; ignored for :memory:
    except Exception:  # noqa: BLE001
        pass
    cur.close()


def _make_engine(url: str) -> Engine:
    kwargs = {"pool_pre_ping": True}
    if _is_sqlite(url):
        kwargs["connect_args"] = {"check_same_thread": False, "timeout": 10}
    engine = create_engine(url, **kwargs)
    if _is_sqlite(url):
        event.listen(engine, "connect", _sqlite_pragmas)
    return engine


def run_migrations(engine: Engine) -> None:
    """Upgrade the database to the latest schema (idempotent)."""
    from alembic import command
    from alembic.config import Config

    cfg = Config()
    cfg.set_main_option("script_location", str(Path(__file__).resolve().parent / "migrations"))
    with engine.begin() as conn:
        cfg.attributes["connection"] = conn
        command.upgrade(cfg, "head")


def get_engine() -> Engine:
    global _engine
    if _engine is None:
        with _lock:
            if _engine is None:
                url = database_url()
                engine = _make_engine(url)
                run_migrations(engine)
                logger.info("Database ready: %s", engine.url.render_as_string(hide_password=True))
                _engine = engine
    return _engine


def init_database() -> Engine:
    """Create/upgrade the schema now (called at API start-up)."""
    return get_engine()


def reset_engine() -> None:
    """Drop the cached engine (tests, or after changing DGB_DATABASE_URL)."""
    global _engine
    with _lock:
        if _engine is not None:
            _engine.dispose()
        _engine = None


@contextmanager
def get_session() -> Iterator[Session]:
    session = Session(get_engine(), expire_on_commit=False)
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
