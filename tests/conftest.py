"""Test-suite defaults.

* DGB_SECRET_KEY: the API refuses to start without it, so the suite supplies a throw-away one
  (never used outside pytest).
* Database: by default every test gets its own empty SQLite file, so tests never touch
  instance/dgb.sqlite3 and never see each other's accounts.
* PostgreSQL: set DGB_TEST_DATABASE_URL to run the same suite against PostgreSQL. The schema of that
  database is wiped before every test, so the database NAME MUST END IN "_test" (enforced below).
"""
import os
import tempfile

import pytest

os.environ.setdefault("DGB_SECRET_KEY", "pytest-only-secret-key-0123456789abcdef0123456789")
# Safe default for anything imported at collection time, before the per-test fixture runs.
os.environ.setdefault("DGB_DATABASE_URL", "sqlite:///" + os.path.join(tempfile.gettempdir(), "dgb_pytest_collect.sqlite3").replace("\\", "/"))

_PG_TEST_URL = os.environ.get("DGB_TEST_DATABASE_URL", "").strip()


def _wipe_postgres(url: str) -> None:
    from sqlalchemy import create_engine, text
    from sqlalchemy.engine import make_url

    parsed = make_url(url)
    if not parsed.get_backend_name().startswith("postgresql"):
        raise RuntimeError("DGB_TEST_DATABASE_URL must be a postgresql:// URL")
    if not (parsed.database or "").endswith("_test"):
        raise RuntimeError("Refusing to wipe database %r: its name must end in '_test'" % parsed.database)
    engine = create_engine(url)
    try:
        with engine.begin() as conn:
            conn.execute(text("DROP SCHEMA public CASCADE"))
            conn.execute(text("CREATE SCHEMA public"))
    finally:
        engine.dispose()


@pytest.fixture(autouse=True)
def _fresh_database(tmp_path, monkeypatch):
    try:
        from database import reset_engine
    except ImportError:          # database package not installed in this checkout
        yield
        return
    if _PG_TEST_URL:
        _wipe_postgres(_PG_TEST_URL)
        monkeypatch.setenv("DGB_DATABASE_URL", _PG_TEST_URL)
    else:
        monkeypatch.setenv("DGB_DATABASE_URL", "sqlite:///" + (tmp_path / "test.sqlite3").as_posix())
    reset_engine()
    yield
    reset_engine()
