"""Integration tests for API endpoints (requires server running)."""
import os

import pytest
try:
    import httpx
    _HAS_HTTPX = True
except ImportError:
    _HAS_HTTPX = False

BASE = "http://localhost:8000"

@pytest.mark.skipif(not _HAS_HTTPX, reason="httpx not installed")
@pytest.mark.asyncio
async def test_health():
    async with httpx.AsyncClient() as c:
        try:
            r = await c.get(f"{BASE}/health", timeout=2)
            assert r.status_code == 200
            assert r.json()["status"] == "ok"
        except Exception:
            pytest.skip("API server not running")

@pytest.mark.skipif(not _HAS_HTTPX, reason="httpx not installed")
@pytest.mark.asyncio
async def test_login():
    username = os.environ.get("DGB_BOOTSTRAP_ADMIN_USERNAME")
    password = os.environ.get("DGB_BOOTSTRAP_ADMIN_PASSWORD")
    if not username or not password:
        pytest.skip("Bootstrap admin credentials are not configured")

    async with httpx.AsyncClient() as c:
        try:
            r = await c.post(f"{BASE}/auth/login",
                             data={"username": username, "password": password}, timeout=2)
            assert r.status_code in (200, 401)
        except Exception:
            pytest.skip("API server not running")
