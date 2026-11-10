import os

import pytest
from fastapi.testclient import TestClient

from api_manager.app import create_app


@pytest.fixture
def client():
    app = create_app()
    with TestClient(app) as c:
        yield c


def test_health_ok(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_admin_health_requires_auth(client):
    resp = client.get("/v1/admin/health")
    assert resp.status_code in (401, 403)


def test_login_requires_valid_credentials(client):
    resp = client.post("/auth/login", data={"username": "bad", "password": "bad"})
    assert resp.status_code == 401


def test_login_with_bootstrap_admin_if_configured(client):
    username = os.getenv("DGB_BOOTSTRAP_ADMIN_USERNAME")
    password = os.getenv("DGB_BOOTSTRAP_ADMIN_PASSWORD")
    if not username or not password:
        pytest.skip("Bootstrap admin creds not configured")

    resp = client.post("/auth/login", data={"username": username, "password": password})
    assert resp.status_code == 200
    body = resp.json()
    assert "access_token" in body
    assert "refresh_token" in body
