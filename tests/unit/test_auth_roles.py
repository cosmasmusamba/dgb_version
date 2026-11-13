"""Role claim handling and admin authorization (authfix1).

Regression for: bootstrap admin logs in, /auth/me says 'UserRole.ADMIN', /v1/admin/health -> 403.
Cause: the token stored str(UserRole.ADMIN) == 'UserRole.ADMIN' instead of 'admin'.
Server-side authorization is unchanged: only a signed token whose role is admin passes.
"""
import time

import pytest

SECRET = "test-secret-key-for-unit-tests-0123456789"


@pytest.fixture()
def api(monkeypatch):
    monkeypatch.setenv("DGB_SECRET_KEY", SECRET)
    from fastapi.testclient import TestClient
    from api_manager.app import create_app
    return TestClient(create_app())


def _bearer(role, token_type="access"):
    from api_manager.auth import api_security as sec
    from datetime import timedelta
    return {"Authorization": "Bearer " + sec._make_token("someone", role, token_type, timedelta(minutes=5))}


def _forged(role_claim, token_type="access"):
    """A correctly signed token with an arbitrary role claim (what an older build could have issued)."""
    from jose import jwt
    payload = {"sub": "admin", "role": role_claim, "type": token_type, "exp": int(time.time()) + 300}
    return {"Authorization": "Bearer " + jwt.encode(payload, SECRET, algorithm="HS256")}


def test_token_role_claim_is_the_plain_value(monkeypatch):
    monkeypatch.setenv("DGB_SECRET_KEY", SECRET)
    from api_manager.auth.api_security import create_token_pair, decode_token
    from configs.constants import UserRole
    pair = create_token_pair("admin", UserRole.ADMIN)
    assert decode_token(pair.access_token).role == "admin"
    assert decode_token(pair.refresh_token).role == "admin"


def test_admin_token_reaches_admin_health_and_me_reports_plain_role(api):
    from configs.constants import UserRole
    h = _bearer(UserRole.ADMIN)
    assert api.get("/auth/me", headers=h).json()["role"] == "admin"
    r = api.get("/v1/admin/health", headers=h)
    assert r.status_code == 200 and r.json()["status"] == "ok"


@pytest.mark.parametrize("role", ["user", "readonly"])
def test_non_admin_roles_are_still_forbidden(api, role):
    assert api.get("/v1/admin/health", headers=_forged(role)).status_code == 403


def test_no_token_and_garbage_token_are_401(api):
    assert api.get("/v1/admin/health").status_code == 401
    assert api.get("/v1/admin/health", headers={"Authorization": "Bearer nope"}).status_code == 401


@pytest.mark.parametrize("claim", ["superuser", "ADMIN", "Admin", "", "admin ", "root"])
def test_unknown_or_malformed_role_claims_are_rejected(api, claim):
    r = api.get("/v1/admin/health", headers=_forged(claim))
    assert r.status_code == 401


def test_legacy_enum_text_claim_is_read_as_its_real_role_not_escalated(api):
    # tokens issued by the buggy build carry 'UserRole.ADMIN' / 'UserRole.USER'
    assert api.get("/auth/me", headers=_forged("UserRole.ADMIN")).json()["role"] == "admin"
    assert api.get("/v1/admin/health", headers=_forged("UserRole.ADMIN")).status_code == 200
    assert api.get("/auth/me", headers=_forged("UserRole.USER")).json()["role"] == "user"
    assert api.get("/v1/admin/health", headers=_forged("UserRole.USER")).status_code == 403
    assert api.get("/v1/admin/health", headers=_forged("UserRole.NOPE")).status_code == 401


def test_refresh_repairs_a_legacy_role_claim(api):
    from api_manager.auth.api_security import decode_token
    legacy = _forged("UserRole.ADMIN", token_type="refresh")["Authorization"].split(" ", 1)[1]
    r = api.post("/auth/refresh", json={"refresh_token": legacy})
    assert r.status_code == 200
    assert decode_token(r.json()["access_token"]).role == "admin"


def test_refresh_token_cannot_be_used_as_access_token(api):
    assert api.get("/v1/admin/health", headers=_forged("admin", token_type="refresh")).status_code == 401


def test_bootstrap_admin_login_flow_end_to_end(api, monkeypatch):
    """The exact scenario from the bug report: /auth/login -> /auth/me -> /v1/admin/health."""
    from api_manager.auth import api_security as sec
    from configs.constants import UserRole
    try:
        hashed = sec.hash_password("correct horse battery")
    except Exception as exc:  # noqa: BLE001 - passlib/bcrypt incompatibility in some environments
        pytest.skip(f"password hashing unavailable here: {exc}")
    monkeypatch.setitem(sec._USERS, "boss", {"hashed_password": hashed, "role": UserRole.ADMIN})
    r = api.post("/auth/login", data={"username": "boss", "password": "correct horse battery"})
    assert r.status_code == 200
    h = {"Authorization": "Bearer " + r.json()["access_token"]}
    assert api.get("/auth/me", headers=h).json() == {"sub": "boss", "role": "admin"}
    assert api.get("/v1/admin/health", headers=h).status_code == 200
    bad = api.post("/auth/login", data={"username": "boss", "password": "wrong"})
    assert bad.status_code == 401
