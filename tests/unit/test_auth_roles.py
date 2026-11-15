"""Role handling and authorization, with accounts stored in the database.

History: the bootstrap admin could log in and /auth/me said 'UserRole.ADMIN', but /v1/admin/health
returned 403, because the token stored str(UserRole.ADMIN) instead of 'admin'. Since the database
foundation, the account record (not the token) is the source of truth for role and active status,
so a demoted or disabled user loses access immediately.
"""
import time

import pytest

SECRET = "test-secret-key-for-unit-tests-0123456789"


@pytest.fixture()
def api(monkeypatch):
    monkeypatch.setenv("DGB_SECRET_KEY", SECRET)
    monkeypatch.delenv("DGB_BOOTSTRAP_ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("DGB_BOOTSTRAP_ADMIN_PASSWORD", raising=False)
    from fastapi.testclient import TestClient
    from api_manager.app import create_app
    return TestClient(create_app())


@pytest.fixture()
def accounts(api):
    """admin (admin), bob (user), reader (readonly), gone (admin, disabled)."""
    from database import get_session
    from database import users as repo
    with get_session() as s:
        repo.create_user(s, "admin", None, "admin")
        repo.create_user(s, "second-admin", None, "admin")
        repo.create_user(s, "bob", None, "user")
        repo.create_user(s, "reader", None, "readonly")
        gone = repo.create_user(s, "gone", None, "admin")
        gone.is_active = False
    return ["admin", "bob", "reader", "gone"]


def _token(sub, role, token_type="access"):
    """A correctly signed token with an arbitrary role claim."""
    from jose import jwt
    payload = {"sub": sub, "role": role, "type": token_type, "exp": int(time.time()) + 300}
    return jwt.encode(payload, SECRET, algorithm="HS256")


def _h(sub, role, token_type="access"):
    return {"Authorization": "Bearer " + _token(sub, role, token_type)}


def test_token_role_claim_is_the_plain_value():
    from api_manager.auth.api_security import create_token_pair, decode_token
    from configs.constants import UserRole
    pair = create_token_pair("admin", UserRole.ADMIN)
    assert decode_token(pair.access_token).role == "admin"
    assert decode_token(pair.refresh_token).role == "admin"


def test_admin_account_reaches_admin_health_and_me_reports_plain_role(api, accounts):
    h = _h("admin", "admin")
    assert api.get("/auth/me", headers=h).json() == {"sub": "admin", "role": "admin"}
    r = api.get("/v1/admin/health", headers=h)
    assert r.status_code == 200 and r.json()["status"] == "ok"


def test_database_role_wins_over_the_token_claim(api, accounts):
    # bob is a plain user in the database; a token that CLAIMS admin does not make him one.
    h = _h("bob", "admin")
    assert api.get("/auth/me", headers=h).json()["role"] == "user"
    assert api.get("/v1/admin/health", headers=h).status_code == 403


@pytest.mark.parametrize("sub", ["bob", "reader"])
def test_non_admin_accounts_are_forbidden(api, accounts, sub):
    assert api.get("/v1/admin/health", headers=_h(sub, "user")).status_code == 403


def test_unknown_and_disabled_accounts_are_401(api, accounts):
    assert api.get("/v1/admin/health", headers=_h("nobody", "admin")).status_code == 401
    assert api.get("/v1/admin/health", headers=_h("gone", "admin")).status_code == 401
    assert api.get("/auth/me", headers=_h("gone", "admin")).status_code == 401


def test_no_token_and_garbage_token_are_401(api, accounts):
    assert api.get("/v1/admin/health").status_code == 401
    assert api.get("/v1/admin/health", headers={"Authorization": "Bearer nope"}).status_code == 401


@pytest.mark.parametrize("claim", ["superuser", "ADMIN", "Admin", "", "admin ", "root"])
def test_unknown_or_malformed_role_claims_are_rejected(api, accounts, claim):
    assert api.get("/v1/admin/health", headers=_h("admin", claim)).status_code == 401


def test_legacy_enum_text_claim_still_decodes_and_database_decides(api, accounts):
    # Tokens issued by the old buggy build carry 'UserRole.ADMIN'. The role comes from the database now.
    assert api.get("/v1/admin/health", headers=_h("admin", "UserRole.ADMIN")).status_code == 200
    assert api.get("/v1/admin/health", headers=_h("bob", "UserRole.ADMIN")).status_code == 403
    assert api.get("/v1/admin/health", headers=_h("admin", "UserRole.NOPE")).status_code == 401


def test_demotion_and_disabling_take_effect_immediately(api, accounts):
    from database import get_session
    from database import users as repo
    h = _h("second-admin", "admin")
    assert api.get("/v1/admin/health", headers=h).status_code == 200
    with get_session() as s:
        repo.set_role(s, "second-admin", "user")
    assert api.get("/v1/admin/health", headers=h).status_code == 403
    with get_session() as s:
        repo.set_active(s, "second-admin", False)
    assert api.get("/auth/me", headers=h).status_code == 401


def test_refresh_takes_the_role_from_the_database_not_the_old_token(api, accounts):
    from api_manager.auth.api_security import decode_token
    r = api.post("/auth/refresh", json={"refresh_token": _token("admin", "UserRole.READONLY", "refresh")})
    assert r.status_code == 200
    assert decode_token(r.json()["access_token"]).role == "admin"
    r = api.post("/auth/refresh", json={"refresh_token": _token("bob", "admin", "refresh")})
    assert decode_token(r.json()["access_token"]).role == "user"


def test_refresh_is_refused_for_disabled_or_missing_accounts(api, accounts):
    for sub in ("gone", "nobody"):
        r = api.post("/auth/refresh", json={"refresh_token": _token(sub, "admin", "refresh")})
        assert r.status_code == 401


def test_refresh_token_cannot_be_used_as_access_token_and_vice_versa(api, accounts):
    assert api.get("/v1/admin/health", headers=_h("admin", "admin", "refresh")).status_code == 401
    r = api.post("/auth/refresh", json={"refresh_token": _token("admin", "admin", "access")})
    assert r.status_code == 401


def test_bootstrap_admin_login_flow_end_to_end(api, monkeypatch):
    """The scenario from the bug report: login -> /auth/me -> /v1/admin/health, via the database."""
    from api_manager.auth import api_security as sec
    from database import get_session, audit
    from database import users as repo
    try:
        hashed = sec.hash_password("correct horse battery")
    except Exception as exc:  # noqa: BLE001 - passlib/bcrypt incompatibility in some environments
        pytest.skip(f"password hashing unavailable here: {exc}")
    with get_session() as s:
        repo.create_user(s, "boss", hashed, "admin")
    r = api.post("/auth/login", data={"username": "Boss", "password": "correct horse battery"})
    assert r.status_code == 200                               # usernames are case-insensitive
    h = {"Authorization": "Bearer " + r.json()["access_token"]}
    assert api.get("/auth/me", headers=h).json() == {"sub": "boss", "role": "admin"}
    assert api.get("/v1/admin/health", headers=h).status_code == 200
    assert api.post("/auth/login", data={"username": "boss", "password": "wrong"}).status_code == 401
    with get_session() as s:
        actions = [(e["action"], e["ok"]) for e in audit.recent(s, 10)]
    assert ("login_success", True) in actions and ("login_failed", False) in actions
    with get_session() as s:
        repo.create_user(s, "backup-admin", None, "admin")    # so boss is not the only admin
        repo.set_active(s, "boss", False)
    assert api.post("/auth/login", data={"username": "boss", "password": "correct horse battery"}).status_code == 401
