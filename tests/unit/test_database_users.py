"""Database foundation: migrations, account rules, bootstrap admin, authentication, audit, CLI."""
import pytest

from database import get_engine, get_session
from database import audit
from database import users as repo


def _hash(pw="correct horse battery"):
    from api_manager.auth import api_security as sec
    try:
        return sec.hash_password(pw)
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"password hashing unavailable here: {exc}")


def test_migrations_create_exactly_the_schema_the_models_describe():
    from alembic.autogenerate import compare_metadata
    from alembic.migration import MigrationContext
    from database.models import Base
    with get_engine().connect() as conn:
        diff = compare_metadata(MigrationContext.configure(conn), Base.metadata)
    assert diff == [], f"models and migrations have drifted: {diff}"


def test_migrations_are_idempotent():
    from database.session import run_migrations
    run_migrations(get_engine())
    run_migrations(get_engine())


def test_usernames_are_normalized_and_validated():
    assert repo.normalize_username("  Alice.B ") == "alice.b"
    for bad in ["", "a", "-x", "has space", "x" * 70, "semi;colon"]:
        with pytest.raises(repo.AccountError):
            repo.normalize_username(bad)


def test_create_duplicates_roles_and_email():
    with get_session() as s:
        repo.create_user(s, "alice", None, "user", "Alice@Example.com")
    with get_session() as s:
        assert repo.get_user(s, "ALICE").email == "alice@example.com"
        with pytest.raises(repo.AccountError):
            repo.create_user(s, "Alice", None, "user")
    with get_session() as s:
        with pytest.raises(repo.AccountError):
            repo.create_user(s, "alice2", None, "user", "alice@example.com")
    with get_session() as s:
        with pytest.raises(repo.AccountError):
            repo.create_user(s, "carol", None, "superuser")


def test_the_last_active_admin_cannot_be_demoted_or_disabled():
    with get_session() as s:
        repo.create_user(s, "root", None, "admin")
    with get_session() as s:
        with pytest.raises(repo.AccountError, match="administrator"):
            repo.set_role(s, "root", "user")
    with get_session() as s:
        with pytest.raises(repo.AccountError, match="administrator"):
            repo.set_active(s, "root", False)
    with get_session() as s:
        repo.create_user(s, "root2", None, "admin")
        repo.set_role(s, "root", "user")           # fine now: another active admin exists
    with get_session() as s:
        assert repo.count_active_admins(s) == 1


def test_password_rules():
    repo.validate_password("a-long-enough-password")
    for bad in ["short", "aaaaaaaaaaaa", ""]:
        with pytest.raises(repo.AccountError):
            repo.validate_password(bad)


def test_authenticate_user_paths_and_timing_pad(monkeypatch):
    from api_manager.auth import api_security as sec
    h = _hash()
    with get_session() as s:
        repo.create_user(s, "dana", h, "user")
        off = repo.create_user(s, "off", h, "user")
        off.is_active = False
        repo.create_user(s, "oauth-only", None, "user")
    assert sec.authenticate_user("Dana", "correct horse battery") == {"sub": "dana", "role": "user"}
    with get_session() as s:
        assert repo.get_user(s, "dana").last_login_at is not None
    calls = []
    real = sec.verify_password
    monkeypatch.setattr(sec, "verify_password", lambda p, hh: calls.append(hh) or real(p, hh))
    assert sec.authenticate_user("dana", "nope") is None
    assert sec.authenticate_user("off", "correct horse battery") is None
    assert sec.authenticate_user("oauth-only", "anything") is None
    assert sec.authenticate_user("nobody", "anything") is None
    assert len(calls) == 4            # wrong password, disabled, password-less and unknown users each cost exactly one password check


def test_bootstrap_admin_is_seeded_once_and_never_overwritten(monkeypatch):
    from api_manager.auth import api_security as sec
    monkeypatch.setenv("DGB_BOOTSTRAP_ADMIN_USERNAME", "Boss")
    monkeypatch.setenv("DGB_BOOTSTRAP_ADMIN_PASSWORD", "first-password-123")
    try:
        assert sec.seed_bootstrap_admin() is True
    except Exception as exc:  # noqa: BLE001
        pytest.skip(f"password hashing unavailable here: {exc}")
    assert sec.seed_bootstrap_admin() is False
    monkeypatch.setenv("DGB_BOOTSTRAP_ADMIN_PASSWORD", "second-password-456")
    assert sec.seed_bootstrap_admin() is False               # environment change does not reset the account
    assert sec.authenticate_user("boss", "first-password-123") == {"sub": "boss", "role": "admin"}
    assert sec.authenticate_user("boss", "second-password-456") is None
    with get_session() as s:
        assert ("bootstrap_admin_created", True) in [(e["action"], e["ok"]) for e in audit.recent(s)]


def test_bootstrap_without_configuration_does_nothing_and_partial_config_is_rejected(monkeypatch):
    from api_manager.auth import api_security as sec
    monkeypatch.delenv("DGB_BOOTSTRAP_ADMIN_USERNAME", raising=False)
    monkeypatch.delenv("DGB_BOOTSTRAP_ADMIN_PASSWORD", raising=False)
    assert sec.seed_bootstrap_admin() is False
    monkeypatch.setenv("DGB_BOOTSTRAP_ADMIN_PASSWORD", "only-a-password-1")
    with pytest.raises(RuntimeError, match="must be set together"):
        sec.seed_bootstrap_admin()


def test_audit_records_and_never_exposes_the_address(monkeypatch):
    monkeypatch.setenv("DGB_SECRET_KEY", "k" * 40)

    class Req:
        class client:
            host = "203.0.113.77"
    key = audit.client_key(Req)
    assert key and "203.0.113.77" not in key and key == audit.client_key(Req)
    with get_session() as s:
        audit.record(s, "x" * 100, actor="a" * 100, ok=False, client=key, detail={"k": 1})
    with get_session() as s:
        row = audit.recent(s, 5)[0]
    assert len(row["action"]) == 48 and len(row["actor"]) == 64 and row["ok"] is False and '"k": 1' in row["detail"]


def test_manage_users_cli(monkeypatch, capsys):
    pytest.importorskip("passlib")
    from tools import manage_users
    _hash()
    monkeypatch.setenv("NEW_PW", "cli-password-12345")
    assert manage_users.main(["create", "erin", "--role", "admin", "--password-env", "NEW_PW"]) == 0
    assert manage_users.main(["create", "erin", "--password-env", "NEW_PW"]) == 1          # duplicate
    assert manage_users.main(["create", "frank", "--password-env", "NEW_PW"]) == 0
    assert manage_users.main(["set-role", "erin", "user"]) == 1                             # last admin
    assert manage_users.main(["disable", "frank"]) == 0
    assert manage_users.main(["list"]) == 0
    out = capsys.readouterr().out
    assert "erin" in out and "frank" in out and "2 account(s)" in out and "cli-password" not in out
    assert manage_users.main(["audit", "--limit", "10"]) == 0
    assert "user_created" in capsys.readouterr().out
    monkeypatch.setenv("SHORT", "short")
    assert manage_users.main(["create", "gina", "--password-env", "SHORT"]) == 1


def test_app_start_up_creates_the_bootstrap_admin_and_login_works(monkeypatch):
    """Regression: the seeding call must run on a normal start (not only on the error path)."""
    _hash()
    from fastapi.testclient import TestClient
    from api_manager.app import create_app
    monkeypatch.setenv("DGB_BOOTSTRAP_ADMIN_USERNAME", "Chief")
    monkeypatch.setenv("DGB_BOOTSTRAP_ADMIN_PASSWORD", "start-up-password-1")
    create_app()
    client = TestClient(create_app())                       # a second start must not fail or duplicate
    r = client.post("/auth/login", data={"username": "chief", "password": "start-up-password-1"})
    assert r.status_code == 200
    h = {"Authorization": "Bearer " + r.json()["access_token"]}
    assert client.get("/v1/admin/health", headers=h).status_code == 200
    with get_session() as s:
        assert len([u for u in repo.list_users(s) if u["username"] == "chief"]) == 1


def test_postgres_urls_are_pointed_at_the_psycopg_driver():
    from database.session import normalize_url

    assert normalize_url("postgresql://u:p@localhost:5434/dgb") == "postgresql+psycopg://u:p@localhost:5434/dgb"
    assert normalize_url("postgres://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert normalize_url("postgresql+psycopg://u:p@h/db") == "postgresql+psycopg://u:p@h/db"
    assert normalize_url("sqlite:///x.db") == "sqlite:///x.db"
