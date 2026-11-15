"""Account operations. All role/active changes go through here so the same safety rules always apply."""
from __future__ import annotations

import re
from datetime import datetime, timezone
from typing import List, Optional

from sqlalchemy import func, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.models import ROLES, User

_USERNAME_RE = re.compile(r"^[a-z0-9][a-z0-9._@+-]{1,62}$")
MIN_PASSWORD_LEN = 10


class AccountError(ValueError):
    """A rejected account operation (bad input, duplicate, or it would remove the last admin)."""


def normalize_username(username: str) -> str:
    name = (username or "").strip().lower()
    if not _USERNAME_RE.match(name):
        raise AccountError("Username must be 2-63 characters: letters, digits, and . _ @ + - (starting with a letter or digit)")
    return name


def normalize_role(role) -> str:
    value = getattr(role, "value", role)
    if value not in ROLES:
        raise AccountError(f"Role must be one of {', '.join(ROLES)}")
    return value


def validate_password(password: str) -> None:
    if len(password or "") < MIN_PASSWORD_LEN:
        raise AccountError(f"Password must be at least {MIN_PASSWORD_LEN} characters")
    if len(set(password)) < 4:
        raise AccountError("Password is too repetitive")


def snapshot(user: User) -> dict:
    """Plain-data copy, safe to use after the session closes."""
    return {"id": user.id, "username": user.username, "email": user.email, "role": user.role,
            "is_active": user.is_active, "password_hash": user.password_hash,
            "created_at": user.created_at, "last_login_at": user.last_login_at}


def get_user(session: Session, username: str) -> Optional[User]:
    try:
        name = normalize_username(username)
    except AccountError:
        return None
    return session.scalar(select(User).where(User.username == name))


def find_account(session: Session, username: str) -> Optional[dict]:
    user = get_user(session, username)
    return snapshot(user) if user else None


def list_users(session: Session) -> List[dict]:
    return [snapshot(u) for u in session.scalars(select(User).order_by(User.username))]


def create_user(session: Session, username: str, password_hash: Optional[str], role="user",
                email: Optional[str] = None) -> User:
    user = User(username=normalize_username(username), password_hash=password_hash,
                role=normalize_role(role), email=(email or "").strip().lower() or None, is_active=True)
    session.add(user)
    try:
        session.flush()
    except IntegrityError as exc:
        session.rollback()
        raise AccountError("Username or email already exists") from exc
    return user


def count_active_admins(session: Session) -> int:
    return session.scalar(select(func.count()).select_from(User).where(User.role == "admin", User.is_active.is_(True))) or 0


def _guard_last_admin(session: Session, user: User, new_role: str, new_active: bool) -> None:
    losing_admin = user.role == "admin" and user.is_active and (new_role != "admin" or not new_active)
    if losing_admin and count_active_admins(session) <= 1:
        raise AccountError("Refusing: this would leave the system without an active administrator")


def set_role(session: Session, username: str, role) -> User:
    user = _require(session, username)
    role = normalize_role(role)
    _guard_last_admin(session, user, role, user.is_active)
    user.role = role
    return user


def set_active(session: Session, username: str, active: bool) -> User:
    user = _require(session, username)
    _guard_last_admin(session, user, user.role, active)
    user.is_active = bool(active)
    return user


def set_password_hash(session: Session, username: str, password_hash: str) -> User:
    user = _require(session, username)
    user.password_hash = password_hash
    return user


def record_login(session: Session, user_id: int) -> None:
    user = session.get(User, user_id)
    if user is not None:
        user.last_login_at = datetime.now(timezone.utc)


def _require(session: Session, username: str) -> User:
    user = get_user(session, username)
    if user is None:
        raise AccountError(f"No such user: {username}")
    return user
