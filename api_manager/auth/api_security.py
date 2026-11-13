"""
api_manager/auth/api_security.py
===================================
JWT authentication and role-based access control.

Tokens
------
  POST /auth/login   → {access_token, refresh_token}
  POST /auth/refresh → {access_token}

Roles
-----
  admin     → all endpoints
  user      → inference endpoints
  readonly  → metrics/status read-only
"""
from __future__ import annotations

import logging
import os
import time
from datetime import datetime, timedelta, timezone
from typing import Optional

from fastapi import Depends, HTTPException, status
from fastapi.security import OAuth2PasswordBearer
from pydantic import BaseModel

from configs.constants import (
    JWT_ALGORITHM, ACCESS_TOKEN_TTL_MIN, REFRESH_TOKEN_TTL_DAYS,
    SECRET_KEY_ENV_VAR, UserRole,
)
from modules.utils.error_handler import AuthError, ForbiddenError

logger = logging.getLogger(__name__)

try:
    from jose import JWTError, jwt
    from passlib.context import CryptContext
    _HAS_JOSE = True
except ImportError:
    _HAS_JOSE = False
    logger.warning("python-jose / passlib not installed — auth disabled")

_pwd_ctx = CryptContext(schemes=["bcrypt"], deprecated="auto") if _HAS_JOSE else None
_oauth2  = OAuth2PasswordBearer(tokenUrl="/auth/login", auto_error=False)


def _require_jwt_backend() -> None:
    if not _HAS_JOSE or _pwd_ctx is None:
        raise RuntimeError(
            "JWT authentication backend is unavailable. Install python-jose[cryptography] and passlib[bcrypt] before starting the API."
        )


def _secret() -> str:
    key = os.environ.get(SECRET_KEY_ENV_VAR, "").strip()
    if not key:
        raise RuntimeError(
            f"{SECRET_KEY_ENV_VAR} is required. Set a strong secret before starting the API."
        )
    return key


def normalize_role(role) -> str:
    """
    Canonical role string: "admin" | "user" | "readonly". Fails closed on anything else.

    str(UserRole.ADMIN) is 'UserRole.ADMIN' (UserRole is a str-Enum), which is what login used to put
    in the token; the admin check then compared 'UserRole.ADMIN' with 'admin' and always said 403.
    Tokens already issued with that text are read as the role they name (no escalation: 'UserRole.USER'
    stays user). Anything unknown raises AuthError, i.e. a 401.
    """
    if isinstance(role, UserRole):
        return role.value
    text = str(role)
    try:
        if text.startswith("UserRole."):
            return UserRole[text[len("UserRole."):]].value
        return UserRole(text).value
    except (KeyError, ValueError):
        raise AuthError(f"Unknown role in token: {text!r}")


class TokenData(BaseModel):
    sub:  str
    role: str = UserRole.USER
    type: str = "access"
    exp:  Optional[float] = None


class TokenPair(BaseModel):
    access_token:  str
    refresh_token: str
    token_type:    str = "bearer"
    expires_in:    int = ACCESS_TOKEN_TTL_MIN * 60


def hash_password(password: str) -> str:
    _require_jwt_backend()
    return _pwd_ctx.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    _require_jwt_backend()
    return _pwd_ctx.verify(plain, hashed)


def _make_token(
    sub:  str,
    role: str,
    type: str,
    ttl:  timedelta,
) -> str:
    _require_jwt_backend()
    exp = datetime.now(timezone.utc) + ttl
    # Build a clean dict – no extra keys
    payload = {
        "sub": str(sub),
        "role": normalize_role(role),
        "type": str(type),
        "exp": int(exp.timestamp())   # integer, not float
    }
    # Ensure no 'iss' key is present
    payload.pop("iss", None)
    logger.debug(f"Creating token for {sub} with payload keys: {list(payload.keys())}")
    return jwt.encode(payload, _secret(), algorithm=JWT_ALGORITHM)


def create_token_pair(sub: str, role: str = UserRole.USER) -> TokenPair:
    access  = _make_token(sub, role, "access",  timedelta(minutes=ACCESS_TOKEN_TTL_MIN))
    refresh = _make_token(sub, role, "refresh", timedelta(days=REFRESH_TOKEN_TTL_DAYS))
    return TokenPair(access_token=access, refresh_token=refresh)


def decode_token(token: str) -> TokenData:
    _require_jwt_backend()
    try:
        payload = jwt.decode(token, _secret(), algorithms=[JWT_ALGORITHM])
        data = TokenData(
            sub=payload["sub"],
            role=normalize_role(payload.get("role", UserRole.USER)),
            type=payload.get("type", "access"),
            exp=payload.get("exp"),
        )
        if data.exp is not None and data.exp <= time.time():
            raise AuthError("Token has expired")
        return data
    except JWTError as exc:
        raise AuthError(f"Invalid token: {exc}")


async def get_current_user(
    token: Optional[str] = Depends(_oauth2),
) -> TokenData:
    if not token:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Not authenticated",
            headers={"WWW-Authenticate": "Bearer"},
        )
    try:
        data = decode_token(token)
        if data.type != "access":
            raise AuthError("Token is not an access token")
        if data.exp and data.exp < time.time():
            raise AuthError("Token has expired")
        return data
    except AuthError as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail=exc.message,
            headers={"WWW-Authenticate": "Bearer"},
        )


def require_role(role: str):
    """Dependency factory that enforces a minimum role."""
    async def _check(user: TokenData = Depends(get_current_user)) -> TokenData:
        hierarchy = {UserRole.READONLY: 0, UserRole.USER: 1, UserRole.ADMIN: 2}
        if hierarchy.get(user.role, 0) < hierarchy.get(role, 99):
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{role}' required — you have '{user.role}'",
            )
        return user
    return _check


def _bootstrap_users() -> dict:
    """Seed the user store only from explicit environment configuration."""
    _require_jwt_backend()
    users: dict = {}
    admin_username = os.environ.get("DGB_BOOTSTRAP_ADMIN_USERNAME", "").strip()
    admin_password = os.environ.get("DGB_BOOTSTRAP_ADMIN_PASSWORD", "").strip()

    if admin_username or admin_password:
        if not admin_username or not admin_password:
            raise RuntimeError(
                "DGB_BOOTSTRAP_ADMIN_USERNAME and DGB_BOOTSTRAP_ADMIN_PASSWORD must be set together. "
                "Partial bootstrap admin configuration is not allowed."
            )

        users[admin_username] = {
            "hashed_password": hash_password(admin_password),
            "role": UserRole.ADMIN,
        }
    return users


# Keep the in-memory store empty by default unless bootstrap creds are supplied
_USERS: dict = _bootstrap_users()


def authenticate_user(username: str, password: str) -> Optional[dict]:
    _require_jwt_backend()
    user = _USERS.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return {"sub": username, "role": user["role"]}
