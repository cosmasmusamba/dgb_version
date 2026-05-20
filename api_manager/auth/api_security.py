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
    SECRET_KEY_ENV_VAR, _FALLBACK_SECRET_KEY, UserRole,
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


def _secret() -> str:
    key = os.environ.get(SECRET_KEY_ENV_VAR, "")
    if not key:
        logger.warning(
            "DGB_SECRET_KEY not set — using insecure fallback. "
            "Set this env var in production."
        )
        return _FALLBACK_SECRET_KEY
    return key


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
    if not _HAS_JOSE:
        return password
    return _pwd_ctx.hash(password)


def verify_password(plain: str, hashed: str) -> bool:
    if not _HAS_JOSE:
        return plain == hashed
    return _pwd_ctx.verify(plain, hashed)


def _make_token(
    sub:  str,
    role: str,
    type: str,
    ttl:  timedelta,
) -> str:
    if not _HAS_JOSE:
        return f"dev-token-{sub}-{role}"
    exp     = datetime.now(timezone.utc) + ttl
    payload = {"sub": sub, "role": role, "type": type, "exp": exp.timestamp()}
    return jwt.encode(payload, _secret(), algorithm=JWT_ALGORITHM)


def create_token_pair(sub: str, role: str = UserRole.USER) -> TokenPair:
    access  = _make_token(sub, role, "access",  timedelta(minutes=ACCESS_TOKEN_TTL_MIN))
    refresh = _make_token(sub, role, "refresh", timedelta(days=REFRESH_TOKEN_TTL_DAYS))
    return TokenPair(access_token=access, refresh_token=refresh)


def decode_token(token: str) -> TokenData:
    if not _HAS_JOSE:
        return TokenData(sub="dev", role=UserRole.ADMIN)
    try:
        payload = jwt.decode(token, _secret(), algorithms=[JWT_ALGORITHM])
        return TokenData(
            sub=payload["sub"],
            role=payload.get("role", UserRole.USER),
            type=payload.get("type", "access"),
            exp=payload.get("exp"),
        )
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


# Simple in-memory user store for demo — replace with DB in production
_USERS: dict = {
    "admin": {
        "hashed_password": hash_password("admin123") if _HAS_JOSE else "admin123",
        "role": UserRole.ADMIN,
    },
    "user": {
        "hashed_password": hash_password("user123") if _HAS_JOSE else "user123",
        "role": UserRole.USER,
    },
}


def authenticate_user(username: str, password: str) -> Optional[dict]:
    user = _USERS.get(username)
    if not user:
        return None
    if not verify_password(password, user["hashed_password"]):
        return None
    return {"sub": username, "role": user["role"]}
