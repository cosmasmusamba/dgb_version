"""
api_manager/routes/auth.py
============================
Authentication endpoints.

POST /auth/login    — username+password → token pair
POST /auth/refresh  — refresh token → new access token
POST /auth/register — disabled by policy; use secure admin bootstrap
GET  /auth/me       — current user info
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel
from starlette.concurrency import run_in_threadpool

from api_manager.auth.api_security import (
    authenticate_user, create_token_pair, decode_token,
    get_current_user, load_account, TokenData, TokenPair,
)
from modules.utils.error_handler import AuthError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


class RefreshRequest(BaseModel):
    refresh_token: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    role:     str | None = None


def _audit(action: str, actor, ok: bool, request: Request, detail=None) -> None:
    """Write an audit row; never let an audit failure break the request."""
    try:
        from database import get_session, audit
        with get_session() as s:
            audit.record(s, action, actor=actor, ok=ok, client=audit.client_key(request), detail=detail)
    except Exception:  # noqa: BLE001
        logger.exception("audit write failed for %s", action)


@router.post("/login", response_model=TokenPair)
async def login(request: Request, form: OAuth2PasswordRequestForm = Depends()):
    user = await run_in_threadpool(authenticate_user, form.username, form.password)
    if not user:
        logger.warning("Failed login attempt for user: %s", form.username)
        await run_in_threadpool(_audit, "login_failed", None, False, request, {"username": form.username[:64]})
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    role = getattr(user["role"], "value", user["role"])
    logger.info("User logged in: %s  role=%s", user["sub"], role)
    await run_in_threadpool(_audit, "login_success", user["sub"], True, request)
    return create_token_pair(sub=user["sub"], role=role)


@router.post("/refresh")
async def refresh(request: Request, req: RefreshRequest):
    try:
        data = decode_token(req.refresh_token)
        if data.type != "refresh":
            raise AuthError("Not a refresh token")
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=exc.message)
    # The account record decides: a disabled, deleted or demoted user cannot keep refreshing old privileges.
    account = await run_in_threadpool(load_account, data.sub)
    if account is None or not account["is_active"]:
        await run_in_threadpool(_audit, "refresh_denied", data.sub, False, request)
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail="Account is not available")
    pair = create_token_pair(sub=account["username"], role=account["role"])
    return {"access_token": pair.access_token, "token_type": "bearer"}


@router.get("/me")
async def me(user: TokenData = Depends(get_current_user)):
    return {"sub": user.sub, "role": user.role}


@router.post("/register", status_code=403)
async def register(req: RegisterRequest):
    """Public self-registration is disabled to prevent privilege escalation."""
    raise HTTPException(
        status_code=status.HTTP_403_FORBIDDEN,
        detail="Self-registration is disabled. Provision users via secure admin bootstrap or admin-only tooling.",
    )
