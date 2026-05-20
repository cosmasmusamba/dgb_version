"""
api_manager/routes/auth.py
============================
Authentication endpoints.

POST /auth/login    — username+password → token pair
POST /auth/refresh  — refresh token → new access token
POST /auth/register — create new user (admin only in production)
GET  /auth/me       — current user info
"""
from __future__ import annotations

import logging

from fastapi import APIRouter, Depends, HTTPException, status
from fastapi.security import OAuth2PasswordRequestForm
from pydantic import BaseModel

from api_manager.auth.api_security import (
    authenticate_user, create_token_pair, decode_token,
    get_current_user, TokenData, TokenPair, _USERS, hash_password,
)
from configs.constants import UserRole
from modules.utils.error_handler import AuthError

logger = logging.getLogger(__name__)
router = APIRouter(prefix="/auth", tags=["auth"])


class RefreshRequest(BaseModel):
    refresh_token: str


class RegisterRequest(BaseModel):
    username: str
    password: str
    role:     str = UserRole.USER


@router.post("/login", response_model=TokenPair)
async def login(form: OAuth2PasswordRequestForm = Depends()):
    user = authenticate_user(form.username, form.password)
    if not user:
        logger.warning("Failed login attempt for user: %s", form.username)
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Incorrect username or password",
            headers={"WWW-Authenticate": "Bearer"},
        )
    logger.info("User logged in: %s  role=%s", user["sub"], user["role"])
    return create_token_pair(sub=user["sub"], role=user["role"])


@router.post("/refresh")
async def refresh(req: RefreshRequest):
    try:
        data = decode_token(req.refresh_token)
        if data.type != "refresh":
            raise AuthError("Not a refresh token")
        pair = create_token_pair(sub=data.sub, role=data.role)
        return {"access_token": pair.access_token, "token_type": "bearer"}
    except AuthError as exc:
        raise HTTPException(status_code=status.HTTP_401_UNAUTHORIZED, detail=exc.message)


@router.get("/me")
async def me(user: TokenData = Depends(get_current_user)):
    return {"sub": user.sub, "role": user.role}


@router.post("/register", status_code=201)
async def register(req: RegisterRequest):
    """Register a new user. Restrict to admin-only in production."""
    if req.username in _USERS:
        raise HTTPException(status_code=409, detail="Username already exists")
    _USERS[req.username] = {
        "hashed_password": hash_password(req.password),
        "role": req.role,
    }
    logger.info("New user registered: %s  role=%s", req.username, req.role)
    return {"username": req.username, "role": req.role}
