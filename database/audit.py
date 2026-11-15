"""Audit log: who did what, when, and whether it worked. Never stores passwords or message text."""
from __future__ import annotations

import hashlib
import hmac
import json
import os
from typing import Any, List, Optional

from sqlalchemy import select
from sqlalchemy.orm import Session

from database.models import AuditLog


def client_key(request) -> Optional[str]:
    """Keyed hash of the caller's network address: lets us spot abuse without storing the address."""
    try:
        host = request.client.host
    except AttributeError:
        return None
    secret = os.environ.get("DGB_SECRET_KEY", "")
    if not host or not secret:
        return None
    return hmac.new(secret.encode(), host.encode(), hashlib.sha256).hexdigest()[:16]


def record(session: Session, action: str, *, actor: Optional[str] = None, target: Optional[str] = None,
           ok: bool = True, client: Optional[str] = None, detail: Optional[Any] = None) -> None:
    text = None if detail is None else (detail if isinstance(detail, str) else json.dumps(detail, default=str))
    session.add(AuditLog(action=action[:48], actor=(actor or None) and actor[:64],
                         target=(target or None) and target[:64], ok=ok, client_key=client,
                         detail=text[:2000] if text else None))


def recent(session: Session, limit: int = 50) -> List[dict]:
    rows = session.scalars(select(AuditLog).order_by(AuditLog.id.desc()).limit(max(1, min(limit, 500))))
    return [{"id": r.id, "ts": r.ts, "actor": r.actor, "action": r.action, "target": r.target,
             "ok": r.ok, "client_key": r.client_key, "detail": r.detail} for r in rows]
