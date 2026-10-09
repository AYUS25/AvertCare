"""
AvertCare Backend · Firebase JWT Authentication Middleware

Validates Firebase ID tokens on protected routes.
Dependency injected via FastAPI — routes stay clean.
"""

from __future__ import annotations

import firebase_admin
from fastapi import Depends, HTTPException, status
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from firebase_admin import auth

from app.core.config import settings

bearer_scheme = HTTPBearer(auto_error=False)

# ── Initialise Firebase Admin SDK once on import ─────────────
_firebase_initialised = False

def _init_firebase() -> None:
    global _firebase_initialised
    if _firebase_initialised or not settings.FIREBASE_PROJECT_ID:
        return
    try:
        firebase_admin.get_app()
    except ValueError:
        firebase_admin.initialize_app(
            options={"projectId": settings.FIREBASE_PROJECT_ID}
        )
    _firebase_initialised = True


# ── Dependency ────────────────────────────────────────────────

async def require_auth(
    creds: HTTPAuthorizationCredentials | None = Depends(bearer_scheme),
) -> dict:
    """
    FastAPI dependency.  Inject into any protected route:
        @router.post("/api/predict", dependencies=[Depends(require_auth)])

    Returns the decoded Firebase token claims dict.

    In DEV mode (FIREBASE_PROJECT_ID not set) this is a no-op passthrough
    so localhost development never requires a real token.
    """
    if not settings.FIREBASE_PROJECT_ID:
        # Dev bypass — remove in production
        return {"uid": "dev-user", "email": "dev@avertcare.local"}

    if creds is None or not creds.credentials:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in again. The login is missing or expired.",
        )

    _init_firebase()
    try:
        decoded = auth.verify_id_token(creds.credentials)
        return decoded
    except Exception as exc:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Sign in again. The login is missing or expired.",
        ) from exc
