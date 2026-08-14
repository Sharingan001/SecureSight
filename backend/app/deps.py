"""FastAPI Dependencies — Authentication, Authorization, RBAC.

This module provides the REAL auth enforcement layer. Every protected
route MUST use one of these dependencies via Depends().

Auth methods supported:
  1. Bearer JWT token (Authorization: Bearer <token>)
  2. API Key (X-API-Key: ss_<hex>)

RBAC Roles (hierarchical):
  admin > examiner > reviewer > viewer
"""

from __future__ import annotations

from typing import Optional

from fastapi import Depends, HTTPException, Request, status
from fastapi.security import OAuth2PasswordBearer, APIKeyHeader
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.db import crud
from app.db.models import User
from app.security import decode_access_token, hash_api_key, Role

from slowapi import Limiter
from slowapi.util import get_remote_address

limiter = Limiter(key_func=get_remote_address)

# ── Security schemes ──────────────────────────────────────────────────

oauth2_scheme = OAuth2PasswordBearer(
    tokenUrl="/api/v1/auth/login",
    auto_error=False,  # Don't auto-raise; we handle both JWT and API key
)

api_key_header = APIKeyHeader(
    name="X-API-Key",
    auto_error=False,
)

# Role hierarchy for permission checks
ROLE_HIERARCHY: dict[str, int] = {
    Role.VIEWER.value: 0,
    Role.REVIEWER.value: 1,
    Role.EXAMINER.value: 2,
    Role.ADMIN.value: 3,
}


async def get_current_user(
    token: Optional[str] = Depends(oauth2_scheme),
    api_key: Optional[str] = Depends(api_key_header),
    db: AsyncSession = Depends(get_db),
) -> User:
    """Extract and validate the authenticated user from JWT or API key.

    Tries JWT first, then API key. Raises 401 if neither is valid.

    If REQUIRE_AUTH=False (local dev only), returns an anonymous admin-level
    user so every route is accessible without credentials.
    """
    # Dev bypass — REQUIRE_AUTH=False skips auth enforcement
    if not settings.REQUIRE_AUTH:
        import logging
        _log = logging.getLogger(__name__)
        _log.warning("AUTH BYPASS ACTIVE: REQUIRE_AUTH=False. Never use in production.")
        # Load the real first admin user from DB — synthetic id=0 causes FK violations
        from sqlalchemy import select
        result = await db.execute(
            select(User).where(User.role == "admin", User.is_active == True).limit(1)
        )
        real_admin = result.scalars().first()
        if real_admin:
            return real_admin
        # Fallback: load ANY active user
        result = await db.execute(select(User).where(User.is_active == True).limit(1))
        any_user = result.scalars().first()
        if any_user:
            return any_user
        # Last resort: create a synthetic user (DB is empty, will fail FK on insert anyway)
        _log.error("AUTH BYPASS: No users in DB! Register a user first.")
        anon = User()
        anon.id = 0
        anon.uid = "dev-anon"
        anon.email = "dev@localhost"
        anon.role = "admin"
        anon.is_active = True
        anon.full_name = "Dev Anon"
        return anon

    user: Optional[User] = None

    # Strategy 1: JWT Bearer token
    if token:
        try:
            payload = decode_access_token(token)
            user_uid = payload.get("sub")
            if user_uid:
                user = await crud.get_user_by_uid(db, user_uid)
        except ValueError:
            pass  # Invalid token — fall through to API key check

    # Strategy 2: API Key header
    if user is None and api_key:
        key_hash = hash_api_key(api_key)
        user = await crud.get_user_by_api_key_hash(db, key_hash)

    if user is None:
        raise HTTPException(
            status_code=status.HTTP_401_UNAUTHORIZED,
            detail="Invalid or missing authentication credentials",
            headers={"WWW-Authenticate": "Bearer"},
        )

    return user


async def get_current_active_user(
    current_user: User = Depends(get_current_user),
) -> User:
    """Ensure the authenticated user account is active."""
    if not current_user.is_active:
        raise HTTPException(
            status_code=status.HTTP_403_FORBIDDEN,
            detail="Account is deactivated",
        )
    return current_user


def require_role(*allowed_roles: str):
    """Factory that returns a dependency enforcing minimum role level.

    Usage:
        @router.post("/admin-only")
        async def admin_route(user: User = Depends(require_role("admin"))):
            ...

        @router.get("/examiner-up")
        async def examiner_route(user: User = Depends(require_role("examiner", "admin"))):
            ...
    """
    async def _check_role(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        if current_user.role not in allowed_roles:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' does not have permission. "
                       f"Required: {', '.join(allowed_roles)}",
            )
        return current_user

    return _check_role


def require_min_role(min_role: str):
    """Factory that returns a dependency enforcing a minimum role LEVEL.

    Uses the role hierarchy: admin > examiner > reviewer > viewer.
    require_min_role("reviewer") allows reviewer, examiner, and admin.
    """
    min_level = ROLE_HIERARCHY.get(min_role, 0)

    async def _check_min_role(
        current_user: User = Depends(get_current_active_user),
    ) -> User:
        user_level = ROLE_HIERARCHY.get(current_user.role, -1)
        if user_level < min_level:
            raise HTTPException(
                status_code=status.HTTP_403_FORBIDDEN,
                detail=f"Role '{current_user.role}' does not have sufficient permissions. "
                       f"Minimum required: {min_role}",
            )
        return current_user

    return _check_min_role
