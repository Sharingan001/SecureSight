"""Auth API — register, login, API key generation, 2FA management.

All endpoints that issue or verify credentials live here.
"""

from typing import Annotated, Optional
from fastapi import APIRouter, Depends, HTTPException, status, Request, Body
from pydantic import BaseModel, Field
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.db import crud
from app.db.models import User
from app.deps import get_current_active_user, limiter, oauth2_scheme
from app.schemas import RegisterRequest, LoginRequest, TokenResponse, ApiKeyResponse
from app.security import (
    hash_password, verify_password, create_access_token,
    generate_api_key, hash_api_key, decode_access_token,
    generate_totp_secret, get_totp_uri, verify_totp,
)

router = APIRouter(prefix="/api/v1/auth", tags=["auth"])


# ── Register ───────────────────────────────────────────────────────────

@router.post("/register", response_model=TokenResponse)
@limiter.limit("5/minute")
async def register(request: Request, req: RegisterRequest = Body(...), db: AsyncSession = Depends(get_db)):
    existing = await crud.get_user_by_email(db, req.email)
    if existing:
        raise HTTPException(status_code=400, detail="Email already registered")

    user = await crud.create_user(
        db,
        email=req.email,
        hashed_password=hash_password(req.password),
        full_name=req.full_name,
        role="examiner",
    )

    token = create_access_token({"sub": user.uid, "email": user.email, "role": user.role})
    return TokenResponse(
        access_token=token,
        user_id=user.uid,
        email=user.email,
        role=user.role,
    )


# ── Login ──────────────────────────────────────────────────────────────

class LoginResponse(BaseModel):
    access_token: Optional[str] = None
    token_type: str = "bearer"
    user_id: str
    email: str
    role: str
    requires_2fa: bool = False


@router.post("/login", response_model=LoginResponse)
@limiter.limit("10/minute")
async def login(request: Request, req: LoginRequest = Body(...), db: AsyncSession = Depends(get_db)):
    user = await crud.get_user_by_email(db, req.email)
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.is_active:
        raise HTTPException(status_code=403, detail="Account is deactivated")

    # If 2FA is enabled, don't issue token yet — require TOTP verification
    if user.twofa_secret:
        return LoginResponse(
            access_token=None,
            user_id=user.uid,
            email=user.email,
            role=user.role,
            requires_2fa=True,
        )

    token = create_access_token({"sub": user.uid, "email": user.email, "role": user.role})
    return LoginResponse(
        access_token=token,
        user_id=user.uid,
        email=user.email,
        role=user.role,
        requires_2fa=False,
    )


# ── 2FA Login Verification ────────────────────────────────────────────

class TwoFALoginRequest(BaseModel):
    email: str
    password: str
    totp_code: str = Field(min_length=6, max_length=6)


@router.post("/login/2fa", response_model=TokenResponse)
@limiter.limit("5/minute")
async def login_with_2fa(request: Request, req: TwoFALoginRequest = Body(...), db: AsyncSession = Depends(get_db)):
    """Complete login for accounts with 2FA enabled."""
    user = await crud.get_user_by_email(db, req.email)
    if not user or not verify_password(req.password, user.hashed_password):
        raise HTTPException(status_code=401, detail="Invalid credentials")

    if not user.twofa_secret:
        raise HTTPException(status_code=400, detail="2FA is not enabled on this account")

    if not verify_totp(user.twofa_secret, req.totp_code):
        raise HTTPException(status_code=401, detail="Invalid 2FA code")

    token = create_access_token({"sub": user.uid, "email": user.email, "role": user.role})
    return TokenResponse(
        access_token=token,
        user_id=user.uid,
        email=user.email,
        role=user.role,
    )


# ── Logout ────────────────────────────────────────────────────────────

@router.post("/logout")
async def logout(
    token: str = Depends(oauth2_scheme)
):
    """Revoke the current access token by adding it to the Redis blocklist.
    
    Always returns 200 OK regardless of token validity — this prevents
    information leakage about whether a token was valid or not.
    """
    if not token:
        return {"status": "ok", "detail": "Logged out"}

    from app.token_blocklist import block_token
    from datetime import datetime, timezone

    try:
        from jose import jwt
        from app.config import settings
        payload = jwt.decode(
            token,
            settings.JWT_SECRET,
            algorithms=[settings.JWT_ALGORITHM],
            options={"verify_exp": False},  # block even expired tokens
        )
        jti = payload.get("jti")
        exp = payload.get("exp")

        if jti and exp:
            now = int(datetime.now(timezone.utc).timestamp())  # consistent UTC
            expires_in = max(1, int(exp) - now)
            block_token(jti, expires_in)

    except Exception:
        pass  # Invalid token — silently succeed (don't leak info)

    return {"status": "ok", "detail": "Logged out successfully"}


# ── API Key Management ────────────────────────────────────────────────

@router.post("/api-key", response_model=ApiKeyResponse)
async def create_api_key_endpoint(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Generate a new API key for the authenticated user.

    The raw key is returned ONCE. Only the SHA-256 hash is stored.
    If the user already has an API key, it is replaced.
    """
    raw_key = generate_api_key()
    key_hash = hash_api_key(raw_key)

    await crud.update_user_api_key(db, current_user, key_hash)

    return ApiKeyResponse(
        api_key=raw_key,
        message="Store this key securely — it cannot be retrieved again.",
    )


# ── 2FA Enrollment ────────────────────────────────────────────────────

class TwoFASetupResponse(BaseModel):
    secret: str
    otpauth_uri: str
    message: str = "Scan the QR code with your authenticator app, then verify with /auth/2fa/verify"


class TwoFAVerifyRequest(BaseModel):
    totp_code: str = Field(min_length=6, max_length=6)


class TwoFAStatusResponse(BaseModel):
    enabled: bool
    message: str


@router.post("/2fa/setup", response_model=TwoFASetupResponse)
async def setup_2fa(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Generate a new TOTP secret for 2FA enrollment.

    Returns the secret and otpauth URI for QR code generation.
    2FA is NOT active until verified via POST /auth/2fa/verify.
    """
    if current_user.twofa_secret:
        raise HTTPException(400, "2FA is already enabled. Disable it first.")

    secret = generate_totp_secret()
    uri = get_totp_uri(secret, current_user.email)

    # Store secret temporarily (will be confirmed on verify)
    # We store it immediately but mark it unverified via a convention:
    # prefix with "pending:" until verified
    await crud.update_user_2fa(db, current_user, f"pending:{secret}")

    return TwoFASetupResponse(secret=secret, otpauth_uri=uri)


@router.post("/2fa/verify", response_model=TwoFAStatusResponse)
async def verify_2fa_setup(
    req: TwoFAVerifyRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Verify the TOTP code to activate 2FA.

    Must be called after /auth/2fa/setup with a valid code from the
    authenticator app.
    """
    if not current_user.twofa_secret or not current_user.twofa_secret.startswith("pending:"):
        raise HTTPException(400, "No pending 2FA setup found. Call /auth/2fa/setup first.")

    secret = current_user.twofa_secret.replace("pending:", "", 1)

    if not verify_totp(secret, req.totp_code):
        raise HTTPException(401, "Invalid TOTP code. Check your authenticator app.")

    # Activate 2FA — store the confirmed secret
    await crud.update_user_2fa(db, current_user, secret)

    return TwoFAStatusResponse(enabled=True, message="2FA is now active on your account.")


@router.delete("/2fa", response_model=TwoFAStatusResponse)
async def disable_2fa(
    req: TwoFAVerifyRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(get_current_active_user),
):
    """Disable 2FA. Requires a valid TOTP code to confirm."""
    if not current_user.twofa_secret or current_user.twofa_secret.startswith("pending:"):
        raise HTTPException(400, "2FA is not enabled on this account.")

    if not verify_totp(current_user.twofa_secret, req.totp_code):
        raise HTTPException(401, "Invalid TOTP code.")

    await crud.update_user_2fa(db, current_user, None)

    return TwoFAStatusResponse(enabled=False, message="2FA has been disabled.")


# ── Profile ────────────────────────────────────────────────────────────

class ProfileResponse(BaseModel):
    user_id: str
    email: str
    full_name: str
    role: str
    has_2fa: bool
    has_api_key: bool


@router.get("/profile", response_model=ProfileResponse)
async def get_profile(current_user: User = Depends(get_current_active_user)):
    """Get the authenticated user's profile."""
    return ProfileResponse(
        user_id=current_user.uid,
        email=current_user.email,
        full_name=current_user.full_name,
        role=current_user.role,
        has_2fa=bool(current_user.twofa_secret and not current_user.twofa_secret.startswith("pending:")),
        has_api_key=bool(current_user.api_key_hash),
    )
