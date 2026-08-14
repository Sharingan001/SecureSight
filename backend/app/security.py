"""Security — JWT tokens, password hashing, API keys, RBAC, 2FA, file safety."""

from __future__ import annotations

import hashlib
import re
import secrets
from datetime import datetime, timedelta, timezone
from enum import Enum
from pathlib import Path
from typing import Any

import bcrypt
import uuid
from jose import JWTError, jwt

from app.config import settings
from app.token_blocklist import is_token_blocked


class Role(str, Enum):
    ADMIN = "admin"
    EXAMINER = "examiner"
    REVIEWER = "reviewer"
    VIEWER = "viewer"


# ── Passwords (direct bcrypt — no passlib wrapper) ─────────────────────

def hash_password(plain: str) -> str:
    """Hash password using bcrypt with auto-generated salt."""
    return bcrypt.hashpw(plain.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")


def verify_password(plain: str, hashed: str) -> bool:
    """Verify a password against its bcrypt hash."""
    return bcrypt.checkpw(plain.encode("utf-8"), hashed.encode("utf-8"))


# ── JWT ────────────────────────────────────────────────────────────────

def create_access_token(data: dict[str, Any], expires_minutes: int | None = None) -> str:
    payload = data.copy()
    expire = datetime.now(timezone.utc) + timedelta(
        minutes=expires_minutes or settings.JWT_EXPIRE_MINUTES
    )
    payload.update({
        "exp": expire,
        "jti": str(uuid.uuid4())
    })
    return jwt.encode(payload, settings.JWT_SECRET, algorithm=settings.JWT_ALGORITHM)


def decode_access_token(token: str) -> dict[str, Any]:
    try:
        payload = jwt.decode(token, settings.JWT_SECRET, algorithms=[settings.JWT_ALGORITHM])
        jti = payload.get("jti")
        if jti and is_token_blocked(jti):
            raise ValueError("Token has been revoked")
        return payload
    except JWTError as exc:
        raise ValueError(f"Invalid token: {exc}") from exc


# ── API Keys ───────────────────────────────────────────────────────────

def generate_api_key() -> str:
    """Generate a cryptographically secure API key (48 hex chars)."""
    return f"ss_{secrets.token_hex(24)}"


def hash_api_key(key: str) -> str:
    """One-way hash the API key for storage."""
    return hashlib.sha256(key.encode()).hexdigest()


# ── 2FA / TOTP ─────────────────────────────────────────────────────────

def generate_totp_secret() -> str:
    """Generate a TOTP secret for 2FA enrollment."""
    import pyotp
    return pyotp.random_base32()


def get_totp_uri(secret: str, email: str) -> str:
    """Build an otpauth:// URI for QR code provisioning."""
    import pyotp
    totp = pyotp.TOTP(secret)
    return totp.provisioning_uri(name=email, issuer_name=settings.APP_NAME)


def verify_totp(secret: str, code: str) -> bool:
    """Verify a 6-digit TOTP code against the user's secret.

    Allows ±1 time window (30s each) to account for clock drift.
    """
    import pyotp
    totp = pyotp.TOTP(secret)
    return totp.verify(code, valid_window=1)


# ── Evidence hashing ───────────────────────────────────────────────────

def compute_file_hash(filepath: str, algorithm: str = "sha256") -> str:
    """Stream-hash a file without loading it entirely into RAM."""
    h = hashlib.new(algorithm)
    with open(filepath, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):  # 1 MB chunks
            h.update(chunk)
    return h.hexdigest()


# ── Filename sanitization ─────────────────────────────────────────────

def sanitize_filename(filename: str) -> str:
    """Sanitize a client-supplied filename to prevent path traversal.

    1. Extract basename (strip directory components)
    2. Replace dangerous characters with underscores
    3. Reject dotfiles
    4. Cap length at 255 characters
    5. Guarantee a fallback name if nothing survives
    """
    # Strip any directory path components (forward slash, backslash)
    name = Path(filename).name

    # Reject empty or dotfile names
    if not name or name.startswith("."):
        name = f"upload_{secrets.token_hex(8)}"

    # Replace anything that isn't alphanumeric, hyphen, underscore, or dot
    name = re.sub(r"[^\w\-.]", "_", name)

    # Collapse multiple underscores/dots
    name = re.sub(r"_{2,}", "_", name)
    name = re.sub(r"\.{2,}", ".", name)

    # Cap length (preserve extension)
    if len(name) > 255:
        stem = Path(name).stem[:240]
        suffix = Path(name).suffix[:15]
        name = stem + suffix

    # Final fallback
    if not name or name == ".":
        name = f"upload_{secrets.token_hex(8)}"

    return name
