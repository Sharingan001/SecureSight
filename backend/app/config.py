"""SecureSight — Configuration via Pydantic Settings.

All tunables are loaded from environment variables / .env file.
Defaults are for LOCAL DEVELOPMENT only. Production deployments MUST
set JWT_SECRET, DATABASE_URL, CORS_ORIGINS, and all passwords via env.
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Literal

from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=str(Path(__file__).resolve().parent.parent.parent / ".env"),
        env_file_encoding="utf-8", 
        extra="ignore"
    )

    # ── App ────────────────────────────────────────────────────────────
    APP_NAME: str = "SecureSight"
    APP_VERSION: str = "1.0.0"
    DEBUG: bool = False
    BASE_DIR: Path = Path(__file__).resolve().parent.parent

    # ── Server ─────────────────────────────────────────────────────────
    HOST: str = "0.0.0.0"
    PORT: int = 8000
    CORS_ORIGINS: list[str] = ["http://localhost:8000", "http://localhost:3000"]

    # ── Auth ───────────────────────────────────────────────────────────
    JWT_SECRET: str = "CHANGE-ME-in-production-use-openssl-rand-hex-32"
    JWT_ALGORITHM: str = "HS256"
    JWT_EXPIRE_MINUTES: int = 1440  # 24 h

    # Set False to disable auth enforcement (local dev only)
    REQUIRE_AUTH: bool = True

    # ── Database ───────────────────────────────────────────────────────
    DATABASE_URL: str = "postgresql+asyncpg://securesight:securesight@localhost:5432/securesight"

    # ── Redis ──────────────────────────────────────────────────────────
    REDIS_URL: str = "redis://localhost:6379/0"

    # ── MinIO / S3 ─────────────────────────────────────────────────────
    MINIO_ENDPOINT: str = "localhost:9000"
    MINIO_ACCESS_KEY: str = "minioadmin"
    MINIO_SECRET_KEY: str = "minioadmin"
    MINIO_BUCKET: str = "securesight"
    MINIO_SECURE: bool = False

    # ── Model / Inference ──────────────────────────────────────────────
    DEVICE: Literal["auto", "cuda", "cpu"] = "auto"
    USE_FP16: bool = True  # half-precision inference on CUDA (2× speed)
    USE_TTA: bool = True   # Test-Time Augmentation for higher accuracy
    EFFICIENTNET_WEIGHTS: str | None = None  # path to fine-tuned .pth
    XCEPTION_WEIGHTS: str | None = None

    # ── Pipeline ───────────────────────────────────────────────────────
    MAX_UPLOAD_MB: int = 500
    ALLOWED_IMAGE_TYPES: list[str] = ["image/jpeg", "image/png", "image/webp", "image/bmp", "image/tiff"]
    ALLOWED_VIDEO_TYPES: list[str] = ["video/mp4", "video/avi", "video/mov", "video/webm", "video/mkv"]
    VIDEO_SAMPLE_FPS: float = 2.0  # frames per second to sample
    MAX_VIDEO_FRAMES: int = 120
    FACE_MIN_SIZE: int = 40  # px — ignore faces smaller than this
    FACE_CROP_SIZE: int = 380  # resize face crops to NxN

    # Note: Pipeline weights are defined in ensemble.py PIPELINE_WEIGHTS dict.
    # They are not configurable via .env to prevent accidental misconfiguration.

    # ── Verdict thresholds ─────────────────────────────────────────────
    THRESH_AUTHENTIC: float = 15.0
    THRESH_LIKELY_AUTH: float = 35.0
    THRESH_SUSPICIOUS: float = 60.0
    THRESH_LIKELY_FAKE: float = 85.0
    # above LIKELY_FAKE → CONFIRMED_FAKE

    # ── Forensic ───────────────────────────────────────────────────────
    EVIDENCE_ID_PREFIX: str = "EV"

    # ── Uploads / Outputs ──────────────────────────────────────────────
    UPLOAD_DIR: Path = Path("/tmp/securesight/uploads")
    OUTPUT_DIR: Path = Path("/tmp/securesight/outputs")

    @property
    def max_upload_bytes(self) -> int:
        return self.MAX_UPLOAD_MB * 1024 * 1024

    @property
    def resolved_device(self) -> str:
        if self.DEVICE != "auto":
            return self.DEVICE
        try:
            import torch
            return "cuda" if torch.cuda.is_available() else "cpu"
        except ImportError:
            return "cpu"

    def _auto_discover_weights(self):
        """Auto-discover trained weights from training directory."""
        weights_dir = self.BASE_DIR / "weights"
        if self.EFFICIENTNET_WEIGHTS is None:
            path = weights_dir / "efficientnet_b4_deepfake.pth"
            if path.exists():
                self.EFFICIENTNET_WEIGHTS = str(path)
        if self.XCEPTION_WEIGHTS is None:
            path = weights_dir / "xception_deepfake.pth"
            if path.exists():
                self.XCEPTION_WEIGHTS = str(path)


settings = Settings()
settings._auto_discover_weights()

# ── Startup validation — JWT_SECRET strength ───────────────────────────
# This check runs at import time (production only, DEBUG=False).
# A weak JWT secret means every token in the system is forgeable.
# Fail loudly at startup rather than silently in production.

_KNOWN_WEAK_SECRETS = {
    "CHANGE-ME-in-production-use-openssl-rand-hex-32",
    "CHANGE-ME-use-openssl-rand-hex-32-in-production",
    "secret", "password", "changeme", "dev", "test", "admin",
    "your-secret-key", "mysecret", "jwt_secret", "supersecret",
    "1234567890", "abcdefghijklmnopqrstuvwxyz",
}


def _jwt_secret_entropy(secret: str) -> float:
    """Shannon entropy in bits per character.
    A random 32-byte hex string has ~3.9 bits/char.
    A password like 'password123' has ~2.8 bits/char.
    Threshold: 3.5 bits/char is a reasonable minimum.
    """
    import math
    from collections import Counter
    counts = Counter(secret)
    total = len(secret)
    return -sum((c / total) * math.log2(c / total) for c in counts.values())


if not settings.DEBUG:
    _secret = settings.JWT_SECRET
    _errors: list[str] = []

    if _secret.lower() in _KNOWN_WEAK_SECRETS:
        _errors.append("JWT_SECRET is a known-weak or placeholder value")

    if len(_secret) < 32:
        _errors.append(
            f"JWT_SECRET is too short ({len(_secret)} chars). "
            "Minimum 32 characters required for HS256 security."
        )

    if len(_secret) >= 8:  # Only check entropy if long enough to be meaningful
        _entropy = _jwt_secret_entropy(_secret)
        if _entropy < 3.5:
            _errors.append(
                f"JWT_SECRET has low entropy ({_entropy:.2f} bits/char, minimum 3.5). "
                "Use a random value: python -c \"import secrets; print(secrets.token_hex(32))\""
            )

    if _errors:
        raise RuntimeError(
            "FATAL: Insecure JWT_SECRET detected at startup — refusing to run.\n"
            + "\n".join(f"  • {e}" for e in _errors) + "\n"
            "Generate a secure secret with: "
            "python -c \"import secrets; print(secrets.token_hex(32))\""
        )

# Note: UPLOAD_DIR and OUTPUT_DIR are created by lifespan() in main.py on startup.
# Do not create them here — side effects at import time break unit tests.
