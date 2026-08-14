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

# ── Startup validation ─────────────────────────────────────────────────
_PLACEHOLDER_SECRETS = {
    "CHANGE-ME-in-production-use-openssl-rand-hex-32",
    "CHANGE-ME-use-openssl-rand-hex-32-in-production",
}

if not settings.DEBUG and settings.JWT_SECRET in _PLACEHOLDER_SECRETS:
    raise RuntimeError(
        "FATAL: JWT_SECRET is still the placeholder value. "
        "Set a real secret via environment variable or .env file before running in production. "
        "Generate one with: python -c \"import secrets; print(secrets.token_hex(32))\""
    )

# Ensure directories exist
settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
