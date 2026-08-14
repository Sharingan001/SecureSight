"""Pydantic schemas — request/response models."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any, Optional

from pydantic import BaseModel, Field, field_validator


# ── Enums ──────────────────────────────────────────────────────────────

class Verdict(str, Enum):
    AUTHENTIC = "AUTHENTIC"
    LIKELY_AUTHENTIC = "LIKELY_AUTHENTIC"
    SUSPICIOUS = "SUSPICIOUS"
    LIKELY_FAKE = "LIKELY_FAKE"
    CONFIRMED_FAKE = "CONFIRMED_FAKE"


class MediaType(str, Enum):
    IMAGE = "image"
    VIDEO = "video"


class AnalysisStatus(str, Enum):
    PENDING = "pending"
    PROCESSING = "processing"
    COMPLETED = "completed"
    FAILED = "failed"


class CaseStatus(str, Enum):
    OPEN = "open"
    IN_PROGRESS = "in_progress"
    REVIEW = "review"
    CLOSED = "closed"


# ── Auth ───────────────────────────────────────────────────────────────

class RegisterRequest(BaseModel):
    email: str
    password: str = Field(min_length=8)
    full_name: str = ""

    @field_validator("password")
    @classmethod
    def validate_password_complexity(cls, v: str) -> str:
        import re
        if not re.search(r"[A-Z]", v):
            raise ValueError("Password must contain at least one uppercase letter")
        if not re.search(r"[a-z]", v):
            raise ValueError("Password must contain at least one lowercase letter")
        if not re.search(r"[0-9]", v):
            raise ValueError("Password must contain at least one digit")
        if not re.search(r"[!@#$%^&*(),.?\":{}|<>]", v):
            raise ValueError("Password must contain at least one special character")
        return v


class LoginRequest(BaseModel):
    email: str
    password: str


class TokenResponse(BaseModel):
    access_token: str
    token_type: str = "bearer"
    user_id: str
    email: str
    role: str


class ApiKeyResponse(BaseModel):
    api_key: str
    message: str = "Store this key securely — it cannot be retrieved again."


# ── Pipeline Results ───────────────────────────────────────────────────

class PipelineScore(BaseModel):
    pipeline: str
    tier: int
    score: float = Field(ge=0, le=1)
    confidence: float = Field(ge=0, le=1)
    execution_ms: int
    details: dict[str, Any] = {}



# ── Analysis ───────────────────────────────────────────────────────────

class AnalysisResponse(BaseModel):
    analysis_id: str
    evidence_id: str
    status: AnalysisStatus
    filename: str
    media_type: MediaType
    sha256: str
    overall_score: float | None = Field(None, ge=0, le=100)
    verdict: Verdict | None = None
    pipeline_scores: list[PipelineScore] = []
    heatmap_urls: list[str] = []
    exif_data: dict[str, Any] | None = None
    report_url: str | None = None
    created_at: datetime
    completed_at: datetime | None = None
    cached: bool = False  # True when result was returned from existing analysis (dedup)


class AnalysisSummary(BaseModel):
    analysis_id: str
    evidence_id: str
    filename: str
    media_type: str
    status: str
    overall_score: float | None = None
    verdict: str | None = None
    created_at: datetime


class HistoryResponse(BaseModel):
    total: int
    page: int
    per_page: int
    analyses: list[AnalysisSummary]


# ── Case Management ───────────────────────────────────────────────────

class CaseCreate(BaseModel):
    name: str
    reference: str = ""
    description: str = ""
    priority: str = "medium"
    classification: str = "unclassified"


class CaseUpdate(BaseModel):
    name: str | None = None
    status: CaseStatus | None = None
    priority: str | None = None
    description: str | None = None


class CaseResponse(BaseModel):
    case_id: str
    name: str
    reference: str
    description: str
    status: str
    priority: str
    classification: str
    evidence_count: int = 0
    created_at: datetime


class CaseNoteCreate(BaseModel):
    content: str


class CaseNoteResponse(BaseModel):
    id: int
    author: str
    content: str
    created_at: datetime


# ── Chain of Custody ───────────────────────────────────────────────────

class CustodyEntry(BaseModel):
    action: str
    actor: str
    ip_address: str
    details: str
    file_hash: str
    timestamp: datetime


# ── Health ─────────────────────────────────────────────────────────────

class HealthResponse(BaseModel):
    status: str
    version: str
    gpu_available: bool
    gpu_name: str | None = None
    device: str
    models_loaded: list[str] = []
    uptime_seconds: float
