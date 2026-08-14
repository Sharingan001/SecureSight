"""ORM models — 12 tables for the forensic platform."""

from __future__ import annotations

import uuid
from datetime import datetime, timezone
from typing import Optional

from sqlalchemy import (
    Boolean,
    DateTime,
    Enum as SAEnum,
    Float,
    ForeignKey,
    Integer,
    String,
    Text,
    JSON,
)
from sqlalchemy.orm import Mapped, mapped_column, relationship

from app.database import Base


def _utcnow() -> datetime:
    return datetime.now(timezone.utc)


def _uuid() -> str:
    return uuid.uuid4().hex


# ── Users ──────────────────────────────────────────────────────────────

class User(Base):
    __tablename__ = "users"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uid: Mapped[str] = mapped_column(String(32), default=_uuid, unique=True, index=True)
    email: Mapped[str] = mapped_column(String(255), unique=True, index=True)
    hashed_password: Mapped[str] = mapped_column(String(255))
    full_name: Mapped[str] = mapped_column(String(255), default="")
    role: Mapped[str] = mapped_column(String(20), default="viewer")
    api_key_hash: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    twofa_secret: Mapped[Optional[str]] = mapped_column(String(64), nullable=True)
    is_active: Mapped[bool] = mapped_column(Boolean, default=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    analyses: Mapped[list["Analysis"]] = relationship(back_populates="user")


# ── Cases ──────────────────────────────────────────────────────────────

class Case(Base):
    __tablename__ = "cases"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uid: Mapped[str] = mapped_column(String(32), default=_uuid, unique=True, index=True)
    name: Mapped[str] = mapped_column(String(255))
    reference: Mapped[str] = mapped_column(String(100), default="")
    description: Mapped[str] = mapped_column(Text, default="")
    status: Mapped[str] = mapped_column(String(20), default="open")  # open/in_progress/review/closed
    priority: Mapped[str] = mapped_column(String(20), default="medium")  # low/medium/high/critical
    classification: Mapped[str] = mapped_column(String(50), default="unclassified")
    created_by: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"))
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    updated_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow, onupdate=_utcnow)

    evidence: Mapped[list["CaseEvidence"]] = relationship(back_populates="case")
    notes: Mapped[list["CaseNote"]] = relationship(back_populates="case")
    assignments: Mapped[list["CaseAssignment"]] = relationship(back_populates="case")


class CaseEvidence(Base):
    __tablename__ = "case_evidence"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[int] = mapped_column(Integer, ForeignKey("cases.id"))
    analysis_id: Mapped[int] = mapped_column(Integer, ForeignKey("analyses.id"))
    added_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    added_by: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"))

    case: Mapped["Case"] = relationship(back_populates="evidence")
    analysis: Mapped["Analysis"] = relationship(back_populates="case_evidence")


class CaseNote(Base):
    __tablename__ = "case_notes"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[int] = mapped_column(Integer, ForeignKey("cases.id"))
    author_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"))
    content: Mapped[str] = mapped_column(Text)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    case: Mapped["Case"] = relationship(back_populates="notes")
    author: Mapped["User"] = relationship("User", foreign_keys=[author_id], lazy="joined")


class CaseAssignment(Base):
    __tablename__ = "case_assignments"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    case_id: Mapped[int] = mapped_column(Integer, ForeignKey("cases.id"))
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"))
    role: Mapped[str] = mapped_column(String(20), default="examiner")
    assigned_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    case: Mapped["Case"] = relationship(back_populates="assignments")


# ── Analyses ───────────────────────────────────────────────────────────

class Analysis(Base):
    __tablename__ = "analyses"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    uid: Mapped[str] = mapped_column(String(32), default=_uuid, unique=True, index=True)
    evidence_id: Mapped[str] = mapped_column(String(30), index=True)  # EV-YYYYMMDD-XXXX
    user_id: Mapped[int] = mapped_column(Integer, ForeignKey("users.id"))
    filename: Mapped[str] = mapped_column(String(255))
    media_type: Mapped[str] = mapped_column(String(10))  # image / video
    mime_type: Mapped[str] = mapped_column(String(100))
    file_size: Mapped[int] = mapped_column(Integer)
    sha256: Mapped[str] = mapped_column(String(64), index=True)
    sha512: Mapped[str] = mapped_column(String(128))
    status: Mapped[str] = mapped_column(String(20), default="pending")  # pending/processing/completed/failed
    overall_score: Mapped[Optional[float]] = mapped_column(Float, nullable=True)
    verdict: Mapped[Optional[str]] = mapped_column(String(30), nullable=True)
    storage_path: Mapped[str] = mapped_column(String(500), default="")
    report_path: Mapped[Optional[str]] = mapped_column(String(500), nullable=True)
    exif_data: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    created_at: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
    completed_at: Mapped[Optional[datetime]] = mapped_column(DateTime(timezone=True), nullable=True)

    user: Mapped["User"] = relationship(back_populates="analyses")
    pipeline_results: Mapped[list["PipelineResult"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    custody_logs: Mapped[list["CustodyLog"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    heatmaps: Mapped[list["Heatmap"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    forensic_artifacts: Mapped[list["ForensicArtifact"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )
    case_evidence: Mapped[list["CaseEvidence"]] = relationship(
        back_populates="analysis", cascade="all, delete-orphan"
    )


class PipelineResult(Base):
    __tablename__ = "pipeline_results"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(Integer, ForeignKey("analyses.id"))
    pipeline: Mapped[str] = mapped_column(String(30))  # efficientnet, xception, ela, ...
    tier: Mapped[int] = mapped_column(Integer)  # 1-4
    score: Mapped[float] = mapped_column(Float)
    confidence: Mapped[float] = mapped_column(Float, default=1.0)
    details: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)
    execution_ms: Mapped[int] = mapped_column(Integer, default=0)

    analysis: Mapped["Analysis"] = relationship(back_populates="pipeline_results")


# ── Chain of Custody ───────────────────────────────────────────────────

class CustodyLog(Base):
    __tablename__ = "custody_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(Integer, ForeignKey("analyses.id"))
    action: Mapped[str] = mapped_column(String(50))
    actor: Mapped[str] = mapped_column(String(255))
    ip_address: Mapped[str] = mapped_column(String(45), default="")
    details: Mapped[str] = mapped_column(Text, default="")
    file_hash: Mapped[str] = mapped_column(String(64), default="")
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)

    analysis: Mapped["Analysis"] = relationship(back_populates="custody_logs")


# ── Heatmaps & Forensic Artifacts ─────────────────────────────────────

class Heatmap(Base):
    __tablename__ = "heatmaps"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(Integer, ForeignKey("analyses.id"))
    heatmap_type: Mapped[str] = mapped_column(String(30))  # gradcam, ela, copy_move, ...
    storage_path: Mapped[str] = mapped_column(String(500))
    frame_index: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)

    analysis: Mapped["Analysis"] = relationship(back_populates="heatmaps")


class ForensicArtifact(Base):
    __tablename__ = "forensic_artifacts"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    analysis_id: Mapped[int] = mapped_column(Integer, ForeignKey("analyses.id"))
    artifact_type: Mapped[str] = mapped_column(String(30))
    storage_path: Mapped[str] = mapped_column(String(500))
    meta_info: Mapped[Optional[dict]] = mapped_column(JSON, nullable=True)

    analysis: Mapped["Analysis"] = relationship(back_populates="forensic_artifacts")


# ── Audit Log ──────────────────────────────────────────────────────────

class AuditLog(Base):
    __tablename__ = "audit_logs"

    id: Mapped[int] = mapped_column(Integer, primary_key=True, autoincrement=True)
    user_id: Mapped[Optional[int]] = mapped_column(Integer, nullable=True)
    user_email: Mapped[str] = mapped_column(String(255), default="")
    endpoint: Mapped[str] = mapped_column(String(255))
    method: Mapped[str] = mapped_column(String(10))
    status_code: Mapped[int] = mapped_column(Integer, default=0)
    ip_address: Mapped[str] = mapped_column(String(45), default="")
    user_agent: Mapped[str] = mapped_column(String(500), default="")
    timestamp: Mapped[datetime] = mapped_column(DateTime(timezone=True), default=_utcnow)
