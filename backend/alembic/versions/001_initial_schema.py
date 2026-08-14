"""Initial schema — all 12 tables.

Revision ID: 001_initial
Revises: None
Create Date: 2026-08-10

This migration creates the complete SecureSight database schema from scratch.
It matches the ORM models defined in app/db/models.py exactly.
"""

from typing import Sequence, Union

from alembic import op
import sqlalchemy as sa

revision: str = "001_initial"
down_revision: Union[str, None] = None
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # ── Users ──────────────────────────────────────────────────────────
    op.create_table(
        "users",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("uid", sa.String(32), unique=True, index=True, nullable=False),
        sa.Column("email", sa.String(255), unique=True, index=True, nullable=False),
        sa.Column("hashed_password", sa.String(255), nullable=False),
        sa.Column("full_name", sa.String(255), server_default=""),
        sa.Column("role", sa.String(20), server_default="viewer"),
        sa.Column("api_key_hash", sa.String(64), nullable=True),
        sa.Column("twofa_secret", sa.String(64), nullable=True),
        sa.Column("is_active", sa.Boolean(), server_default=sa.text("true")),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── Cases ──────────────────────────────────────────────────────────
    op.create_table(
        "cases",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("uid", sa.String(32), unique=True, index=True, nullable=False),
        sa.Column("name", sa.String(255), nullable=False),
        sa.Column("reference", sa.String(100), server_default=""),
        sa.Column("description", sa.Text(), server_default=""),
        sa.Column("status", sa.String(20), server_default="open"),
        sa.Column("priority", sa.String(20), server_default="medium"),
        sa.Column("classification", sa.String(50), server_default="unclassified"),
        sa.Column("created_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("updated_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── Analyses ───────────────────────────────────────────────────────
    op.create_table(
        "analyses",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("uid", sa.String(32), unique=True, index=True, nullable=False),
        sa.Column("evidence_id", sa.String(30), index=True, nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("filename", sa.String(255), nullable=False),
        sa.Column("media_type", sa.String(10), nullable=False),
        sa.Column("mime_type", sa.String(100), nullable=False),
        sa.Column("file_size", sa.Integer(), nullable=False),
        sa.Column("sha256", sa.String(64), index=True, nullable=False),
        sa.Column("sha512", sa.String(128), nullable=False),
        sa.Column("status", sa.String(20), server_default="pending"),
        sa.Column("overall_score", sa.Float(), nullable=True),
        sa.Column("verdict", sa.String(30), nullable=True),
        sa.Column("storage_path", sa.String(500), server_default=""),
        sa.Column("report_path", sa.String(500), nullable=True),
        sa.Column("exif_data", sa.JSON(), nullable=True),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("completed_at", sa.DateTime(timezone=True), nullable=True),
    )

    # ── Case Evidence (junction) ───────────────────────────────────────
    op.create_table(
        "case_evidence",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("case_id", sa.Integer(), sa.ForeignKey("cases.id"), nullable=False),
        sa.Column("analysis_id", sa.Integer(), sa.ForeignKey("analyses.id"), nullable=False),
        sa.Column("added_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
        sa.Column("added_by", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
    )

    # ── Case Notes ─────────────────────────────────────────────────────
    op.create_table(
        "case_notes",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("case_id", sa.Integer(), sa.ForeignKey("cases.id"), nullable=False),
        sa.Column("author_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("content", sa.Text(), nullable=False),
        sa.Column("created_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── Case Assignments ───────────────────────────────────────────────
    op.create_table(
        "case_assignments",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("case_id", sa.Integer(), sa.ForeignKey("cases.id"), nullable=False),
        sa.Column("user_id", sa.Integer(), sa.ForeignKey("users.id"), nullable=False),
        sa.Column("role", sa.String(20), server_default="examiner"),
        sa.Column("assigned_at", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── Pipeline Results ───────────────────────────────────────────────
    op.create_table(
        "pipeline_results",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("analysis_id", sa.Integer(), sa.ForeignKey("analyses.id"), nullable=False),
        sa.Column("pipeline", sa.String(30), nullable=False),
        sa.Column("tier", sa.Integer(), nullable=False),
        sa.Column("score", sa.Float(), nullable=False),
        sa.Column("confidence", sa.Float(), server_default="1.0"),
        sa.Column("details", sa.JSON(), nullable=True),
        sa.Column("execution_ms", sa.Integer(), server_default="0"),
    )

    # ── Custody Logs ───────────────────────────────────────────────────
    op.create_table(
        "custody_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("analysis_id", sa.Integer(), sa.ForeignKey("analyses.id"), nullable=False),
        sa.Column("action", sa.String(50), nullable=False),
        sa.Column("actor", sa.String(255), nullable=False),
        sa.Column("ip_address", sa.String(45), server_default=""),
        sa.Column("details", sa.Text(), server_default=""),
        sa.Column("file_hash", sa.String(64), server_default=""),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )

    # ── Heatmaps ───────────────────────────────────────────────────────
    op.create_table(
        "heatmaps",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("analysis_id", sa.Integer(), sa.ForeignKey("analyses.id"), nullable=False),
        sa.Column("heatmap_type", sa.String(30), nullable=False),
        sa.Column("storage_path", sa.String(500), nullable=False),
        sa.Column("frame_index", sa.Integer(), nullable=True),
    )

    # ── Forensic Artifacts ─────────────────────────────────────────────
    op.create_table(
        "forensic_artifacts",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("analysis_id", sa.Integer(), sa.ForeignKey("analyses.id"), nullable=False),
        sa.Column("artifact_type", sa.String(30), nullable=False),
        sa.Column("storage_path", sa.String(500), nullable=False),
        sa.Column("meta_info", sa.JSON(), nullable=True),
    )

    # ── Audit Logs ─────────────────────────────────────────────────────
    op.create_table(
        "audit_logs",
        sa.Column("id", sa.Integer(), primary_key=True, autoincrement=True),
        sa.Column("user_id", sa.Integer(), nullable=True),
        sa.Column("user_email", sa.String(255), server_default=""),
        sa.Column("endpoint", sa.String(255), nullable=False),
        sa.Column("method", sa.String(10), nullable=False),
        sa.Column("status_code", sa.Integer(), server_default="0"),
        sa.Column("ip_address", sa.String(45), server_default=""),
        sa.Column("user_agent", sa.String(500), server_default=""),
        sa.Column("timestamp", sa.DateTime(timezone=True), server_default=sa.func.now()),
    )


def downgrade() -> None:
    op.drop_table("audit_logs")
    op.drop_table("forensic_artifacts")
    op.drop_table("heatmaps")
    op.drop_table("custody_logs")
    op.drop_table("pipeline_results")
    op.drop_table("case_assignments")
    op.drop_table("case_notes")
    op.drop_table("case_evidence")
    op.drop_table("analyses")
    op.drop_table("cases")
    op.drop_table("users")
