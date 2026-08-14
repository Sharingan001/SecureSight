"""DB CRUD Operations — optimized with bulk inserts and index usage."""

from __future__ import annotations

from datetime import datetime, timezone
from typing import Any, Optional

from sqlalchemy import select, func
from sqlalchemy.ext.asyncio import AsyncSession

from app.db.models import (
    User, Case, CaseEvidence, CaseNote, CaseAssignment,
    Analysis, PipelineResult, CustodyLog, Heatmap, AuditLog,
)


# ── Users ──────────────────────────────────────────────────────────────

async def get_user_by_email(db: AsyncSession, email: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.email == email))
    return result.scalars().first()


async def get_user_by_uid(db: AsyncSession, uid: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.uid == uid))
    return result.scalars().first()


async def get_user_by_api_key_hash(db: AsyncSession, key_hash: str) -> Optional[User]:
    result = await db.execute(select(User).where(User.api_key_hash == key_hash))
    return result.scalars().first()


async def create_user(db: AsyncSession, **kwargs) -> User:
    user = User(**kwargs)
    db.add(user)
    await db.flush()  # Get the auto-generated id/uid
    await db.commit()  # Commit immediately so the row exists before JWT is issued
    await db.refresh(user)  # Refresh to pick up DB-generated defaults
    return user


# ── Analyses ───────────────────────────────────────────────────────────

async def create_analysis(db: AsyncSession, **kwargs) -> Analysis:
    analysis = Analysis(**kwargs)
    db.add(analysis)
    await db.flush()
    return analysis


async def get_analysis_by_uid(db: AsyncSession, uid: str) -> Optional[Analysis]:
    result = await db.execute(select(Analysis).where(Analysis.uid == uid))
    return result.scalars().first()


async def update_analysis(db: AsyncSession, analysis: Analysis, **kwargs) -> Analysis:
    """Update analysis fields. Uses flush only — get_db() manages the commit boundary."""
    for key, value in kwargs.items():
        setattr(analysis, key, value)
    await db.flush()  # Stage changes without committing
    return analysis


async def delete_analysis(db: AsyncSession, analysis: Analysis) -> None:
    """Delete a pre-fetched Analysis object. Cascade handles child records.
    
    The caller must NOT re-fetch — pass the already-loaded object to avoid
    an unnecessary DB round-trip and a mid-session commit.
    """
    await db.delete(analysis)
    await db.flush()  # Stage the delete; get_db() commits on exit


async def get_user_analyses(db: AsyncSession, user_id: int, page: int = 1, per_page: int = 20):
    offset = (page - 1) * per_page
    count_q = select(func.count()).select_from(Analysis).where(Analysis.user_id == user_id)
    total = (await db.execute(count_q)).scalar() or 0

    items_q = (select(Analysis).where(Analysis.user_id == user_id)
               .order_by(Analysis.created_at.desc()).offset(offset).limit(per_page))
    items = (await db.execute(items_q)).scalars().all()

    return total, items




# ── Pipeline Results ─────────────────────────────────────────────

async def get_pipeline_results(db: AsyncSession, analysis_id: int) -> list[PipelineResult]:
    result = await db.execute(
        select(PipelineResult).where(PipelineResult.analysis_id == analysis_id)
    )
    return list(result.scalars().all())


# ── Custody Log ────────────────────────────────────────────────────────

async def add_custody_log(db: AsyncSession, analysis_id: int, **kwargs) -> CustodyLog:
    entry = CustodyLog(analysis_id=analysis_id, **kwargs)
    db.add(entry)
    await db.flush()
    return entry


async def get_custody_logs(db: AsyncSession, analysis_id: int) -> list[CustodyLog]:
    result = await db.execute(
        select(CustodyLog).where(CustodyLog.analysis_id == analysis_id).order_by(CustodyLog.timestamp)
    )
    return list(result.scalars().all())


# ── Cases ──────────────────────────────────────────────────────────────

async def create_case(db: AsyncSession, **kwargs) -> Case:
    case = Case(**kwargs)
    db.add(case)
    await db.flush()
    return case


async def get_case_by_uid(db: AsyncSession, uid: str) -> Optional[Case]:
    result = await db.execute(select(Case).where(Case.uid == uid))
    return result.scalars().first()


async def get_user_cases(db: AsyncSession, user_id: int) -> list[Case]:
    result = await db.execute(
        select(Case).where(Case.created_by == user_id).order_by(Case.created_at.desc())
    )
    return list(result.scalars().all())


async def get_all_cases(db: AsyncSession, page: int = 1, per_page: int = 50) -> list[Case]:
    """Admin-only: fetch all cases across all users, paginated."""
    offset = (page - 1) * per_page
    result = await db.execute(
        select(Case).order_by(Case.created_at.desc()).offset(offset).limit(per_page)
    )
    return list(result.scalars().all())


async def add_evidence_to_case(db: AsyncSession, case_id: int, analysis_id: int, user_id: int) -> CaseEvidence:
    ev = CaseEvidence(case_id=case_id, analysis_id=analysis_id, added_by=user_id)
    db.add(ev)
    await db.flush()
    return ev


async def add_case_note(db: AsyncSession, case_id: int, author_id: int, content: str) -> CaseNote:
    note = CaseNote(case_id=case_id, author_id=author_id, content=content)
    db.add(note)
    await db.flush()
    return note


async def get_case_notes(db: AsyncSession, case_id: int) -> list[CaseNote]:
    result = await db.execute(
        select(CaseNote).where(CaseNote.case_id == case_id).order_by(CaseNote.created_at)
    )
    return list(result.scalars().all())


# ── Heatmaps ───────────────────────────────────────────────────────────

async def get_heatmaps(db: AsyncSession, analysis_id: int) -> list[Heatmap]:
    result = await db.execute(select(Heatmap).where(Heatmap.analysis_id == analysis_id))
    return list(result.scalars().all())


# ── User updates ───────────────────────────────────────────────────────

async def update_user_api_key(db: AsyncSession, user: User, api_key_hash: str) -> None:
    """Store hashed API key on user record."""
    user.api_key_hash = api_key_hash
    await db.flush()


async def update_user_2fa(db: AsyncSession, user: User, twofa_secret: str | None) -> None:
    """Update user's 2FA secret. Pass None to disable."""
    user.twofa_secret = twofa_secret
    await db.flush()


# ── Admin queries ──────────────────────────────────────────────────────

async def get_all_analyses(db: AsyncSession, page: int = 1, per_page: int = 20):
    """Admin view — all analyses across all users."""
    offset = (page - 1) * per_page
    count_q = select(func.count()).select_from(Analysis)
    total = (await db.execute(count_q)).scalar() or 0

    items_q = (select(Analysis).order_by(Analysis.created_at.desc()).offset(offset).limit(per_page))
    items = (await db.execute(items_q)).scalars().all()

    return total, items


# ── Audit ──────────────────────────────────────────────────────────────

async def add_audit_log(db: AsyncSession, **kwargs) -> None:
    db.add(AuditLog(**kwargs))
    # Don't flush here — let it batch with the next commit
