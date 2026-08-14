"""Case Management API — Full CRUD for forensic cases.

Cases group multiple analyses (evidence items) together with notes
and team assignments. Every endpoint requires authentication.
"""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException, status
from sqlalchemy.ext.asyncio import AsyncSession

from app.database import get_db
from app.db import crud
from app.db.models import User
from app.deps import get_current_active_user, require_min_role
from app.schemas import (
    CaseCreate, CaseUpdate, CaseResponse,
    CaseNoteCreate, CaseNoteResponse,
)

router = APIRouter(prefix="/api/v1/cases", tags=["cases"])


# ── List cases ─────────────────────────────────────────────────────────

@router.get("", response_model=list[CaseResponse])
async def list_cases(
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_min_role("viewer")),
):
    """List all cases for the current user (or all cases for admin)."""
    if current_user.role == "admin":
        cases = await crud.get_all_cases(db)  # Admins see ALL cases
    else:
        cases = await crud.get_user_cases(db, current_user.id)

    results = []
    for c in cases:
        evidence_count = len(c.evidence) if c.evidence else 0
        results.append(CaseResponse(
            case_id=c.uid,
            name=c.name,
            reference=c.reference,
            description=c.description,
            status=c.status,
            priority=c.priority,
            classification=c.classification,
            evidence_count=evidence_count,
            created_at=c.created_at,
        ))
    return results


# ── Create case ────────────────────────────────────────────────────────

@router.post("", response_model=CaseResponse, status_code=status.HTTP_201_CREATED)
async def create_case(
    req: CaseCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_min_role("examiner")),
):
    """Create a new forensic case."""
    case = await crud.create_case(
        db,
        name=req.name,
        reference=req.reference,
        description=req.description,
        priority=req.priority,
        classification=req.classification,
        created_by=current_user.id,
    )

    return CaseResponse(
        case_id=case.uid,
        name=case.name,
        reference=case.reference,
        description=case.description,
        status=case.status,
        priority=case.priority,
        classification=case.classification,
        evidence_count=0,
        created_at=case.created_at,
    )


# ── Get case ───────────────────────────────────────────────────────────

@router.get("/{case_id}", response_model=CaseResponse)
async def get_case(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_min_role("viewer")),
):
    """Get case details by ID."""
    case = await crud.get_case_by_uid(db, case_id)
    if not case:
        raise HTTPException(404, "Case not found")

    if case.created_by != current_user.id and current_user.role not in ("admin", "reviewer"):
        raise HTTPException(403, "Access denied to this case")

    evidence_count = len(case.evidence) if case.evidence else 0

    return CaseResponse(
        case_id=case.uid,
        name=case.name,
        reference=case.reference,
        description=case.description,
        status=case.status,
        priority=case.priority,
        classification=case.classification,
        evidence_count=evidence_count,
        created_at=case.created_at,
    )


# ── Update case ────────────────────────────────────────────────────────

@router.put("/{case_id}", response_model=CaseResponse)
async def update_case(
    case_id: str,
    req: CaseUpdate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_min_role("examiner")),
):
    """Update case metadata (name, status, priority, description)."""
    case = await crud.get_case_by_uid(db, case_id)
    if not case:
        raise HTTPException(404, "Case not found")

    if case.created_by != current_user.id and current_user.role != "admin":
        raise HTTPException(403, "Only the case creator or admin can update")

    # Apply updates
    if req.name is not None:
        case.name = req.name
    if req.status is not None:
        case.status = req.status.value
    if req.priority is not None:
        case.priority = req.priority
    if req.description is not None:
        case.description = req.description

    await db.flush()

    evidence_count = len(case.evidence) if case.evidence else 0

    return CaseResponse(
        case_id=case.uid,
        name=case.name,
        reference=case.reference,
        description=case.description,
        status=case.status,
        priority=case.priority,
        classification=case.classification,
        evidence_count=evidence_count,
        created_at=case.created_at,
    )


# ── Add evidence to case ──────────────────────────────────────────────

from pydantic import BaseModel

class AddEvidenceRequest(BaseModel):
    analysis_id: str


class EvidenceResponse(BaseModel):
    case_id: str
    analysis_id: str
    message: str


@router.post("/{case_id}/evidence", response_model=EvidenceResponse, status_code=201)
async def add_evidence(
    case_id: str,
    req: AddEvidenceRequest,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_min_role("examiner")),
):
    """Link an analysis result to a case as evidence."""
    case = await crud.get_case_by_uid(db, case_id)
    if not case:
        raise HTTPException(404, "Case not found")

    if case.created_by != current_user.id and current_user.role not in ("admin", "examiner"):
        raise HTTPException(403, "Insufficient permissions")

    analysis = await crud.get_analysis_by_uid(db, req.analysis_id)
    if not analysis:
        raise HTTPException(404, "Analysis not found")

    await crud.add_evidence_to_case(db, case.id, analysis.id, current_user.id)

    return EvidenceResponse(
        case_id=case.uid,
        analysis_id=analysis.uid,
        message="Evidence linked to case",
    )


# ── Case notes ─────────────────────────────────────────────────────────

@router.get("/{case_id}/notes", response_model=list[CaseNoteResponse])
async def get_notes(
    case_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_min_role("viewer")),
):
    """Get all notes for a case."""
    case = await crud.get_case_by_uid(db, case_id)
    if not case:
        raise HTTPException(404, "Case not found")

    if case.created_by != current_user.id and current_user.role not in ("admin", "reviewer"):
        raise HTTPException(403, "Access denied")

    notes = await crud.get_case_notes(db, case.id)
    return [
        CaseNoteResponse(
            id=n.id,
            author=n.author.email if n.author else f"user:{n.author_id}",  # Use joined author
            content=n.content,
            created_at=n.created_at,
        )
        for n in notes
    ]


@router.post("/{case_id}/notes", response_model=CaseNoteResponse, status_code=201)
async def add_note(
    case_id: str,
    req: CaseNoteCreate,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_min_role("examiner")),
):
    """Add a note to a case."""
    case = await crud.get_case_by_uid(db, case_id)
    if not case:
        raise HTTPException(404, "Case not found")

    if case.created_by != current_user.id and current_user.role not in ("admin", "examiner"):
        raise HTTPException(403, "Insufficient permissions")

    note = await crud.add_case_note(db, case.id, current_user.id, req.content)

    return CaseNoteResponse(
        id=note.id,
        author=current_user.email,
        content=note.content,
        created_at=note.created_at,
    )
