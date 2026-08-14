"""Core Analysis API — /analyze, /results, /history, /health endpoints.

All analysis routes require authentication via JWT or API key.
The /health endpoint is public.
"""

from __future__ import annotations

import logging
import os
import time
import asyncio
import aiofiles
from datetime import datetime, timezone
from pathlib import Path

from fastapi import APIRouter, Depends, File, HTTPException, Request, UploadFile, Query
from fastapi.responses import FileResponse
from sqlalchemy.ext.asyncio import AsyncSession

from app.config import settings
from app.database import get_db
from app.db import crud
from app.db.models import User
from app.deps import require_min_role
from app.forensic.chain_of_custody import generate_evidence_id
from app.schemas import (
    AnalysisResponse, AnalysisSummary, HealthResponse, HistoryResponse,
    PipelineScore, AnalysisStatus, MediaType,
)
from app.security import compute_file_hash, sanitize_filename

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/api/v1", tags=["analysis"])

_start_time = time.time()


# ── Magic-byte signatures ──────────────────────────────────────────────
# Maps (offset, magic_bytes) → media_type it proves.
# offset is where in the file to look (most are 0, RIFF-based files differ).
_IMAGE_MAGIC: list[tuple[int, bytes]] = [
    (0, b"\xff\xd8\xff"),           # JPEG
    (0, b"\x89PNG\r\n\x1a\n"),      # PNG
    (0, b"RIFF"),                   # WebP container (checked with WEBP below)
    (0, b"BM"),                     # BMP
    (0, b"II*\x00"),                # TIFF little-endian
    (0, b"MM\x00*"),                # TIFF big-endian
]
_VIDEO_MAGIC: list[tuple[int, bytes]] = [
    (4,  b"ftyp"),                  # MP4 / MOV / M4V (ISO base media)
    (0,  b"RIFF"),                  # AVI container
    (0,  b"\x1aE\xdf\xa3"),         # MKV / WebM (EBML header)
    (0,  b"FLV\x01"),               # FLV (just in case)
]

def _validate_magic_bytes(file_path: str, media_type: str) -> str | None:
    """Read first 16 bytes of file and verify against known magic signatures.

    Returns an error message string on failure, None on success.
    This runs in a thread (blocking I/O is fine here since it's off the event loop).
    """
    try:
        with open(file_path, "rb") as fh:
            header = fh.read(16)
    except OSError as e:
        return f"Cannot read uploaded file: {e}"

    if len(header) < 4:
        return "File is too small to be a valid image or video"

    if media_type == "image":
        # Special case: RIFF container could be WebP — check bytes 8-12
        if header[:4] == b"RIFF" and header[8:12] == b"WEBP":
            return None  # Valid WebP
        for offset, magic in _IMAGE_MAGIC:
            if offset == 0 and header[:len(magic)] == magic and magic != b"RIFF":
                return None  # Known image magic
        return (
            "File content does not match an allowed image format. "
            "Supported: JPEG, PNG, WebP, BMP, TIFF"
        )

    if media_type == "video":
        # RIFF with AVI list
        if header[:4] == b"RIFF":
            return None  # AVI
        for offset, magic in _VIDEO_MAGIC:
            chunk = header[offset:offset + len(magic)]
            if chunk == magic:
                return None
        return (
            "File content does not match an allowed video format. "
            "Supported: MP4, AVI, MOV, WebM, MKV"
        )

    return "Unknown media type"


# ── Health (PUBLIC — no auth required) ─────────────────────────────────


@router.get("/health", response_model=HealthResponse)
async def health():
    gpu_available = False
    gpu_name = None
    device = settings.resolved_device
    try:
        import torch
        gpu_available = torch.cuda.is_available()
        if gpu_available:
            gpu_name = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    return HealthResponse(
        status="healthy",
        version=settings.APP_VERSION,
        gpu_available=gpu_available,
        gpu_name=gpu_name,
        device=device,
        uptime_seconds=time.time() - _start_time,
    )


# ── Analyze (AUTHENTICATED — examiner+) ───────────────────────────────

@router.post("/analyze", response_model=AnalysisResponse)
async def analyze(
    request: Request,
    file: UploadFile = File(...),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_min_role("examiner")),
):
    """Upload media file and run full analysis pipeline."""
    # Validate MIME type from HTTP header (first gate)
    content_type = file.content_type or ""
    is_image = content_type in settings.ALLOWED_IMAGE_TYPES
    is_video = content_type in settings.ALLOWED_VIDEO_TYPES

    if not is_image and not is_video:
        raise HTTPException(400, f"Unsupported file type: {content_type}")

    media_type = "image" if is_image else "video"

    # ── Sanitize filename (CRITICAL: prevents path traversal) ─────────
    safe_filename = sanitize_filename(file.filename or "upload")

    # Save uploaded file with STREAMING (don't buffer into RAM)
    evidence_id = generate_evidence_id(settings.EVIDENCE_ID_PREFIX)
    upload_dir = settings.UPLOAD_DIR / evidence_id
    upload_dir.mkdir(parents=True, exist_ok=True)
    file_path = str(upload_dir / safe_filename)

    # Verify resolved path is inside upload_dir (belt + suspenders)
    resolved = Path(file_path).resolve()
    if not str(resolved).startswith(str(upload_dir.resolve())):
        raise HTTPException(400, "Invalid filename")

    total_bytes = 0
    oversized = False
    async with aiofiles.open(file_path, "wb") as f:
        while True:
            chunk = await file.read(1024 * 1024)  # 1 MB chunks
            if not chunk:
                break
            total_bytes += len(chunk)
            if total_bytes > settings.max_upload_bytes:
                # Flag it and break — let async with close the file cleanly
                oversized = True
                break
            await f.write(chunk)

    # Unlink partial file AFTER async with exits (file is fully closed now)
    if oversized:
        try:
            os.unlink(file_path)
        except OSError:
            pass
        raise HTTPException(413, f"File exceeds {settings.MAX_UPLOAD_MB}MB limit")

    # Magic byte validation (second gate — the real check)
    # HTTP Content-Type is set by the client and trivially spoofable.
    # The file's actual bytes cannot be faked.
    _magic_error = await asyncio.to_thread(_validate_magic_bytes, file_path, media_type)
    if _magic_error:
        try:
            os.unlink(file_path)
        except OSError:
            pass
        raise HTTPException(400, _magic_error)

    sha256 = await asyncio.to_thread(compute_file_hash, file_path, "sha256")
    sha512 = await asyncio.to_thread(compute_file_hash, file_path, "sha512")

    # Persist to MinIO for durable storage without blocking event loop
    from app.storage import store_if_available, upload_analysis_file
    await asyncio.to_thread(store_if_available, upload_analysis_file, evidence_id, file_path, content_type)

    # Get client IP for custody log
    client_ip = request.client.host if request.client else "unknown"

    # Create analysis record
    analysis = await crud.create_analysis(
        db,
        evidence_id=evidence_id,
        user_id=current_user.id,
        filename=safe_filename,
        media_type=media_type,
        mime_type=content_type,
        file_size=total_bytes,
        sha256=sha256,
        sha512=sha512,
        storage_path=file_path,
        status="processing",
    )

    # Chain of custody — evidence intake
    await crud.add_custody_log(
        db, analysis.id,
        action="evidence_intake",
        actor=current_user.email,
        ip_address=client_ip,
        details=f"File uploaded: {safe_filename} ({total_bytes} bytes)",
        file_hash=sha256,
    )

    # Commit evidence intake BEFORE dispatching to Celery.
    # If Redis is down and .delay() throws, the custody log must already
    # be durable — the file is on disk and the DB record must reflect that.
    await db.commit()

    # ── Run the full pipeline ─────────────────────────────────────────
    # Dispatch to Celery to prevent blocking the event loop
    from app.tasks.analysis import run_analysis_task
    run_analysis_task.delay(analysis.id, file_path, media_type)

    # Build response (Processing)
    return AnalysisResponse(
        analysis_id=analysis.uid,
        evidence_id=evidence_id,
        status=AnalysisStatus.PROCESSING,
        filename=safe_filename,
        media_type=MediaType(media_type),
        sha256=sha256,
        created_at=analysis.created_at,
    )


# ── Results (AUTHENTICATED — viewer+) ─────────────────────────────────

@router.get("/results/{analysis_id}", response_model=AnalysisResponse)
async def get_results(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_min_role("viewer")),
):
    analysis = await crud.get_analysis_by_uid(db, analysis_id)
    if not analysis:
        raise HTTPException(404, "Analysis not found")

    # Users can only see their own analyses unless admin/reviewer
    if analysis.user_id != current_user.id and current_user.role not in ("admin", "reviewer"):
        raise HTTPException(403, "Access denied to this analysis")

    pipe_results = await crud.get_pipeline_results(db, analysis.id)
    heatmaps = await crud.get_heatmaps(db, analysis.id)

    return AnalysisResponse(
        analysis_id=analysis.uid,
        evidence_id=analysis.evidence_id,
        status=AnalysisStatus(analysis.status),
        filename=analysis.filename,
        media_type=MediaType(analysis.media_type),
        sha256=analysis.sha256,
        overall_score=analysis.overall_score,
        verdict=analysis.verdict,
        pipeline_scores=[
            PipelineScore(
                pipeline=pr.pipeline, tier=pr.tier, score=pr.score,
                confidence=pr.confidence, execution_ms=pr.execution_ms,
                details=pr.details or {},
            )
            for pr in pipe_results
        ],
        heatmap_urls=[f"/api/v1/results/{analysis.uid}/heatmap?type={h.heatmap_type}" for h in heatmaps],
        exif_data=analysis.exif_data,
        report_url=f"/api/v1/results/{analysis.uid}/report" if analysis.report_path else None,
        created_at=analysis.created_at,
        completed_at=analysis.completed_at,
    )


@router.get("/results/{analysis_id}/report")
async def download_report(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_min_role("viewer")),
):
    analysis = await crud.get_analysis_by_uid(db, analysis_id)
    if not analysis or not analysis.report_path:
        raise HTTPException(404, "Report not found")

    if analysis.user_id != current_user.id and current_user.role not in ("admin", "reviewer"):
        raise HTTPException(403, "Access denied")

    if not os.path.exists(analysis.report_path):
        raise HTTPException(404, "Report file missing")
    return FileResponse(analysis.report_path, media_type="application/pdf",
                        filename=f"SecureSight_Report_{analysis_id}.pdf")


@router.get("/results/{analysis_id}/heatmap")
async def get_heatmap(
    analysis_id: str,
    type: str = "gradcam_efficientnet",
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_min_role("viewer")),
):
    analysis = await crud.get_analysis_by_uid(db, analysis_id)
    if not analysis:
        raise HTTPException(404, "Analysis not found")

    if analysis.user_id != current_user.id and current_user.role not in ("admin", "reviewer"):
        raise HTTPException(403, "Access denied")

    heatmaps = await crud.get_heatmaps(db, analysis.id)
    match = [h for h in heatmaps if h.heatmap_type == type]
    if not match or not os.path.exists(match[0].storage_path):
        raise HTTPException(404, f"Heatmap '{type}' not found")

    return FileResponse(match[0].storage_path, media_type="image/png")


@router.get("/results/{analysis_id}/original")
async def get_original_file(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_min_role("viewer")),
):
    """Serve the original uploaded evidence file for the forensic viewer."""
    analysis = await crud.get_analysis_by_uid(db, analysis_id)
    if not analysis:
        raise HTTPException(404, "Analysis not found")

    if analysis.user_id != current_user.id and current_user.role not in ("admin", "reviewer"):
        raise HTTPException(403, "Access denied")

    if not analysis.storage_path or not os.path.exists(analysis.storage_path):
        raise HTTPException(404, "Original file not found on disk")

    # Determine media type from mime_type stored on record
    media_type = analysis.mime_type or "application/octet-stream"
    return FileResponse(
        analysis.storage_path,
        media_type=media_type,
        filename=analysis.filename,
    )


@router.get("/history", response_model=HistoryResponse)
async def history(
    page: int = Query(1, ge=1),
    per_page: int = Query(20, ge=1, le=100),
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_min_role("viewer")),
):
    total, items = await crud.get_user_analyses(db, user_id=current_user.id, page=page, per_page=per_page)

    return HistoryResponse(
        total=total, page=page, per_page=per_page,
        analyses=[
            AnalysisSummary(
                analysis_id=a.uid, evidence_id=a.evidence_id, filename=a.filename,
                media_type=a.media_type, status=a.status,
                overall_score=a.overall_score, verdict=a.verdict,
                created_at=a.created_at,
            )
            for a in items
        ],
    )


@router.get("/results/{analysis_id}/custody")
async def get_custody_log(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_min_role("viewer")),
):
    analysis = await crud.get_analysis_by_uid(db, analysis_id)
    if not analysis:
        raise HTTPException(404, "Analysis not found")

    if analysis.user_id != current_user.id and current_user.role not in ("admin", "reviewer"):
        raise HTTPException(403, "Access denied")

    logs = await crud.get_custody_logs(db, analysis.id)
    return [
        {"action": l.action, "actor": l.actor, "ip_address": l.ip_address,
         "details": l.details, "file_hash": l.file_hash, "timestamp": str(l.timestamp)}
        for l in logs
    ]

@router.delete("/results/{analysis_id}")
async def delete_analysis_endpoint(
    analysis_id: str,
    db: AsyncSession = Depends(get_db),
    current_user: User = Depends(require_min_role("examiner")),
):
    """Delete an analysis and enqueue background cleanup of MinIO storage."""
    analysis = await crud.get_analysis_by_uid(db, analysis_id)
    if not analysis:
        raise HTTPException(404, "Analysis not found")
        
    if analysis.user_id != current_user.id and current_user.role != "admin":
        raise HTTPException(403, "Access denied: can only delete your own analyses unless admin")

    # Queue storage cleanup BEFORE deleting from DB
    # (need evidence_id and uid which will be gone after delete)
    from app.tasks.analysis import delete_analysis_files_task
    delete_analysis_files_task.delay(analysis.uid, analysis.evidence_id)

    # Delete from DB — pass pre-fetched object directly (Bug 3 fix)
    await crud.delete_analysis(db, analysis)

    return {"status": "ok", "detail": "Analysis deleted and storage cleanup queued"}
