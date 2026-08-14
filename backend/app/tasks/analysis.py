"""Celery task: Async analysis pipeline execution.

This task runs the full forensic analysis pipeline in a Celery worker,
decoupled from the HTTP request. The API enqueues the task and returns
immediately with status='queued'.

Uses SYNCHRONOUS SQLAlchemy (Celery workers don't run an asyncio event loop).
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from sqlalchemy import create_engine, select
from sqlalchemy.orm import Session, sessionmaker

from app.config import settings
from app.tasks.celery_app import celery_app

logger = logging.getLogger(__name__)

# ── Sync DB session for Celery workers ─────────────────────────────────
_sync_url = settings.DATABASE_URL.replace("+asyncpg", "+psycopg2").replace("+aiosqlite", "")
_sync_engine = create_engine(_sync_url, pool_size=5, pool_pre_ping=True)
SyncSession = sessionmaker(_sync_engine)

from celery.signals import worker_process_init

@worker_process_init.connect
def init_worker_db_pool(**kwargs):
    """Dispose of the engine pool inherited from the parent process.
    
    In Celery's prefork multiprocessing model, child processes inherit the parent's
    memory. If they share the parent's DB connection sockets, packets will interleave,
    causing SSL decryption failures and DB disconnects. Calling dispose() forces
    each worker to create its own clean connections.
    """
    _sync_engine.dispose()
    logger.info("Celery worker process initialized: DB engine disposed for clean prefork.")


# ── Exception classification for intelligent retry ─────────────────────

# Errors caused by the file/user — retrying wastes GPU cycles (will always fail)
_NON_RETRYABLE: tuple = (
    ValueError,
    AssertionError,
    AttributeError,      # e.g. NoneType has no attribute X on bad input
    NotImplementedError,
)

# File format errors — retrying makes no sense
_NON_RETRYABLE_NAMES: frozenset = frozenset({
    "UnidentifiedImageError",  # PIL: file is not an image
    "DecompressionBombError",  # PIL: oversized image
    "IsADirectoryError",
    "FileNotFoundError",
})


def _classify_exception_for_retry(exc: Exception, retry_count: int) -> tuple[bool, int]:
    """Classify an exception as retryable or not, with appropriate delay.

    Returns:
        (should_retry: bool, delay_seconds: int)

    Rules:
        - File/user errors (ValueError, PIL errors): NEVER retry — broken input
        - CUDA OOM: retry after 60s (transient GPU resource issue)
        - DB/network errors: retry with exponential backoff (30 * 2^retry)
        - Unknown: retry with base delay (give benefit of the doubt once)
    """
    exc_type_name = type(exc).__name__

    # Permanent failures — don't waste resources
    if isinstance(exc, _NON_RETRYABLE):
        return False, 0
    if exc_type_name in _NON_RETRYABLE_NAMES:
        return False, 0

    # CUDA Out of Memory — GPU is under pressure, back off then retry
    if "OutOfMemoryError" in exc_type_name or "CUDA out of memory" in str(exc):
        return True, 60  # Fixed 60s for GPU to recover

    # DB / network connection errors — exponential backoff
    exc_str = str(exc).lower()
    if any(k in exc_str for k in ("connection", "timeout", "operational", "network", "broker")):
        delay = min(30 * (2 ** retry_count), 300)  # Max 5 min
        return True, delay

    # Unknown error — retry once with base delay, then give up
    return True, 30


@celery_app.task(bind=True, max_retries=2, default_retry_delay=30)
def run_analysis_task(self, analysis_id: int, file_path: str, media_type: str):
    """Execute the full forensic analysis pipeline asynchronously.

    Args:
        analysis_id: DB primary key of the Analysis record
        file_path: Path to the uploaded file on shared storage
        media_type: "image" or "video"
    """
    from app.db.models import Analysis, PipelineResult, CustodyLog, Heatmap

    logger.info(f"[Task {self.request.id}] Starting analysis #{analysis_id} ({media_type})")

    with SyncSession() as db:
        # Load analysis record
        analysis = db.execute(
            select(Analysis).where(Analysis.id == analysis_id)
        ).scalars().first()

        if not analysis:
            logger.error(f"Analysis #{analysis_id} not found")
            return {"status": "error", "detail": "Analysis not found"}

        sha256 = analysis.sha256

        try:
            # Mark as processing
            analysis.status = "processing"
            db.commit()

            # Run preprocessing
            from app.pipeline.preprocessor import preprocess
            from app.pipeline.ensemble import run_all_pipelines
            from app.pipeline.explainability import generate_all_visuals

            preprocess_result = preprocess(file_path, media_type)

            # Custody log — analysis start
            db.add(CustodyLog(
                analysis_id=analysis.id,
                action="analysis_start",
                actor="celery_worker",
                details=f"Pipeline analysis initiated. Faces detected: {len(preprocess_result.face_crops)}",
                file_hash=sha256,
            ))
            db.commit()

            # Run all pipelines
            ensemble_result = run_all_pipelines(preprocess_result, file_path)

            # Generate visuals
            primary_face = preprocess_result.face_crops[0].image if preprocess_result.face_crops else None
            original_frame = preprocess_result.original_frames[0] if preprocess_result.original_frames else None

            visuals = []
            if original_frame is not None:
                visuals = generate_all_visuals(
                    analysis.uid, primary_face, original_frame, ensemble_result["results_raw"]
                )

            # Save pipeline results
            for ps in ensemble_result["pipeline_scores"]:
                db.add(PipelineResult(
                    analysis_id=analysis.id,
                    pipeline=ps["pipeline"],
                    tier=ps["tier"],
                    score=ps["score"],
                    confidence=ps["confidence"],
                    execution_ms=ps["execution_ms"],
                    details=ps["details"],
                ))

            # Save heatmaps
            for v in visuals:
                db.add(Heatmap(
                    analysis_id=analysis.id,
                    heatmap_type=v["type"],
                    storage_path=v["path"],
                ))

            # Generate PDF report
            from app.pipeline.report import generate_report
            report_dir = settings.OUTPUT_DIR / analysis.uid
            report_dir.mkdir(parents=True, exist_ok=True)
            report_path = str(report_dir / "report.pdf")

            custody_logs = db.execute(
                select(CustodyLog).where(CustodyLog.analysis_id == analysis.id)
                .order_by(CustodyLog.timestamp)
            ).scalars().all()

            custody_dicts = [
                {"timestamp": str(cl.timestamp), "action": cl.action, "actor": cl.actor, "file_hash": cl.file_hash}
                for cl in custody_logs
            ]

            generate_report(
                output_path=report_path,
                analysis_id=analysis.uid,
                evidence_id=analysis.evidence_id,
                filename=analysis.filename,
                sha256=sha256,
                sha512=analysis.sha512,
                overall_score=ensemble_result["overall_score"],
                verdict=ensemble_result["verdict"],
                pipeline_scores=ensemble_result["pipeline_scores"],
                custody_log=custody_dicts,
                heatmap_paths=[v["path"] for v in visuals],
            )

            # Persist report + heatmaps to MinIO
            from app.storage import store_if_available, upload_report as minio_upload_report, upload_heatmap
            store_if_available(minio_upload_report, analysis.uid, report_path)
            for v in visuals:
                store_if_available(upload_heatmap, analysis.uid, v["path"], v["type"])

            # Get EXIF data
            exif_data = None
            exif_result = ensemble_result["results_raw"].get("exif", {})
            if "exif_data" in exif_result:
                exif_data = exif_result["exif_data"]

            # Update analysis — success
            analysis.overall_score = ensemble_result["overall_score"]
            analysis.verdict = ensemble_result["verdict"]
            analysis.status = "completed"
            analysis.report_path = report_path
            analysis.exif_data = exif_data
            analysis.completed_at = datetime.now(timezone.utc)

            # Custody log — complete
            db.add(CustodyLog(
                analysis_id=analysis.id,
                action="analysis_complete",
                actor="celery_worker",
                details=f"Verdict: {ensemble_result['verdict']} (score: {ensemble_result['overall_score']:.1f})",
                file_hash=sha256,
            ))
            db.commit()

            logger.info(f"[Task {self.request.id}] Analysis #{analysis_id} completed: {ensemble_result['verdict']}")

            return {
                "status": "completed",
                "analysis_id": analysis.uid,
                "verdict": ensemble_result["verdict"],
                "score": ensemble_result["overall_score"],
            }

        except Exception as exc:
            logger.exception(f"[Task {self.request.id}] Analysis #{analysis_id} failed")
            db.rollback()

            # Update analysis — failed
            # Use a separate session so the rollback above doesn't affect this
            try:
                with SyncSession() as db2:
                    a = db2.execute(select(Analysis).where(Analysis.id == analysis_id)).scalars().first()
                    if a:
                        a.status = "failed"
                        a.completed_at = datetime.now(timezone.utc)
                        db2.add(CustodyLog(
                            analysis_id=analysis_id,
                            action="analysis_failed",
                            actor="celery_worker",
                            details=f"Error: {str(exc)[:500]}",
                            file_hash=sha256 or "",
                        ))
                        db2.commit()
            except Exception as inner_exc:
                # The failure handler itself failed. Log it but don't re-raise
                # to prevent masking the original exception.
                logger.error(
                    f"[Task {self.request.id}] Failed to write failure record for "
                    f"analysis #{analysis_id}: {inner_exc}"
                )

            # ── Intelligent retry classification ──────────────────────
            # NOT all failures are retryable. Retrying a broken file wastes
            # GPU cycles and queue slots — it will fail every time.
            should_retry, retry_delay = _classify_exception_for_retry(exc, self.request.retries)

            if should_retry and self.request.retries < self.max_retries:
                logger.warning(
                    f"[Task {self.request.id}] Retrying analysis #{analysis_id} "
                    f"(attempt {self.request.retries + 1}/{self.max_retries}) "
                    f"in {retry_delay}s: {type(exc).__name__}"
                )
                raise self.retry(exc=exc, countdown=retry_delay)

            if not should_retry:
                logger.error(
                    f"[Task {self.request.id}] Analysis #{analysis_id} NOT retrying: "
                    f"{type(exc).__name__} is a non-transient error (file/user problem)"
                )

            return {"status": "failed", "detail": str(exc)[:500]}


@celery_app.task(bind=True, max_retries=3, default_retry_delay=10)
def delete_analysis_files_task(self, analysis_uid: str, evidence_id: str):
    """Clean up MinIO objects for a deleted analysis.

    Evidence files are only deleted if NO other Analysis rows reference
    the same evidence_id — preventing data destruction for re-analyses.
    """
    try:
        from app.storage import _get_client
        from app.db.models import Analysis
        from sqlalchemy import func
        client = _get_client()

        # 1. Always delete reports (analysis-specific)
        report_objs = client.list_objects(
            settings.MINIO_BUCKET, prefix=f"reports/{analysis_uid}/", recursive=True
        )
        for obj in report_objs:
            client.remove_object(settings.MINIO_BUCKET, obj.object_name)

        # 2. Always delete heatmaps (analysis-specific)
        heatmap_objs = client.list_objects(
            settings.MINIO_BUCKET, prefix=f"heatmaps/{analysis_uid}/", recursive=True
        )
        for obj in heatmap_objs:
            client.remove_object(settings.MINIO_BUCKET, obj.object_name)

        # 3. Delete evidence ONLY if this was the last analysis referencing it
        with SyncSession() as db:
            remaining = db.execute(
                select(func.count())
                .select_from(Analysis)
                .where(Analysis.evidence_id == evidence_id)
            ).scalar() or 0

        if remaining == 0:
            evidence_objs = client.list_objects(
                settings.MINIO_BUCKET, prefix=f"evidence/{evidence_id}/", recursive=True
            )
            for obj in evidence_objs:
                client.remove_object(settings.MINIO_BUCKET, obj.object_name)
            logger.info(f"Evidence {evidence_id} deleted (no remaining analyses)")
        else:
            logger.info(
                f"Evidence {evidence_id} preserved: {remaining} other "
                f"analysis/analyses still reference it"
            )

        logger.info(f"Storage cleanup complete for analysis {analysis_uid}")
    except Exception as exc:
        logger.error(f"Storage cleanup failed for analysis {analysis_uid}: {exc}")
        if self.request.retries < self.max_retries:
            raise self.retry(exc=exc)
