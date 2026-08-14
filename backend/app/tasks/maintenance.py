"""Maintenance tasks -- periodic cleanup and housekeeping.

Scheduled via Celery Beat (see celery_app.py beat_schedule).
Run without blocking the main analysis queue.
"""

from __future__ import annotations

import logging
import shutil
from datetime import datetime, timezone, timedelta

from app.tasks.celery_app import celery_app
from app.config import settings

logger = logging.getLogger(__name__)

# Local files eligible for deletion after this many hours
# MinIO already has the evidence; local copies are just for active processing
_LOCAL_RETENTION_HOURS: int = 24


@celery_app.task(name="app.tasks.maintenance.cleanup_local_files", bind=False)
def cleanup_local_files() -> dict:
    """Delete local upload/output dirs for analyses older than 24 hours.

    Strategy:
    - Walk UPLOAD_DIR and OUTPUT_DIR for subdirectories
    - Delete any subdir older than _LOCAL_RETENTION_HOURS (by mtime)
    - Log bytes reclaimed and directories removed
    - NEVER touches MinIO -- only local /tmp disk

    Runs every hour via Celery Beat. At 500MB max upload, a single
    day of analyses without cleanup could consume 50+ GB.
    """
    upload_root = settings.UPLOAD_DIR
    output_root = settings.OUTPUT_DIR
    cutoff_ts = (datetime.now(timezone.utc) - timedelta(hours=_LOCAL_RETENTION_HOURS)).timestamp()

    removed_dirs = 0
    freed_bytes = 0
    errors = 0

    for root_dir in (upload_root, output_root):
        if not root_dir.exists():
            continue

        for entry in root_dir.iterdir():
            if not entry.is_dir():
                continue

            try:
                if entry.stat().st_mtime < cutoff_ts:
                    dir_size = sum(f.stat().st_size for f in entry.rglob("*") if f.is_file())
                    shutil.rmtree(entry, ignore_errors=False)
                    removed_dirs += 1
                    freed_bytes += dir_size
                    logger.info(f"Cleanup: removed {entry} ({dir_size / 1024:.1f} KB)")
            except PermissionError as e:
                logger.warning(f"Cleanup: permission denied on {entry}: {e}")
                errors += 1
            except Exception as e:
                logger.error(f"Cleanup: failed to remove {entry}: {e}")
                errors += 1

    freed_mb = freed_bytes / (1024 * 1024)
    logger.info(
        f"Local file cleanup complete: {removed_dirs} dirs removed, "
        f"{freed_mb:.1f} MB freed, {errors} errors"
    )
    return {
        "status": "ok",
        "dirs_removed": removed_dirs,
        "freed_mb": round(freed_mb, 2),
        "errors": errors,
        "cutoff_hours": _LOCAL_RETENTION_HOURS,
    }