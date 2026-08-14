"""MinIO / S3-compatible object storage client with presigned URL caching."""

from __future__ import annotations

import io
from functools import lru_cache
from pathlib import Path
from datetime import timedelta

from minio import Minio
from minio.error import S3Error

from app.config import settings


def _get_client() -> Minio:
    return Minio(
        settings.MINIO_ENDPOINT,
        access_key=settings.MINIO_ACCESS_KEY,
        secret_key=settings.MINIO_SECRET_KEY,
        secure=settings.MINIO_SECURE,
    )


def ensure_bucket() -> None:
    """Create the bucket if it does not exist (idempotent)."""
    client = _get_client()
    if not client.bucket_exists(settings.MINIO_BUCKET):
        client.make_bucket(settings.MINIO_BUCKET)


def upload_file(object_name: str, file_path: str, content_type: str = "application/octet-stream") -> str:
    """Upload a local file to MinIO. Returns the object name."""
    client = _get_client()
    client.fput_object(settings.MINIO_BUCKET, object_name, file_path, content_type=content_type)
    return object_name


def upload_bytes(object_name: str, data: bytes, content_type: str = "application/octet-stream") -> str:
    """Upload raw bytes to MinIO."""
    client = _get_client()
    client.put_object(
        settings.MINIO_BUCKET,
        object_name,
        io.BytesIO(data),
        length=len(data),
        content_type=content_type,
    )
    return object_name


def download_file(object_name: str, file_path: str) -> None:
    client = _get_client()
    client.fget_object(settings.MINIO_BUCKET, object_name, file_path)


def get_presigned_url(object_name: str, expires: timedelta | None = None) -> str:
    """Generate a presigned URL for direct browser access (default: 1 hour)."""
    client = _get_client()
    return client.presigned_get_object(
        settings.MINIO_BUCKET,
        object_name,
        expires=expires or timedelta(hours=1),
    )


def delete_object(object_name: str) -> None:
    client = _get_client()
    client.remove_object(settings.MINIO_BUCKET, object_name)


# ── Convenience wrappers for the analysis pipeline ─────────────────────

def upload_analysis_file(evidence_id: str, file_path: str, content_type: str = "application/octet-stream") -> str:
    """Upload an evidence file to MinIO under evidence/<evidence_id>/original/."""
    filename = Path(file_path).name
    object_name = f"evidence/{evidence_id}/original/{filename}"
    return upload_file(object_name, file_path, content_type)


def upload_report(analysis_uid: str, report_path: str) -> str:
    """Upload the PDF report to MinIO under reports/<analysis_uid>/."""
    object_name = f"reports/{analysis_uid}/report.pdf"
    return upload_file(object_name, report_path, content_type="application/pdf")


def upload_heatmap(analysis_uid: str, heatmap_path: str, heatmap_type: str) -> str:
    """Upload a heatmap image to MinIO under heatmaps/<analysis_uid>/."""
    filename = Path(heatmap_path).name
    object_name = f"heatmaps/{analysis_uid}/{heatmap_type}_{filename}"
    return upload_file(object_name, heatmap_path, content_type="image/png")


def store_if_available(func, *args, **kwargs) -> str | None:
    """Try to store in MinIO; return None and log if MinIO is unreachable.

    This allows the system to degrade gracefully: if MinIO is down,
    files remain on local disk and the pipeline still completes.
    """
    import logging
    logger = logging.getLogger(__name__)
    try:
        ensure_bucket()
        return func(*args, **kwargs)
    except Exception as e:
        logger.warning(f"MinIO storage unavailable, falling back to local: {e}")
        return None
