"""Celery application instance.

Configured to use Redis as both broker and result backend.
"""

from celery import Celery
from celery.schedules import crontab

from app.config import settings

celery_app = Celery(
    "securesight",
    broker=settings.REDIS_URL,
    backend=settings.REDIS_URL,
    include=["app.tasks.analysis", "app.tasks.maintenance"],
)

celery_app.conf.update(
    task_serializer="json",
    result_serializer="json",
    accept_content=["json"],
    timezone="UTC",
    enable_utc=True,
    task_track_started=True,
    task_acks_late=True,
    worker_prefetch_multiplier=1,  # One task at a time per worker (GPU bound)
    task_soft_time_limit=600,      # 10 min soft limit
    task_time_limit=900,           # 15 min hard limit
    broker_connection_retry_on_startup=True,  # Suppress Celery 5.x deprecation warning

    # ── Celery Beat schedule (periodic tasks) ──────────────────────────
    beat_schedule={
        "cleanup-local-files-hourly": {
            "task": "app.tasks.maintenance.cleanup_local_files",
            "schedule": crontab(minute=0),  # Every hour on the hour
            "options": {"expires": 3600},   # Skip if still pending after 1 hour
        },
    },
)
