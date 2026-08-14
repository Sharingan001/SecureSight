"""FastAPI Application — main entry point.

Mounts all routers, configures CORS, lifespan (model preloading),
static file serving, and audit logging middleware.
"""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles

from app.config import settings

logger = logging.getLogger("securesight")
logging.basicConfig(level=logging.INFO if not settings.DEBUG else logging.DEBUG)


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Startup: init DB tables, ensure storage. Shutdown: cleanup."""
    from app.database import init_db

    if settings.DEBUG:
        # Dev mode: auto-create tables (Alembic not required)
        await init_db()
        logger.warning(
            "DEV MODE: Tables created via create_all(). "
            "In production, use: alembic upgrade head"
        )
    else:
        logger.info("Production mode: expecting Alembic-managed schema (alembic upgrade head)")

    # Ensure output/upload dirs
    settings.UPLOAD_DIR.mkdir(parents=True, exist_ok=True)
    settings.OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    logger.info(f"SecureSight {settings.APP_VERSION} started (device={settings.resolved_device})")
    logger.info(f"Auth enforcement: {'ENABLED' if settings.REQUIRE_AUTH else 'DISABLED (dev mode)'}")

    yield  # App runs here

    logger.info("SecureSight shutting down")


app = FastAPI(
    title=settings.APP_NAME,
    version=settings.APP_VERSION,
    description="Forensic & Cyber-Lab Grade Deepfake + Media Integrity Detection Platform",
    lifespan=lifespan,
)

# ── CORS ───────────────────────────────────────────────────────────────
app.add_middleware(
    CORSMiddleware,
    allow_origins=settings.CORS_ORIGINS,
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Rate Limiting ──────────────────────────────────────────────────────
from slowapi import _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from app.deps import limiter

app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)


# ── Audit Logging Middleware ───────────────────────────────────────────

@app.middleware("http")
async def audit_logging_middleware(request: Request, call_next) -> Response:
    """Log every API request for audit trail.

    This middleware captures the request method, path, response status,
    client IP, and user-agent for every request. The data is stored in
    the audit_logs table.
    """
    start = time.perf_counter()
    response: Response = await call_next(request)
    elapsed_ms = int((time.perf_counter() - start) * 1000)

    # Only audit API routes (skip static files)
    path = request.url.path
    if path.startswith("/api/"):
        # Extract user info from auth header if present (best-effort, non-blocking)
        user_email = ""
        user_id = None
        auth_header = request.headers.get("authorization", "")
        if auth_header.startswith("Bearer "):
            try:
                from app.security import decode_access_token
                payload = decode_access_token(auth_header[7:])
                user_email = payload.get("email", "")
            except (ValueError, Exception):
                pass

        # Fire-and-forget audit log (don't block the response)
        try:
            from app.database import async_session
            from app.db.crud import add_audit_log
            async with async_session() as db:
                await add_audit_log(
                    db,
                    user_email=user_email,
                    endpoint=path,
                    method=request.method,
                    status_code=response.status_code,
                    ip_address=request.client.host if request.client else "",
                    user_agent=request.headers.get("user-agent", "")[:500],
                )
                await db.commit()
        except Exception:
            # Audit logging must never break the request
            logger.debug(f"Audit log failed for {request.method} {path}", exc_info=True)

    return response


# ── Routers ────────────────────────────────────────────────────────────
from app.api.routes import router as analysis_router
from app.api.auth import router as auth_router
from app.api.cases import router as cases_router

app.include_router(analysis_router)
app.include_router(auth_router)
app.include_router(cases_router)

# ── Static files (frontend) ───────────────────────────────────────────
frontend_dir = Path(__file__).resolve().parent.parent.parent / "frontend"
if frontend_dir.exists():
    app.mount("/", StaticFiles(directory=str(frontend_dir), html=True), name="frontend")
