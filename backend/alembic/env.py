"""Alembic environment configuration for SecureSight.

Loads the database URL from the app config (which reads from env vars).
Supports both online (connected to DB) and offline (SQL generation) modes.
Uses the app's SQLAlchemy Base.metadata for autogenerate support.
"""

from logging.config import fileConfig

from alembic import context
from sqlalchemy import engine_from_config, pool, create_engine

# ── Load app config + models ──────────────────────────────────────────
# This import causes all model classes to register with Base.metadata,
# which is required for autogenerate to detect schema changes.
import sys
import os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from app.config import settings
from app.database import Base
from app.db import models  # noqa: F401 — triggers model registration

# ── Alembic Config ────────────────────────────────────────────────────
config = context.config

# Interpret the alembic.ini file for Python logging
if config.config_file_name is not None:
    fileConfig(config.config_file_name)

# SQLAlchemy metadata for autogenerate
target_metadata = Base.metadata

# Convert async URL to sync for Alembic (it runs synchronously)
_sync_url = (
    settings.DATABASE_URL
    .replace("+asyncpg", "+psycopg2")
    .replace("+aiosqlite", "")
)


def run_migrations_offline() -> None:
    """Run migrations in 'offline' mode.

    Generates SQL statements without connecting to the database.
    Useful for generating migration SQL to apply manually.
    """
    context.configure(
        url=_sync_url,
        target_metadata=target_metadata,
        literal_binds=True,
        dialect_opts={"paramstyle": "named"},
        compare_type=True,
        compare_server_default=True,
    )

    with context.begin_transaction():
        context.run_migrations()


def run_migrations_online() -> None:
    """Run migrations in 'online' mode.

    Creates a connection to the database and applies migrations directly.
    """
    connectable = create_engine(_sync_url, poolclass=pool.NullPool)

    with connectable.connect() as connection:
        context.configure(
            connection=connection,
            target_metadata=target_metadata,
            compare_type=True,
            compare_server_default=True,
        )

        with context.begin_transaction():
            context.run_migrations()


if context.is_offline_mode():
    run_migrations_offline()
else:
    run_migrations_online()
