"""Performance indexes — high-frequency query columns.

Revision ID: 002_indexes
Revises: 001_initial
Create Date: 2026-08-14

Adds explicit indexes for the 8 most-queried columns that were
missing from the initial migration. Without these, every history
page, pipeline result fetch, and custody log lookup is a full table scan.
"""

from typing import Sequence, Union
from alembic import op

revision: str = "002_indexes"
down_revision: Union[str, None] = "001_initial"
branch_labels: Union[str, Sequence[str], None] = None
depends_on: Union[str, Sequence[str], None] = None


def upgrade() -> None:
    # analyses: supports /history pagination and status checks
    op.create_index("ix_analyses_user_id",    "analyses", ["user_id"])
    op.create_index("ix_analyses_status",     "analyses", ["status"])
    op.create_index("ix_analyses_created_at", "analyses", ["created_at"])
    op.create_index("ix_analyses_verdict",    "analyses", ["verdict"])

    # pipeline_results: fetched on every /results/{id} call
    op.create_index("ix_pipeline_results_analysis_id", "pipeline_results", ["analysis_id"])

    # custody_logs: fetched and ordered on every /results/{id}/custody call
    op.create_index("ix_custody_logs_analysis_id", "custody_logs", ["analysis_id"])
    op.create_index("ix_custody_logs_timestamp",   "custody_logs", ["timestamp"])

    # heatmaps: fetched on every /results/{id} call
    op.create_index("ix_heatmaps_analysis_id", "heatmaps", ["analysis_id"])

    # audit_logs: time-range queries for audit reporting
    op.create_index("ix_audit_logs_timestamp",  "audit_logs", ["timestamp"])
    op.create_index("ix_audit_logs_user_email", "audit_logs", ["user_email"])

    # cases: list queries filtered by creator
    op.create_index("ix_cases_created_by", "cases", ["created_by"])
    op.create_index("ix_cases_status",     "cases", ["status"])


def downgrade() -> None:
    op.drop_index("ix_cases_status",                 table_name="cases")
    op.drop_index("ix_cases_created_by",             table_name="cases")
    op.drop_index("ix_audit_logs_user_email",        table_name="audit_logs")
    op.drop_index("ix_audit_logs_timestamp",         table_name="audit_logs")
    op.drop_index("ix_heatmaps_analysis_id",         table_name="heatmaps")
    op.drop_index("ix_custody_logs_timestamp",       table_name="custody_logs")
    op.drop_index("ix_custody_logs_analysis_id",     table_name="custody_logs")
    op.drop_index("ix_pipeline_results_analysis_id", table_name="pipeline_results")
    op.drop_index("ix_analyses_verdict",             table_name="analyses")
    op.drop_index("ix_analyses_created_at",          table_name="analyses")
    op.drop_index("ix_analyses_status",              table_name="analyses")
    op.drop_index("ix_analyses_user_id",             table_name="analyses")
