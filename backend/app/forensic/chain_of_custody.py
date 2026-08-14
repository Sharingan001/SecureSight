"""Forensic — Chain of Custody Engine + Evidence Integrity."""

from __future__ import annotations

from datetime import datetime, timezone


def generate_evidence_id(prefix: str = "EV") -> str:
    """Generate unique evidence ID: EV-YYYYMMDD-XXXX."""
    import uuid
    now = datetime.now(timezone.utc)
    date_str = now.strftime("%Y%m%d")
    short_id = uuid.uuid4().hex[:4].upper()
    return f"{prefix}-{date_str}-{short_id}"


def create_custody_entry(
    action: str,
    actor: str,
    ip_address: str = "",
    details: str = "",
    file_hash: str = "",
) -> dict:
    """Create a custody log entry dict (to be inserted to DB)."""
    return {
        "action": action,
        "actor": actor,
        "ip_address": ip_address,
        "details": details,
        "file_hash": file_hash,
        "timestamp": datetime.now(timezone.utc),
    }


def verify_integrity(file_path: str, expected_hash: str) -> bool:
    """Re-hash file and compare with expected hash.  Returns True if match."""
    from app.security import compute_file_hash
    actual = compute_file_hash(file_path, algorithm="sha256")
    return actual == expected_hash
