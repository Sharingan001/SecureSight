"""Tier 2 — EXIF Metadata Forensic Analyzer.

Extracts and inspects metadata for signs of manipulation:
software tags, stripped data, timestamp anomalies, thumbnail mismatches.
"""

from __future__ import annotations

import io
import time
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from PIL import Image


def _sanitize_exif_value(key: str, val: str, max_val_len: int = 512) -> tuple[str, str]:
    """Sanitize a raw EXIF key/value pair before storage.

    EXIF data is attacker-controlled (embedded in the uploaded file).
    Raw values must be sanitized before being stored in the DB JSON column
    and returned via the API.

    Sanitization rules:
      - Strip null bytes and ASCII control chars (breaks JSON parsers)
      - Strip HTML tags (defense-in-depth; frontend uses textContent anyway)
      - Truncate key to 128 chars
      - Truncate value to 512 chars
      - Skip binary-looking values (high proportion of non-printable chars)
    """
    import re

    # Strip null bytes and control characters (keep tab/newline for readability)
    _ctrl = re.compile(r"[\x00-\x08\x0b\x0c\x0e-\x1f\x7f]")
    key = _ctrl.sub("", str(key))[:128]
    val = _ctrl.sub("", str(val))

    # Strip HTML tags (prevent stored XSS in any future template that uses innerHTML)
    val = re.sub(r"<[^>]{0,200}>", "", val)

    # Skip values that look binary (>30% non-printable chars)
    printable_ratio = sum(1 for c in val if c.isprintable()) / max(len(val), 1)
    if printable_ratio < 0.7:
        val = "[binary data]"

    # Truncate
    val = val[:max_val_len]

    return key, val


def _extract_exif(file_path: str) -> dict[str, Any]:
    """Extract all EXIF metadata from image file.

    All values are sanitized before return — EXIF is attacker-controlled
    (embedded in the uploaded file) and must not be stored/returned raw.
    """
    _MAX_FIELDS = 100  # Cap total fields to prevent memory exhaustion
    data: dict[str, Any] = {}
    try:
        import exifread
        with open(file_path, "rb") as f:
            tags = exifread.process_file(f, details=True)
        for key, val in tags.items():
            if len(data) >= _MAX_FIELDS:
                break
            k, v = _sanitize_exif_value(key, str(val))
            if k:
                data[k] = v
    except Exception:
        pass

    # Also try Pillow for additional fields
    try:
        pil_img = Image.open(file_path)
        exif_raw = pil_img.getexif()
        if exif_raw:
            for tag_id, value in exif_raw.items():
                if len(data) >= _MAX_FIELDS:
                    break
                tag_name = f"PIL_{tag_id}"
                try:
                    from PIL.ExifTags import TAGS
                    tag_name = TAGS.get(tag_id, f"Unknown_{tag_id}")
                except Exception:
                    pass
                k, v = _sanitize_exif_value(f"pil_{tag_name}", str(value))
                if k:
                    data[k] = v
    except Exception:
        pass

    return data


def _check_software_tags(exif: dict[str, Any]) -> list[str]:
    """Check for known editing software signatures."""
    suspicious_software = [
        "photoshop", "gimp", "faceapp", "faceswap", "deepfacelab",
        "aftereffects", "premiere", "davinci", "lightroom",
        "snapseed", "picsart", "facetune", "reface",
    ]
    findings = []
    for key, val in exif.items():
        val_lower = val.lower()
        for sw in suspicious_software:
            if sw in val_lower:
                findings.append(f"Editing software detected: {val} (field: {key})")
                break
    return findings


def _check_thumbnail_mismatch(file_path: str) -> dict[str, Any]:
    """Compare embedded thumbnail with actual image — mismatch = tampering."""
    result = {"has_thumbnail": False, "mismatch": False, "similarity": 1.0}
    try:
        pil_img = Image.open(file_path)
        exif_raw = pil_img.getexif()

        # Try to get thumbnail
        if hasattr(pil_img, "_getexif") and pil_img._getexif():
            exif_dict = pil_img._getexif()
            if exif_dict and 0x0201 in exif_dict:  # JPEGInterchangeFormat
                result["has_thumbnail"] = True
                # Load thumbnail
                thumb_offset = exif_dict[0x0201]
                thumb_length = exif_dict.get(0x0202, 0)
                if thumb_length > 0:
                    with open(file_path, "rb") as f:
                        f.seek(thumb_offset)
                        thumb_data = f.read(thumb_length)
                    thumb = cv2.imdecode(
                        np.frombuffer(thumb_data, np.uint8), cv2.IMREAD_COLOR
                    )
                    if thumb is not None:
                        # Resize both to same size and compare
                        main_small = cv2.resize(
                            cv2.imread(file_path), (thumb.shape[1], thumb.shape[0])
                        )
                        similarity = float(cv2.matchTemplate(
                            main_small, thumb, cv2.TM_CCOEFF_NORMED
                        ).max())
                        result["similarity"] = similarity
                        result["mismatch"] = similarity < 0.8
    except Exception:
        pass

    return result


def predict(file_path: str) -> dict:
    """Analyze EXIF metadata for forensic indicators. Returns score 0-1."""
    t0 = time.perf_counter()

    exif = _extract_exif(file_path)
    findings: list[str] = []
    score = 0.0

    # Check 1: No EXIF at all (stripped) — suspicious for claiming to be a camera photo
    if len(exif) < 3:
        findings.append("EXIF data is missing or stripped")
        score += 0.15

    # Check 2: Software tags
    sw_findings = _check_software_tags(exif)
    findings.extend(sw_findings)
    if sw_findings:
        score += 0.3

    # Check 3: Thumbnail mismatch
    thumb_result = _check_thumbnail_mismatch(file_path)
    if thumb_result["mismatch"]:
        findings.append(f"Thumbnail mismatch (similarity: {thumb_result['similarity']:.2f})")
        score += 0.35

    # Check 4: Inconsistent timestamps
    dates = {}
    for key, val in exif.items():
        key_lower = key.lower()
        if "date" in key_lower or "time" in key_lower:
            dates[key] = val

    if len(set(dates.values())) > 2:
        findings.append(f"Multiple different timestamps found: {len(set(dates.values()))}")
        score += 0.1

    # Check 5: GPS data present (just informational)
    gps_keys = [k for k in exif if "gps" in k.lower()]
    gps_info = {k: exif[k] for k in gps_keys}

    elapsed = int((time.perf_counter() - t0) * 1000)
    return {
        "score": min(score, 1.0),
        "execution_ms": elapsed,
        "details": {
            "total_fields": len(exif),
            "findings": findings,
            "gps": gps_info if gps_info else None,
            "thumbnail": thumb_result,
            "timestamps": dates,
        },
        "exif_data": exif,
    }
