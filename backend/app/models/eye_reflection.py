"""Tier 3 — Eye Reflection Consistency Analysis.

Compares corneal specular highlights between left and right eyes.
GAN-generated faces often have inconsistent eye reflections.
"""

from __future__ import annotations

import time

import cv2
import numpy as np


def _extract_eye_region(face: np.ndarray, side: str) -> np.ndarray | None:
    """Extract eye region using proportional face landmarks (no ML needed)."""
    h, w = face.shape[:2]

    if side == "left":
        # Left eye approximate region: top 25-45% height, left 10-45% width
        y1, y2 = int(h * 0.25), int(h * 0.45)
        x1, x2 = int(w * 0.10), int(w * 0.45)
    else:
        # Right eye: top 25-45% height, right 55-90% width
        y1, y2 = int(h * 0.25), int(h * 0.45)
        x1, x2 = int(w * 0.55), int(w * 0.90)

    eye = face[y1:y2, x1:x2]
    if eye.size == 0:
        return None
    return eye


def _extract_specular_highlights(eye: np.ndarray) -> np.ndarray:
    """Extract bright specular highlights (reflections) from eye region."""
    gray = cv2.cvtColor(eye, cv2.COLOR_BGR2GRAY)

    # Adaptive thresholding to find bright spots
    # Specular reflections are the brightest pixels
    thresh = max(200, int(np.percentile(gray, 97)))
    _, bright_mask = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    return bright_mask


def _compare_reflections(mask_l: np.ndarray, mask_r: np.ndarray) -> dict:
    """Compare reflection patterns between left and right eye."""
    # Resize both to same dimensions for comparison
    target_size = (64, 32)
    mask_l_resized = cv2.resize(mask_l, target_size)
    mask_r_resized = cv2.resize(mask_r, target_size)

    # Flip right to compare with left (mirror symmetry)
    mask_r_flipped = cv2.flip(mask_r_resized, 1)

    # Count bright pixels (reflection areas)
    n_l = np.sum(mask_l_resized > 0)
    n_r = np.sum(mask_r_resized > 0)

    # Area ratio similarity
    if max(n_l, n_r) > 0:
        area_ratio = min(n_l, n_r) / (max(n_l, n_r) + 1e-8)
    else:
        area_ratio = 1.0  # both have no reflections = consistent

    # Structural similarity via normalized cross-correlation
    l_f = mask_l_resized.astype(np.float32)
    r_f = mask_r_flipped.astype(np.float32)

    if np.std(l_f) > 0 and np.std(r_f) > 0:
        ncc = float(cv2.matchTemplate(l_f, r_f, cv2.TM_CCOEFF_NORMED).max())
    else:
        ncc = 1.0 if (np.std(l_f) == 0 and np.std(r_f) == 0) else 0.0

    # Position similarity: centroid comparison
    def centroid(mask):
        ys, xs = np.where(mask > 0)
        if len(xs) == 0:
            return 0.5, 0.5
        return float(np.mean(xs)) / mask.shape[1], float(np.mean(ys)) / mask.shape[0]

    cl = centroid(mask_l_resized)
    cr = centroid(mask_r_flipped)
    position_diff = np.sqrt((cl[0] - cr[0]) ** 2 + (cl[1] - cr[1]) ** 2)

    return {
        "area_ratio": area_ratio,
        "ncc": ncc,
        "position_diff": position_diff,
        "left_pixels": int(n_l),
        "right_pixels": int(n_r),
    }


def predict(face_crop: np.ndarray) -> dict:
    """Analyze eye reflection consistency. Returns score 0-1."""
    t0 = time.perf_counter()

    left_eye = _extract_eye_region(face_crop, "left")
    right_eye = _extract_eye_region(face_crop, "right")

    if left_eye is None or right_eye is None:
        return {"score": 0.0, "execution_ms": 0, "details": {"status": "eyes_not_found"}}

    mask_l = _extract_specular_highlights(left_eye)
    mask_r = _extract_specular_highlights(right_eye)

    comparison = _compare_reflections(mask_l, mask_r)

    # Score: low consistency = higher deepfake probability
    score = 0.0

    if comparison["area_ratio"] < 0.3:
        score += 0.35
    elif comparison["area_ratio"] < 0.6:
        score += 0.15

    if comparison["ncc"] < 0.2:
        score += 0.35
    elif comparison["ncc"] < 0.5:
        score += 0.15

    if comparison["position_diff"] > 0.3:
        score += 0.2
    elif comparison["position_diff"] > 0.15:
        score += 0.1

    elapsed = int((time.perf_counter() - t0) * 1000)
    return {"score": min(score, 1.0), "execution_ms": elapsed, "details": comparison}
