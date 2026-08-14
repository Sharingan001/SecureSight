"""Tier 3 — Shadow / Lighting Consistency Analysis.

Estimates light source direction from edges and checks for physically
impossible multiple light sources or shadow inconsistencies.
"""

from __future__ import annotations

import time

import cv2
import numpy as np


def _estimate_light_direction(gray: np.ndarray) -> np.ndarray:
    """Estimate dominant light direction from gradient analysis.

    Uses Sobel gradients to compute the dominant illumination vector.
    Returns unit vector [dx, dy].
    """
    grad_x = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=5)
    grad_y = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=5)

    # Weighted average gradient direction (brighter areas weight more)
    weights = gray.astype(np.float64) / 255.0
    weighted_gx = np.sum(grad_x * weights)
    weighted_gy = np.sum(grad_y * weights)

    magnitude = np.sqrt(weighted_gx ** 2 + weighted_gy ** 2) + 1e-8
    return np.array([weighted_gx / magnitude, weighted_gy / magnitude])


def _analyze_shadow_consistency(image: np.ndarray) -> dict:
    """Analyze shadow regions for consistency with a single light source."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    h, w = gray.shape

    # Divide image into quadrants → estimate light direction per quadrant
    quad_size_h, quad_size_w = h // 2, w // 2
    quadrants = [
        gray[:quad_size_h, :quad_size_w],
        gray[:quad_size_h, quad_size_w:],
        gray[quad_size_h:, :quad_size_w],
        gray[quad_size_h:, quad_size_w:],
    ]

    directions = np.array([_estimate_light_direction(q) for q in quadrants])

    # Compute pairwise angular differences
    angles = []
    for i in range(4):
        for j in range(i + 1, 4):
            cos_sim = np.clip(np.dot(directions[i], directions[j]), -1.0, 1.0)
            angle_deg = float(np.degrees(np.arccos(cos_sim)))
            angles.append(angle_deg)

    mean_angle_diff = float(np.mean(angles))
    max_angle_diff = float(np.max(angles))

    # Overall light direction
    overall_dir = _estimate_light_direction(gray)

    return {
        "mean_angle_diff": mean_angle_diff,
        "max_angle_diff": max_angle_diff,
        "light_direction": overall_dir.tolist(),
        "quadrant_directions": directions.tolist(),
    }


def _analyze_brightness_asymmetry(face: np.ndarray) -> float:
    """Check face halves for brightness asymmetry beyond natural variation."""
    gray = cv2.cvtColor(face, cv2.COLOR_BGR2GRAY).astype(np.float64)
    h, w = gray.shape
    mid = w // 2

    left_mean = np.mean(gray[:, :mid])
    right_mean = np.mean(gray[:, mid:])
    overall_mean = np.mean(gray) + 1e-8

    asymmetry = abs(left_mean - right_mean) / overall_mean
    return float(asymmetry)


def predict(image: np.ndarray, face_crop: np.ndarray | None = None) -> dict:
    """Analyze shadow/lighting consistency. Returns score 0-1."""
    t0 = time.perf_counter()

    shadow_analysis = _analyze_shadow_consistency(image)
    face_asymmetry = 0.0
    if face_crop is not None:
        face_asymmetry = _analyze_brightness_asymmetry(face_crop)

    score = 0.0

    # High angular difference between quadrant light directions
    # Raised thresholds — normal images often show 40-60° due to scene content
    if shadow_analysis["mean_angle_diff"] > 75:
        score += 0.4
    elif shadow_analysis["mean_angle_diff"] > 45:
        score += 0.2

    if shadow_analysis["max_angle_diff"] > 120:
        score += 0.3
    elif shadow_analysis["max_angle_diff"] > 60:
        score += 0.15

    # Extreme face asymmetry beyond natural side-lighting
    if face_asymmetry > 0.3:
        score += 0.2
    elif face_asymmetry > 0.15:
        score += 0.1

    shadow_analysis["face_asymmetry"] = face_asymmetry
    elapsed = int((time.perf_counter() - t0) * 1000)

    return {"score": min(score, 1.0), "execution_ms": elapsed, "details": shadow_analysis}
