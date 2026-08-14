"""Tier 2 — JPEG Ghost / Double Compression Detection.

Re-compresses at sweep of quality levels → minimum-diff quality reveals original compression.
Mismatched regions indicate splicing from a differently-compressed source.
"""

from __future__ import annotations

import time

import cv2
import numpy as np


def _recompress_diff(image: np.ndarray, quality: int) -> float:
    """Compute mean squared difference between image and re-compressed version."""
    params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode(".jpg", image, params)
    recomp = cv2.imdecode(encoded, cv2.IMREAD_COLOR)
    diff = (image.astype(np.float32) - recomp.astype(np.float32)) ** 2
    return float(np.mean(diff))


def predict(image: np.ndarray) -> dict:
    """Detect JPEG ghosting / double compression artifacts.

    Sweeps quality 50-100 in steps of 5, finds the quality with minimum diff.
    If the estimated quality differs from JPEG standard or shows localized ghosts,
    scores higher for manipulation.
    """
    t0 = time.perf_counter()

    # Sweep quality levels — vectorized as much as possible
    qualities = list(range(50, 101, 5))
    diffs = np.array([_recompress_diff(image, q) for q in qualities])

    # Find quality with minimum diff (= likely original compression quality)
    best_idx = int(np.argmin(diffs))
    estimated_quality = qualities[best_idx]
    min_diff = diffs[best_idx]

    # Compute diff variance across quality sweep
    diff_variance = float(np.var(diffs))
    diff_range = float(np.max(diffs) - np.min(diffs))

    # Check for localized inconsistency at estimated quality
    params = [cv2.IMWRITE_JPEG_QUALITY, estimated_quality]
    _, encoded = cv2.imencode(".jpg", image, params)
    recomp = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    block_diffs = []
    h, w = image.shape[:2]
    block_size = 64
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block_orig = image[y:y + block_size, x:x + block_size].astype(np.float32)
            block_recomp = recomp[y:y + block_size, x:x + block_size].astype(np.float32)
            block_diffs.append(float(np.mean((block_orig - block_recomp) ** 2)))

    block_diffs_arr = np.array(block_diffs) if block_diffs else np.array([0.0])
    block_std = float(np.std(block_diffs_arr))
    block_mean = float(np.mean(block_diffs_arr))

    # High block variance at best quality = different regions compressed differently
    block_cv = block_std / (block_mean + 1e-8)

    # Scoring
    score = 0.0

    # Indicator 1: High block-level variance suggests mixed compression
    if block_cv > 1.0:
        score += 0.35
    elif block_cv > 0.5:
        score += 0.2

    # Indicator 2: Very low estimated quality with blocks at different levels
    if estimated_quality < 70 and block_cv > 0.3:
        score += 0.2

    # Indicator 3: Diff curve shape — double-compressed images show unique dip pattern
    if diff_range > 50 and diff_variance > 200:
        score += 0.15

    # Baseline noise
    score = max(score, 0.03)

    elapsed = int((time.perf_counter() - t0) * 1000)
    return {
        "score": min(score, 1.0),
        "execution_ms": elapsed,
        "details": {
            "estimated_quality": estimated_quality,
            "min_diff": min_diff,
            "block_cv": block_cv,
            "diff_range": diff_range,
            "diff_variance": diff_variance,
        },
    }
