"""Tier 3 — Noise Pattern Consistency Analysis.

Extracts sensor noise residual via wavelet denoising, then checks for
inconsistent noise levels across image blocks (indicates splicing).
"""

from __future__ import annotations

import time

import cv2
import numpy as np


def _extract_noise_residual(image: np.ndarray) -> np.ndarray:
    """Extract noise residual: original - denoised. Vectorized."""
    denoised = cv2.fastNlMeansDenoisingColored(image, None, 10, 10, 7, 21)
    residual = image.astype(np.float32) - denoised.astype(np.float32)
    return residual


def _block_noise_analysis(residual: np.ndarray, block_size: int = 64) -> dict:
    """Compute noise variance per block, check for inconsistencies."""
    gray_residual = np.mean(residual, axis=2)  # average across channels
    h, w = gray_residual.shape

    variances = []
    for y in range(0, h - block_size + 1, block_size):
        for x in range(0, w - block_size + 1, block_size):
            block = gray_residual[y:y + block_size, x:x + block_size]
            variances.append(float(np.var(block)))

    variances_arr = np.array(variances) if variances else np.array([0.0])

    mean_var = float(np.mean(variances_arr))
    std_var = float(np.std(variances_arr))
    cv = std_var / (mean_var + 1e-8)

    # Find outlier blocks (> 2 std from mean)
    threshold = mean_var + 2 * std_var
    n_outliers = int(np.sum(variances_arr > threshold))
    outlier_ratio = n_outliers / (len(variances_arr) + 1e-8)

    return {
        "mean_noise_variance": mean_var,
        "noise_std": std_var,
        "noise_cv": cv,
        "n_blocks": len(variances_arr),
        "n_outlier_blocks": n_outliers,
        "outlier_ratio": outlier_ratio,
    }


def predict(image: np.ndarray) -> dict:
    """Analyze noise pattern consistency. Returns score 0-1."""
    t0 = time.perf_counter()

    residual = _extract_noise_residual(image)
    analysis = _block_noise_analysis(residual, block_size=64)

    score = 0.0

    # High coefficient of variation = inconsistent noise = likely spliced
    if analysis["noise_cv"] > 1.5:
        score += 0.4
    elif analysis["noise_cv"] > 0.8:
        score += 0.2
    elif analysis["noise_cv"] > 0.4:
        score += 0.1

    # Outlier blocks
    if analysis["outlier_ratio"] > 0.15:
        score += 0.3
    elif analysis["outlier_ratio"] > 0.05:
        score += 0.15

    # Very low noise overall (possible synthetic / heavily processed)
    if analysis["mean_noise_variance"] < 0.5:
        score += 0.15

    score = max(score, 0.02)  # baseline

    elapsed = int((time.perf_counter() - t0) * 1000)
    return {"score": min(score, 1.0), "execution_ms": elapsed, "details": analysis}
