"""Tier 2 — Error Level Analysis (ELA).

Re-saves image at known quality → computes pixel diff → manipulation shows as bright regions.
Optimized: single OpenCV encode/decode, vectorized diff computation.
"""

from __future__ import annotations

import time

import cv2
import numpy as np


def compute_ela(image: np.ndarray, quality: int = 95, scale: int = 15) -> np.ndarray:
    """Compute ELA map. Returns a single-channel heatmap (0-255)."""
    # Re-encode at specified quality
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode(".jpg", image, encode_params)
    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    # Absolute difference — vectorized, no Python loops
    diff = cv2.absdiff(image, recompressed).astype(np.float32)

    # Scale and convert
    ela_map = np.clip(diff * scale, 0, 255).astype(np.uint8)

    # Convert to single channel (max across BGR)
    ela_gray = np.max(ela_map, axis=2)

    return ela_gray


def compute_ela_color(image: np.ndarray, quality: int = 95, scale: int = 15) -> np.ndarray:
    """Compute ELA map preserving color channels. Returns BGR image."""
    encode_params = [cv2.IMWRITE_JPEG_QUALITY, quality]
    _, encoded = cv2.imencode(".jpg", image, encode_params)
    recompressed = cv2.imdecode(encoded, cv2.IMREAD_COLOR)

    diff = cv2.absdiff(image, recompressed).astype(np.float32)
    ela_color = np.clip(diff * scale, 0, 255).astype(np.uint8)

    return ela_color


def _analyze_ela_regions(ela_gray: np.ndarray) -> dict:
    """Analyze ELA map for suspicious regions."""
    mean_val = float(np.mean(ela_gray))
    std_val = float(np.std(ela_gray))

    # Dynamic threshold — adapt to the actual ELA distribution
    # For clean JPEGs: mean is very low (0-5), so threshold should be higher
    # For manipulated: mean is moderate (20-80), std is high
    threshold = max(mean_val + 2.5 * std_val, 25.0)  # minimum threshold 25
    threshold = min(threshold, 200)

    suspicious_mask = (ela_gray > threshold).astype(np.uint8)
    suspicious_ratio = float(np.sum(suspicious_mask)) / suspicious_mask.size

    # Region analysis using connected components
    n_labels, labels, stats, centroids = cv2.connectedComponentsWithStats(suspicious_mask, connectivity=8)

    # Filter regions: must be significant relative to image size
    image_area = ela_gray.shape[0] * ela_gray.shape[1]
    min_region_area = max(500, image_area * 0.002)  # at least 0.2% of image

    suspicious_regions = []
    total_suspicious_area = 0
    for i in range(1, n_labels):  # skip background
        area = stats[i, cv2.CC_STAT_AREA]
        if area > min_region_area:
            # Compute mean ELA intensity within this region
            region_mask = (labels == i)
            region_intensity = float(np.mean(ela_gray[region_mask]))

            suspicious_regions.append({
                "x": int(stats[i, cv2.CC_STAT_LEFT]),
                "y": int(stats[i, cv2.CC_STAT_TOP]),
                "w": int(stats[i, cv2.CC_STAT_WIDTH]),
                "h": int(stats[i, cv2.CC_STAT_HEIGHT]),
                "area": int(area),
                "intensity": region_intensity,
            })
            total_suspicious_area += area

    # Sort by area descending
    suspicious_regions.sort(key=lambda r: r["area"], reverse=True)

    return {
        "mean_ela": mean_val,
        "std_ela": std_val,
        "suspicious_ratio": suspicious_ratio,
        "significant_area_ratio": total_suspicious_area / image_area if image_area > 0 else 0.0,
        "suspicious_regions": suspicious_regions[:10],
    }


def predict(image: np.ndarray) -> dict:
    """Run ELA analysis. Returns deepfake score 0-1."""
    t0 = time.perf_counter()

    # Run at multiple quality levels for robustness
    ela_95 = compute_ela(image, quality=95, scale=15)
    ela_75 = compute_ela(image, quality=75, scale=10)

    analysis_95 = _analyze_ela_regions(ela_95)
    analysis_75 = _analyze_ela_regions(ela_75)

    # Use the Q95 analysis as primary (more sensitive to manipulation)
    analysis = analysis_95
    mean_ela = analysis["mean_ela"]
    std_ela = analysis["std_ela"]
    sig_ratio = analysis["significant_area_ratio"]
    n_regions = len(analysis["suspicious_regions"])

    # Cross-quality consistency check: if Q75 and Q95 show similar patterns,
    # it's more likely manipulation vs just compression artifacts
    mean_ela_75 = analysis_75["mean_ela"]
    cross_quality_corr = abs(mean_ela - mean_ela_75) / (max(mean_ela, mean_ela_75) + 1e-8)

    # ── Scoring ──────────────────────────────────────────────────────
    # A clean JPEG: mean_ela < 3, std < 5, no significant regions
    # Manipulated:  mean_ela > 15, localized bright regions, high std
    # Re-encoded:   mean_ela > 50, uniformly bright (less informative)

    score = 0.0

    if mean_ela < 3.0 and n_regions == 0:
        # Very clean — likely original JPEG
        score = 0.05
    elif mean_ela < 8.0 and n_regions <= 1:
        # Minor artifacts — normal for social media compression
        score = 0.10 + 0.05 * n_regions
    elif sig_ratio > 0.3:
        # Large portion of image differs — heavy re-encoding, less informative
        score = 0.25
    elif sig_ratio > 0.05:
        # Significant localized manipulation
        score = 0.35 + 0.35 * min(sig_ratio / 0.3, 1.0)
    elif n_regions > 0:
        # Some suspicious regions detected
        avg_intensity = np.mean([r.get("intensity", 0) for r in analysis["suspicious_regions"]])
        intensity_factor = min(avg_intensity / 100.0, 1.0)
        region_factor = min(n_regions / 5.0, 1.0)
        score = 0.20 + 0.30 * (0.6 * intensity_factor + 0.4 * region_factor)
    else:
        score = 0.08

    # Boost score if cross-quality analysis is consistent (suggests real manipulation)
    if cross_quality_corr < 0.3 and n_regions > 0:
        score = min(score * 1.15, 0.95)

    score = min(max(score, 0.0), 0.95)  # Never output exactly 1.0

    elapsed = int((time.perf_counter() - t0) * 1000)
    return {"score": score, "execution_ms": elapsed, "details": analysis, "ela_map": ela_95}
