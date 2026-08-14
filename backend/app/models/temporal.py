"""Tier 4 — Temporal Consistency Analysis (video only).

Inter-frame SSIM + optical flow for detecting flickering, warping,
and identity inconsistency in deepfake videos.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from skimage.metrics import structural_similarity as ssim


def _compute_frame_ssim(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """SSIM between two face frames (grayscale, fast)."""
    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(frame2, (g1.shape[1], g1.shape[0])), cv2.COLOR_BGR2GRAY)
    return float(ssim(g1, g2))


def _compute_optical_flow_magnitude(frame1: np.ndarray, frame2: np.ndarray) -> float:
    """Mean optical flow magnitude between consecutive frames."""
    g1 = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
    g2 = cv2.cvtColor(cv2.resize(frame2, (g1.shape[1], g1.shape[0])), cv2.COLOR_BGR2GRAY)

    flow = cv2.calcOpticalFlowFarneback(g1, g2, None, 0.5, 3, 15, 3, 5, 1.2, 0)
    magnitude = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
    return float(np.mean(magnitude))


def predict(face_crops: list[np.ndarray]) -> dict:
    """Analyze temporal consistency across video frames. Returns score 0-1."""
    t0 = time.perf_counter()

    if len(face_crops) < 3:
        return {"score": 0.0, "execution_ms": 0, "details": {"status": "too_few_frames"}}

    # Sample for speed (max 40 pair comparisons)
    step = max(1, (len(face_crops) - 1) // 40)
    ssim_values = []
    flow_values = []

    for i in range(0, len(face_crops) - 1, step):
        s = _compute_frame_ssim(face_crops[i], face_crops[i + 1])
        ssim_values.append(s)

        f = _compute_optical_flow_magnitude(face_crops[i], face_crops[i + 1])
        flow_values.append(f)

    ssim_arr = np.array(ssim_values)
    flow_arr = np.array(flow_values)

    mean_ssim = float(np.mean(ssim_arr))
    ssim_std = float(np.std(ssim_arr))
    min_ssim = float(np.min(ssim_arr))

    mean_flow = float(np.mean(flow_arr))
    flow_std = float(np.std(flow_arr))

    # Frame-level scores (for timeline visualization)
    frame_scores = []
    for i, (s, f) in enumerate(zip(ssim_values, flow_values)):
        anomaly = 0.0
        if s < mean_ssim - 2 * ssim_std:
            anomaly += 0.5
        if f > mean_flow + 2 * flow_std:
            anomaly += 0.5
        frame_scores.append({"index": i * step, "ssim": s, "flow": f, "anomaly": min(anomaly, 1.0)})

    # Overall score
    score = 0.0

    # Low mean SSIM = high face instability
    if mean_ssim < 0.7:
        score += 0.3
    elif mean_ssim < 0.85:
        score += 0.15

    # High SSIM variance = flickering
    if ssim_std > 0.15:
        score += 0.3
    elif ssim_std > 0.08:
        score += 0.15

    # Very sudden SSIM drops = identity swap
    if min_ssim < 0.5:
        score += 0.25
    elif min_ssim < 0.65:
        score += 0.1

    # Erratic optical flow
    if flow_std > 3.0:
        score += 0.15

    elapsed = int((time.perf_counter() - t0) * 1000)
    return {
        "score": min(score, 1.0),
        "execution_ms": elapsed,
        "details": {
            "mean_ssim": mean_ssim,
            "ssim_std": ssim_std,
            "min_ssim": min_ssim,
            "mean_flow": mean_flow,
            "flow_std": flow_std,
            "frame_scores": frame_scores,
        },
    }
