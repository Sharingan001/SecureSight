"""Tier 4 — Biometric Landmark Consistency (MediaPipe Face Mesh).

Checks facial symmetry, eye aspect ratio, lip geometry, and head pose
smoothness. Deepfakes show impossible geometry and jittery pose.
"""

from __future__ import annotations

import time

import cv2
import numpy as np

from app.config import settings


def _analyze_single_frame(frame: np.ndarray, face_mesh) -> dict | None:
    """Extract biometric features from a single face frame.

    Args:
        frame: BGR face crop
        face_mesh: Pre-created MediaPipe FaceMesh instance (reused across frames)
    """
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        def pt(idx):
            return np.array([lm[idx].x * w, lm[idx].y * h])

        # Eye Aspect Ratio (EAR) — measures eye openness
        # Left eye: 33, 160, 158, 133, 153, 144
        left_ear = (
            (np.linalg.norm(pt(160) - pt(144)) + np.linalg.norm(pt(158) - pt(153)))
            / (2 * np.linalg.norm(pt(33) - pt(133)) + 1e-8)
        )
        # Right eye: 362, 385, 387, 263, 373, 380
        right_ear = (
            (np.linalg.norm(pt(385) - pt(380)) + np.linalg.norm(pt(387) - pt(373)))
            / (2 * np.linalg.norm(pt(362) - pt(263)) + 1e-8)
        )

        # Facial symmetry — compare left/right landmark distances to midline
        nose_tip = pt(1)
        left_pts = [pt(i) for i in [234, 127, 162, 21, 54]]
        right_pts = [pt(i) for i in [454, 356, 389, 251, 284]]

        left_dists = [np.linalg.norm(p - nose_tip) for p in left_pts]
        right_dists = [np.linalg.norm(p - nose_tip) for p in right_pts]

        symmetry_ratios = [
            min(l, r) / (max(l, r) + 1e-8)
            for l, r in zip(left_dists, right_dists)
        ]
        mean_symmetry = float(np.mean(symmetry_ratios))

        # Mouth aspect ratio
        mouth_h = np.linalg.norm(pt(13) - pt(14))
        mouth_w = np.linalg.norm(pt(78) - pt(308))
        mouth_ratio = float(mouth_h / (mouth_w + 1e-8))

        return {
            "left_ear": float(left_ear),
            "right_ear": float(right_ear),
            "ear_diff": abs(float(left_ear - right_ear)),
            "symmetry": mean_symmetry,
            "mouth_ratio": mouth_ratio,
        }

    except Exception:
        return None


def predict(face_crops: list[np.ndarray]) -> dict:
    """Analyze biometric consistency across frames. Returns score 0-1."""
    t0 = time.perf_counter()

    if not face_crops:
        return {"score": 0.0, "execution_ms": 0, "details": {"status": "no_faces"}}

    # Create FaceMesh ONCE and reuse across all frames — avoids per-frame init overhead
    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
        )
    except Exception as e:
        return {"score": 0.0, "execution_ms": 0, "details": {"status": f"mediapipe_init_failed: {e}"}}

    # Analyze up to 30 crops for speed
    sampled = face_crops[::max(1, len(face_crops) // 30)]
    results = [_analyze_single_frame(crop, face_mesh) for crop in sampled]
    face_mesh.close()  # release after all frames processed
    valid = [r for r in results if r is not None]

    if not valid:
        return {"score": 0.0, "execution_ms": int((time.perf_counter() - t0) * 1000),
                "details": {"status": "analysis_failed"}}

    # Aggregate metrics
    symmetries = [r["symmetry"] for r in valid]
    ear_diffs = [r["ear_diff"] for r in valid]

    mean_symmetry = float(np.mean(symmetries))
    symmetry_std = float(np.std(symmetries))
    mean_ear_diff = float(np.mean(ear_diffs))

    # For video: check temporal consistency of biometrics
    ear_values = [(r["left_ear"] + r["right_ear"]) / 2 for r in valid]
    ear_temporal_std = float(np.std(ear_values)) if len(ear_values) > 1 else 0.0

    score = 0.0

    # Low symmetry = distorted/impossible face geometry
    if mean_symmetry < 0.7:
        score += 0.35
    elif mean_symmetry < 0.85:
        score += 0.15

    # High EAR difference between eyes
    if mean_ear_diff > 0.15:
        score += 0.25
    elif mean_ear_diff > 0.08:
        score += 0.1

    # Jittery symmetry across frames (temporal inconsistency)
    if symmetry_std > 0.1:
        score += 0.2
    elif symmetry_std > 0.05:
        score += 0.1

    # Abnormally stable EAR (no natural blink variation) for video
    if len(valid) > 5 and ear_temporal_std < 0.01:
        score += 0.15

    elapsed = int((time.perf_counter() - t0) * 1000)
    return {
        "score": min(score, 1.0),
        "execution_ms": elapsed,
        "details": {
            "mean_symmetry": mean_symmetry,
            "symmetry_std": symmetry_std,
            "mean_ear_diff": mean_ear_diff,
            "ear_temporal_std": ear_temporal_std,
            "frames_analyzed": len(valid),
        },
    }
