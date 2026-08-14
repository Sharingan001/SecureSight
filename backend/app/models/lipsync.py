"""Tier 1 — Lip-sync verification (audio-visual correlation).

Correlates mouth landmark motion with audio energy envelope.
Desynchronized lip movement indicates lip-sync deepfake.
"""

from __future__ import annotations

import time
from typing import Optional

import cv2
import numpy as np

from app.config import settings


def _get_mouth_openness(frame: np.ndarray, face_mesh) -> float | None:
    """Compute mouth openness ratio using pre-created MediaPipe FaceMesh.

    Args:
        frame: BGR video frame
        face_mesh: Pre-created FaceMesh instance (reused across all frames)
    """
    try:
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb)

        if not results.multi_face_landmarks:
            return None

        lm = results.multi_face_landmarks[0].landmark
        h, w = frame.shape[:2]

        # Upper lip (13) and lower lip (14) landmarks
        upper = np.array([lm[13].x * w, lm[13].y * h])
        lower = np.array([lm[14].x * w, lm[14].y * h])
        left  = np.array([lm[78].x * w, lm[78].y * h])
        right = np.array([lm[308].x * w, lm[308].y * h])

        vertical   = np.linalg.norm(upper - lower)
        horizontal = np.linalg.norm(left - right) + 1e-8

        return float(vertical / horizontal)

    except Exception:
        return None


def _get_audio_energy(audio_path: str, n_windows: int) -> Optional[np.ndarray]:
    """Compute audio RMS energy in windows matching video frame rate."""
    try:
        import librosa
        y, sr = librosa.load(audio_path, sr=16000, mono=True)
        if len(y) < 1600:
            return None

        samples_per_window = len(y) // n_windows
        energies = np.array([
            float(np.sqrt(np.mean(y[i * samples_per_window:(i + 1) * samples_per_window] ** 2)))
            for i in range(n_windows)
        ])

        # Normalize to 0-1
        emax = energies.max()
        if emax > 0:
            energies /= emax
        return energies

    except Exception:
        return None


def predict(frames: list[np.ndarray], audio_path: Optional[str]) -> dict:
    """Compute lip-sync correlation between mouth motion and audio energy."""
    t0 = time.perf_counter()

    if audio_path is None or len(frames) < 5:
        return {"score": 0.0, "execution_ms": 0, "details": {"status": "insufficient_data"}}

    # Sample frames evenly (max 60 for speed)
    step = max(1, len(frames) // 60)
    sampled = frames[::step]

    # Create FaceMesh ONCE and reuse across all sampled frames
    try:
        import mediapipe as mp
        face_mesh = mp.solutions.face_mesh.FaceMesh(
            static_image_mode=True, max_num_faces=1, min_detection_confidence=0.5
        )
    except Exception as e:
        return {"score": 0.0, "execution_ms": 0, "details": {"status": f"mediapipe_init_failed: {e}"}}

    # Extract mouth openness for each sampled frame
    mouth_values = []
    for frame in sampled:
        val = _get_mouth_openness(frame, face_mesh)
        mouth_values.append(val if val is not None else 0.0)
    face_mesh.close()  # release after all frames processed

    mouth_arr = np.array(mouth_values, dtype=np.float64)

    # Get audio energy windows
    audio_energy = _get_audio_energy(audio_path, len(sampled))
    if audio_energy is None:
        return {"score": 0.0, "execution_ms": int((time.perf_counter() - t0) * 1000),
                "details": {"status": "audio_extraction_failed"}}

    # Pearson correlation between mouth openness and audio energy
    if np.std(mouth_arr) < 1e-6 or np.std(audio_energy) < 1e-6:
        correlation = 0.0
    else:
        correlation = float(np.corrcoef(mouth_arr, audio_energy)[0, 1])

    # Low or negative correlation = likely lip-sync deepfake
    # Real speech: correlation typically 0.3-0.8
    if correlation > 0.4:
        fake_score = 0.0
    elif correlation > 0.2:
        fake_score = 0.3 * (1.0 - (correlation - 0.2) / 0.2)
    elif correlation > 0.0:
        fake_score = 0.3 + 0.3 * (1.0 - correlation / 0.2)
    else:
        fake_score = 0.6 + 0.4 * min(abs(correlation), 1.0)

    elapsed = int((time.perf_counter() - t0) * 1000)
    return {
        "score": min(fake_score, 1.0),
        "execution_ms": elapsed,
        "details": {
            "correlation": correlation,
            "mouth_frames_analyzed": len(sampled),
        },
    }
