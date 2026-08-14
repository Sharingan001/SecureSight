"""Tier 4 — Frequency Domain Analysis (DCT + FFT).

Analyzes spectral energy distribution — GANs show suppressed high-frequency
components due to upsampling artifacts.
"""

from __future__ import annotations

import time

import cv2
import numpy as np
from scipy.fft import fft2, fftshift


def _compute_fft_spectrum(gray: np.ndarray) -> np.ndarray:
    """Compute magnitude spectrum of 2D FFT (log-scaled)."""
    f_transform = fft2(gray.astype(np.float64))
    f_shifted = fftshift(f_transform)
    magnitude = np.log1p(np.abs(f_shifted))
    return magnitude


def _azimuthal_average(spectrum: np.ndarray) -> np.ndarray:
    """Compute azimuthal average of 2D power spectrum → 1D radial profile.

    Vectorized using integer radius indexing.
    """
    h, w = spectrum.shape
    cy, cx = h // 2, w // 2
    y, x = np.ogrid[:h, :w]
    r = np.sqrt((x - cx) ** 2 + (y - cy) ** 2).astype(int)

    max_r = min(cy, cx)
    r_clipped = np.clip(r, 0, max_r - 1)

    avg = np.zeros(max_r)
    counts = np.zeros(max_r)

    np.add.at(avg, r_clipped.ravel(), spectrum.ravel())
    np.add.at(counts, r_clipped.ravel(), 1)

    counts[counts == 0] = 1
    return avg / counts


def _spectral_energy_ratio(radial_profile: np.ndarray) -> float:
    """Ratio of high-frequency to low-frequency energy.

    GAN images: lower ratio (suppressed high freq).
    Real images: higher ratio.
    """
    mid = len(radial_profile) // 2
    low_freq = np.sum(radial_profile[:mid]) + 1e-8
    high_freq = np.sum(radial_profile[mid:]) + 1e-8
    return float(high_freq / low_freq)


def _compute_dct_features(gray: np.ndarray) -> dict:
    """Compute DCT-based features for deepfake detection."""
    # Block-wise DCT (8×8 blocks, like JPEG)
    h, w = gray.shape
    h8, w8 = (h // 8) * 8, (w // 8) * 8
    gray = gray[:h8, :w8].astype(np.float64)

    n_blocks = (h8 // 8) * (w8 // 8)
    dc_values = []
    high_freq_energy = []

    for y in range(0, h8, 8):
        for x in range(0, w8, 8):
            block = gray[y:y + 8, x:x + 8]
            dct_block = cv2.dct(block)
            dc_values.append(dct_block[0, 0])
            # Sum of high-frequency coefficients (bottom-right triangle)
            high_freq_energy.append(float(np.sum(np.abs(dct_block[4:, 4:]))))

    dc_arr = np.array(dc_values)
    hf_arr = np.array(high_freq_energy)

    return {
        "dc_mean": float(np.mean(dc_arr)),
        "dc_std": float(np.std(dc_arr)),
        "highfreq_mean": float(np.mean(hf_arr)),
        "highfreq_std": float(np.std(hf_arr)),
        "highfreq_cv": float(np.std(hf_arr) / (np.mean(hf_arr) + 1e-8)),
    }


def predict(face_crop: np.ndarray) -> dict:
    """Frequency domain analysis. Returns score 0-1."""
    t0 = time.perf_counter()

    gray = cv2.cvtColor(face_crop, cv2.COLOR_BGR2GRAY)

    # FFT analysis
    spectrum = _compute_fft_spectrum(gray)
    radial = _azimuthal_average(spectrum)
    energy_ratio = _spectral_energy_ratio(radial)

    # DCT analysis
    dct_features = _compute_dct_features(gray)

    # Scoring
    score = 0.0

    # Low high-freq energy ratio = GAN artifact
    if energy_ratio < 0.15:
        score += 0.45
    elif energy_ratio < 0.25:
        score += 0.25
    elif energy_ratio < 0.35:
        score += 0.1

    # Very uniform DCT high-freq (low variance) = synthetic
    if dct_features["highfreq_cv"] < 0.3:
        score += 0.25
    elif dct_features["highfreq_cv"] < 0.5:
        score += 0.1

    # Very low high-freq energy in DCT
    if dct_features["highfreq_mean"] < 5.0:
        score += 0.2

    elapsed = int((time.perf_counter() - t0) * 1000)
    return {
        "score": min(score, 1.0),
        "execution_ms": elapsed,
        "details": {
            "energy_ratio": energy_ratio,
            "dct": dct_features,
        },
    }
