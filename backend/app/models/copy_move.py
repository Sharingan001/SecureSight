"""Tier 2 — Copy-Move Forgery Detection (ORB keypoints + FLANN matching).

Detects cloned/duplicated regions within a single image.
Optimized: ORB is faster than SIFT/SURF, FLANN for fast matching, RANSAC for outlier filtering.
"""

from __future__ import annotations

import time

import cv2
import numpy as np


def predict(image: np.ndarray) -> dict:
    """Detect copy-move forgery using ORB keypoints + FLANN matching."""
    t0 = time.perf_counter()

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # ORB is ~100× faster than SIFT and patent-free
    orb = cv2.ORB_create(nfeatures=3000, scaleFactor=1.2, nlevels=8)
    keypoints, descriptors = orb.detectAndCompute(gray, None)

    if descriptors is None or len(keypoints) < 10:
        return {"score": 0.0, "execution_ms": int((time.perf_counter() - t0) * 1000),
                "details": {"keypoints": 0, "matches": 0}}

    # FLANN-based matcher (for binary descriptors like ORB, use LSH)
    index_params = dict(algorithm=6, table_number=6, key_size=12, multi_probe_level=1)
    search_params = dict(checks=50)
    flann = cv2.FlannBasedMatcher(index_params, search_params)

    # Self-match: find pairs within the same image
    matches = flann.knnMatch(descriptors, descriptors, k=5)

    # Filter: keep matches that are far apart spatially (same descriptor, different location)
    good_matches = []
    min_distance_px = 60  # minimum pixel distance to avoid local texture matches

    for match_group in matches:
        for m in match_group:
            if m.queryIdx == m.trainIdx:
                continue  # skip self-match
            pt1 = np.array(keypoints[m.queryIdx].pt)
            pt2 = np.array(keypoints[m.trainIdx].pt)
            spatial_dist = np.linalg.norm(pt1 - pt2)

            # Tighter distance threshold for ORB binary descriptors
            if spatial_dist > min_distance_px and m.distance < 30:
                good_matches.append((m, pt1, pt2))

    # Remove duplicate pairs (A→B and B→A)
    unique_matches = {}
    for m, pt1, pt2 in good_matches:
        key = tuple(sorted([m.queryIdx, m.trainIdx]))
        if key not in unique_matches:
            unique_matches[key] = (m, pt1, pt2)

    # RANSAC geometric verification to filter false matches
    verified_count = len(unique_matches)
    if verified_count >= 4:
        src_pts = np.array([pt1 for _, pt1, _ in unique_matches.values()], dtype=np.float32)
        dst_pts = np.array([pt2 for _, _, pt2 in unique_matches.values()], dtype=np.float32)
        _, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
        if mask is not None:
            verified_count = int(mask.sum())

    n_matches = verified_count

    # Score based on number of RANSAC-verified matching pairs
    # Much higher thresholds to avoid false positives on natural images
    if n_matches > 80:
        score = 0.85
    elif n_matches > 40:
        score = 0.55 + 0.30 * ((n_matches - 40) / 40)
    elif n_matches > 15:
        score = 0.25 + 0.30 * ((n_matches - 15) / 25)
    elif n_matches > 5:
        score = 0.08 + 0.17 * ((n_matches - 5) / 10)
    else:
        score = 0.02

    # Build overlay mask for visualization
    overlay_mask = np.zeros(gray.shape, dtype=np.uint8)
    for _, pt1, pt2 in list(unique_matches.values())[:100]:
        cv2.circle(overlay_mask, (int(pt1[0]), int(pt1[1])), 8, 255, -1)
        cv2.circle(overlay_mask, (int(pt2[0]), int(pt2[1])), 8, 255, -1)

    elapsed = int((time.perf_counter() - t0) * 1000)
    return {
        "score": min(score, 1.0),
        "execution_ms": elapsed,
        "details": {
            "keypoints": len(keypoints),
            "suspicious_matches": n_matches,
        },
        "overlay_mask": overlay_mask,
    }
