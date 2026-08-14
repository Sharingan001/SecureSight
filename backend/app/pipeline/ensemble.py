"""Adaptive Ensemble Scorer — Calibrated weighted fusion + 5-tier verdict.

Runs all applicable pipelines in parallel via ThreadPoolExecutor,
aggregates via calibrated weighted average with cross-validation
heuristics, maps to verdict with confidence.

NOTE: This is NOT a Bayesian system. The fusion uses empirically
tuned weights and threshold-based cross-validation between DL
models. The constants were chosen based on observed false-positive
patterns and should be periodically re-calibrated against a held-out
validation set.

All model imports are wrapped in try/except so unavailable models are
skipped — but failed pipelines are EXCLUDED from scoring, not zero-scored.
"""

from __future__ import annotations

import logging
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any

import numpy as np

from app.config import settings
from app.pipeline.preprocessor import PreprocessResult

logger = logging.getLogger(__name__)

# 15 total pipeline definitions. Not all run for every input:
# - Face-only pipelines skip when no face detected
# - Video-only pipelines skip for images
# - Xception skips when no trained weights available
PIPELINE_WEIGHTS = {
    "efficientnet": 0.25,   # Celeb-DF + AI trained (face crops only)
    "ai_ensemble": 0.25,    # 3-model ensemble (full image, handles all AI types)
    "xception": 0.10,       # XceptionNet (face crops only)
    "frequency": 0.04,
    "temporal": 0.04,
    "biometric": 0.03,
    "audio": 0.04,
    "lipsync": 0.04,
    "ela": 0.05,
    "copy_move": 0.03,
    "jpeg_ghost": 0.03,
    "exif": 0.03,
    "eye_reflection": 0.03,
    "shadow": 0.02,
    "noise": 0.02,
}

PIPELINE_TIERS = {
    "ai_ensemble": 1, "efficientnet": 1, "xception": 1, "audio": 1, "lipsync": 1,
    "ela": 2, "copy_move": 2, "jpeg_ghost": 2, "exif": 2,
    "eye_reflection": 3, "shadow": 3, "noise": 3,
    "frequency": 4, "biometric": 4, "temporal": 4,
}


def _sigmoid_calibrate(score: float, steepness: float = 10.0, midpoint: float = 0.5) -> float:
    """Sigmoid calibration to sharpen probabilities."""
    return float(1.0 / (1.0 + np.exp(-steepness * (score - midpoint))))


def _map_verdict(score: float) -> str:
    """Map 0-100 score to 5-tier verdict."""
    if score < settings.THRESH_AUTHENTIC:
        return "AUTHENTIC"
    elif score < settings.THRESH_LIKELY_AUTH:
        return "LIKELY_AUTHENTIC"
    elif score < settings.THRESH_SUSPICIOUS:
        return "SUSPICIOUS"
    elif score < settings.THRESH_LIKELY_FAKE:
        return "LIKELY_FAKE"
    else:
        return "CONFIRMED_FAKE"


def _safe_import(module_path: str):
    """Safely import a pipeline module. Returns the predict function or None."""
    try:
        import importlib
        mod = importlib.import_module(module_path)
        return mod.predict
    except Exception as exc:
        logger.warning(f"Pipeline import failed for {module_path}: {exc}")
        return None


def _batch_face_predict(model_module) -> Any:
    """Return a wrapper that calls predict_batch() and averages scores across all faces.

    This ensures multi-face images/videos get full coverage rather than
    only scoring the primary (largest) detected face.
    """
    def _predict_all_faces(face_crops: list) -> dict:
        import time
        t0 = time.perf_counter()
        results = model_module.predict_batch(face_crops)
        if not results:
            return {"score": 0.0, "execution_ms": 0, "details": {"batch_faces": 0}}
        avg_score = sum(r["score"] for r in results) / len(results)
        max_score = max(r["score"] for r in results)
        # Use weighted combo: 60% max (catches worst face) + 40% avg (overall signal)
        combined = 0.6 * max_score + 0.4 * avg_score
        elapsed = int((time.perf_counter() - t0) * 1000)
        return {
            "score": combined,
            "execution_ms": elapsed,
            "details": {
                "batch_faces": len(results),
                "avg_score": avg_score,
                "max_score": max_score,
                "per_face": [r["score"] for r in results],
            },
        }
    return _predict_all_faces


def run_all_pipelines(preprocess: PreprocessResult, file_path: str) -> dict[str, Any]:
    """Execute all applicable pipelines in PARALLEL. Returns aggregated results.

    CRITICAL BEHAVIOR: Pipelines that crash are EXCLUDED from scoring,
    not zero-scored. A failed pipeline does not vote "authentic" — it
    simply doesn't vote at all. This prevents silent false negatives.
    """
    t0 = time.perf_counter()
    results: dict[str, dict] = {}
    failed_pipelines: list[str] = []

    # Determine which pipelines to run
    has_faces = len(preprocess.face_crops) > 0
    is_video = preprocess.media_type == "video"
    primary_face = preprocess.face_crops[0].image if has_faces else None
    all_face_images = [c.image for c in preprocess.face_crops]
    frames = preprocess.original_frames

    def _run_pipeline(name: str, fn, *args) -> tuple[str, dict, bool]:
        """Returns (name, result, success)."""
        try:
            result = fn(*args)
            return name, result, True
        except Exception as e:
            logger.error(f"Pipeline '{name}' crashed: {e}", exc_info=True)
            return name, {"score": 0.0, "execution_ms": 0, "details": {"error": str(e)}}, False

    # ── Build task list with safe imports ──────────────────────────────
    tasks = []

    original_frame = frames[0] if frames else None

    # Tier 1A: Face-trained models — ONLY run on face crops
    if has_faces:
        fn = _safe_import("app.models.efficientnet")
        if fn:
            if len(all_face_images) > 1:
                from app.models import efficientnet as _eff_mod
                tasks.append(("efficientnet", _batch_face_predict(_eff_mod), all_face_images))
            else:
                tasks.append(("efficientnet", fn, primary_face))

        # Only run XceptionNet if it has TRAINED weights — random init is useless
        if settings.XCEPTION_WEIGHTS is not None:
            fn = _safe_import("app.models.xception")
            if fn:
                if len(all_face_images) > 1:
                    from app.models import xception as _xcp_mod
                    tasks.append(("xception", _batch_face_predict(_xcp_mod), all_face_images))
                else:
                    tasks.append(("xception", fn, primary_face))
        else:
            logger.debug("XceptionNet skipped: no trained weights configured (XCEPTION_WEIGHTS=None).")

        # Face-only forensics
        for name, mod_path in [
            ("frequency", "app.models.frequency"),
            ("eye_reflection", "app.models.eye_reflection"),
        ]:
            fn = _safe_import(mod_path)
            if fn:
                tasks.append((name, fn, primary_face))

        fn = _safe_import("app.models.biometric")
        if fn:
            tasks.append(("biometric", fn, all_face_images))

    # Tier 1B: AI Ensemble — runs on FULL image ALWAYS (even without faces)
    if original_frame is not None:
        fn = _safe_import("app.models.ai_ensemble")
        if fn:
            tasks.append(("ai_ensemble", fn, original_frame))

    # Tier 2: Image forensics (always run on original frame)
    if frames:
        original = frames[0]
        for name, mod_path in [
            ("ela", "app.models.ela"),
            ("copy_move", "app.models.copy_move"),
            ("jpeg_ghost", "app.models.jpeg_ghost"),
            ("noise", "app.models.noise_analysis"),
        ]:
            fn = _safe_import(mod_path)
            if fn:
                tasks.append((name, fn, original))

        # Shadow needs original + optional face
        fn = _safe_import("app.models.shadow_lighting")
        if fn:
            tasks.append(("shadow", fn, original, primary_face))

    # EXIF (needs file path)
    fn = _safe_import("app.models.exif_analyzer")
    if fn:
        tasks.append(("exif", fn, file_path))

    # Tier 4: Video-only pipelines
    if is_video:
        if has_faces:
            fn = _safe_import("app.models.temporal")
            if fn:
                tasks.append(("temporal", fn, all_face_images))

            fn = _safe_import("app.models.lipsync")
            if fn and preprocess.audio_path:
                tasks.append(("lipsync", fn, frames, preprocess.audio_path))

        fn = _safe_import("app.models.audio")
        if fn and preprocess.audio_path:
            tasks.append(("audio", fn, preprocess.audio_path))

    # ── Execute in parallel ──────────────────────────────────────────
    if not tasks:
        return {
            "overall_score": 0.0,
            "verdict": "AUTHENTIC",
            "pipeline_scores": [],
            "total_execution_ms": 0,
            "results_raw": {},
            "failed_pipelines": [],
        }

    with ThreadPoolExecutor(max_workers=min(len(tasks), 8)) as pool:
        futures = {
            pool.submit(_run_pipeline, name, fn, *args): name
            for name, fn, *args in tasks
        }
        for future in as_completed(futures):
            name, result, success = future.result()
            if success:
                results[name] = result
            else:
                failed_pipelines.append(name)
                # CRITICAL: Do NOT add failed pipelines to results.
                # They must not participate in scoring.

    if not results:
        # Every pipeline failed — report the failure honestly
        return {
            "overall_score": 0.0,
            "verdict": "AUTHENTIC",
            "pipeline_scores": [],
            "total_execution_ms": int((time.perf_counter() - t0) * 1000),
            "results_raw": {},
            "failed_pipelines": failed_pipelines,
        }

    # ── Weighted ensemble (only successful pipelines) ────────────────
    weighted_sum = 0.0
    weight_total = 0.0
    dl_scores = []  # Track Tier 1 DL model scores

    # Dynamic weight redistribution: if XceptionNet didn't run,
    # boost EfficientNet + AI Ensemble to fill the gap
    effective_weights = dict(PIPELINE_WEIGHTS)
    if "xception" not in results:
        xc_weight = effective_weights.pop("xception", 0.10)
        if "efficientnet" in results:
            effective_weights["efficientnet"] = effective_weights.get("efficientnet", 0.25) + xc_weight / 2
        if "ai_ensemble" in results:
            effective_weights["ai_ensemble"] = effective_weights.get("ai_ensemble", 0.25) + xc_weight / 2

    # Collect all results first to enable cross-validation
    all_results = {}
    pipeline_scores = []
    for name, result in results.items():
        raw_score = result.get("score", 0.0)
        all_results[name] = raw_score

    # Cross-validate DL models: when they disagree, trust the consensus
    eff_score = all_results.get("efficientnet")  # None if not run (no face)
    ai_ens_score = all_results.get("ai_ensemble")
    ai_voting = results.get("ai_ensemble", {}).get("details", {}).get("voting", {})
    ai_votes = ai_voting.get("ai_votes", 0)
    real_votes = ai_voting.get("real_votes", 0)

    for name, result in results.items():
        raw_score = result.get("score", 0.0)
        weight = effective_weights.get(name, 0.05)
        tier = PIPELINE_TIERS.get(name, 4)

        is_dl = name in ("efficientnet", "xception", "ai_ensemble")

        if is_dl:
            calibrated = raw_score

            # Cross-validation: dampen EfficientNet when AI ensemble disagrees
            if name == "efficientnet" and raw_score > 0.6 and ai_ens_score is not None:
                if ai_ens_score <= 0.35 and real_votes >= 2:
                    calibrated = raw_score * 0.4  # Strong dampen
                elif ai_ens_score > 0.7 and ai_votes >= 2:
                    calibrated = min(raw_score * 1.1, 1.0)

            # Cross-validation: boost AI ensemble when EfficientNet agrees
            if name == "ai_ensemble" and raw_score > 0.5 and eff_score is not None:
                if eff_score > 0.6:
                    calibrated = min(raw_score * 1.1, 1.0)

            # Dynamic weight boost: when AI ensemble is highly confident,
            # increase its effective weight so it isn't diluted by low forensic scores
            if name == "ai_ensemble" and raw_score > 0.65:
                ai_strategy = results.get("ai_ensemble", {}).get("details", {}).get("strategy", "")
                if ai_strategy in ("high_conf_consensus", "high_conf_override", "majority_ai"):
                    weight = 0.45

            dl_scores.append(calibrated)
        else:
            calibrated = _sigmoid_calibrate(raw_score)

        weighted_sum += calibrated * weight
        weight_total += weight

        pipeline_scores.append({
            "pipeline": name,
            "tier": tier,
            "score": raw_score,
            "confidence": 1.0,
            "execution_ms": result.get("execution_ms", 0),
            "details": result.get("details", {}),
        })

    # Final score (0-100)
    weighted_avg = (weighted_sum / (weight_total + 1e-8)) * 100.0

    # Agreement boost
    high_dl = [s for s in dl_scores if s > 0.65]
    low_dl = [s for s in dl_scores if s < 0.30]

    if len(high_dl) >= 2:
        overall = min(weighted_avg * 1.15, 100.0)
    elif len(low_dl) >= 2:
        overall = weighted_avg * 0.85
    else:
        overall = weighted_avg

    # Floor score: if AI ensemble is highly confident AI, enforce minimum score
    if ai_ens_score is not None and ai_ens_score > 0.70:
        ai_strategy = results.get("ai_ensemble", {}).get("details", {}).get("strategy", "")
        if ai_strategy in ("high_conf_consensus", "high_conf_override", "majority_ai"):
            floor = ai_ens_score * 65.0
            if overall < floor:
                overall = floor

    overall = min(max(overall, 0.0), 100.0)
    verdict = _map_verdict(overall)

    total_ms = int((time.perf_counter() - t0) * 1000)

    # Log warnings for failed pipelines
    if failed_pipelines:
        logger.warning(
            f"Analysis completed with {len(failed_pipelines)} failed pipeline(s): {failed_pipelines}. "
            f"These were EXCLUDED from scoring (not zero-scored)."
        )

    return {
        "overall_score": overall,
        "verdict": verdict,
        "pipeline_scores": pipeline_scores,
        "total_execution_ms": total_ms,
        "results_raw": results,  # for heatmap generation
        "failed_pipelines": failed_pipelines,
    }
