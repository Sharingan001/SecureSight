"""
SecureSight — Industry-Grade Multi-Model AI Image Detector

Ensembles 3 best pre-trained models from HuggingFace:
1. haywoodsloan/ai-image-detector-dev-deploy (160K downloads, ViT-based)
   - Best at: Real photo verification, low false positive rate
2. umm-maybe/AI-image-detector (42K downloads, CLIP-based)
   - Best at: Balanced detection, understanding image semantics
3. Ateeqq/ai-vs-human-image-detector (44K downloads, SigLIP+DINOv2)
   - Best at: Catching compressed/small AI images, highest sensitivity

Scoring strategy:
  - If ANY model is >95% confident → trust that signal (high-confidence override)
  - If 2+ models agree (>60%) → consensus boost
  - Otherwise → weighted average with model-specific strengths
"""

from __future__ import annotations

import time
import cv2
import numpy as np
from PIL import Image

_models: dict = {}
_model_configs = {
    "haywoodsloan": {
        "repo": "haywoodsloan/ai-image-detector-dev-deploy",
        "fake_label": "artificial",
        "real_label": "real",
        "weight": 0.30,
        "strength": "real_verification",  # Best at confirming real
    },
    "umm_maybe": {
        "repo": "umm-maybe/AI-image-detector",
        "fake_label": "artificial",
        "real_label": "human",
        "weight": 0.30,
        "strength": "balanced",
    },
    "ateeqq": {
        "repo": "Ateeqq/ai-vs-human-image-detector",
        "fake_label": "ai",
        "real_label": "hum",
        "weight": 0.40,  # Highest weight — most sensitive, catches compressed AI
        "strength": "ai_sensitivity",
    },
}


def _get_device() -> int:
    """Return torch device index: 0 for CUDA GPU, -1 for CPU."""
    try:
        import torch
        return 0 if torch.cuda.is_available() else -1
    except ImportError:
        return -1


def _load_models():
    """Load all 3 detection models (cached after first call)."""
    global _models
    if _models:
        return _models

    from transformers import pipeline as hf_pipeline

    device = _get_device()
    device_label = "GPU" if device == 0 else "CPU"

    for name, cfg in _model_configs.items():
        try:
            pipe = hf_pipeline(
                "image-classification",
                model=cfg["repo"],
                device=device,
            )
            _models[name] = pipe
            print(f"[AI Ensemble] ✓ Loaded {name} on {device_label}: {cfg['repo']}")
        except Exception as e:
            print(f"[AI Ensemble] ✗ Failed {name}: {e}")

    return _models


def predict(image_input) -> dict:
    """
    Industry-grade AI detection using 3-model ensemble with smart fusion.

    Strategy:
      1. Run all 3 models
      2. If any model >95% confident → high-confidence override
      3. If 2+ models agree (>60%) → consensus-boosted average
      4. Else → weighted average

    Args:
        image_input: BGR numpy array (from cv2) or PIL Image

    Returns:
        dict with 'score' (0-1), 'execution_ms', 'details'
    """
    t0 = time.perf_counter()

    # Convert to PIL
    if isinstance(image_input, np.ndarray):
        rgb = cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB)
        pil_img = Image.fromarray(rgb)
    elif isinstance(image_input, Image.Image):
        pil_img = image_input
    else:
        return {"score": 0.0, "execution_ms": 0, "details": {"error": "Invalid input"}}

    if pil_img.mode != "RGB":
        pil_img = pil_img.convert("RGB")

    models = _load_models()
    if not models:
        return {"score": 0.0, "execution_ms": 0, "details": {"error": "No models loaded"}}

    # ── Run each model ──────────────────────────────────────────────
    model_scores = {}
    model_details = {}

    for name, pipe in models.items():
        cfg = _model_configs[name]
        try:
            result = pipe(pil_img)
            scores = {r["label"]: r["score"] for r in result}
            fake_score = scores.get(cfg["fake_label"], 0.0)

            model_scores[name] = fake_score
            model_details[name] = {
                "ai_score": fake_score,
                "raw": scores,
                "weight": cfg["weight"],
                "strength": cfg["strength"],
            }
        except Exception as e:
            model_details[name] = {"error": str(e)}

    if not model_scores:
        ms = int((time.perf_counter() - t0) * 1000)
        return {"score": 0.0, "execution_ms": ms, "details": {"error": "All models failed"}}

    # ── Smart fusion strategy ───────────────────────────────────────
    scores_list = list(model_scores.values())

    # Strategy 1: High-confidence override
    # If ANY model is >95% confident, it's a very strong signal
    max_score = max(scores_list)
    max_model = max(model_scores, key=model_scores.get)
    min_score = min(scores_list)
    min_model = min(model_scores, key=model_scores.get)

    # Count votes
    ai_votes = sum(1 for s in scores_list if s > 0.60)
    real_votes = sum(1 for s in scores_list if s < 0.40)
    uncertain = sum(1 for s in scores_list if 0.40 <= s <= 0.60)

    strategy_used = "weighted_avg"

    if max_score > 0.95 and ai_votes >= 2:
        # Very high confidence + majority agree → strong AI signal
        ensemble_score = max_score * 0.85 + sum(scores_list) / len(scores_list) * 0.15
        strategy_used = "high_conf_consensus"

    elif max_score > 0.90 and ai_votes == 1 and real_votes >= 2:
        # OUTLIER REJECTION: One model says AI but both others strongly say real
        # Check if the disagreeing models are very confident it's real
        other_scores = [s for n, s in model_scores.items() if n != max_model]
        others_confident_real = all(s < 0.20 for s in other_scores)

        if others_confident_real:
            # Strong outlier — the high-confidence model is a false positive
            # Use the majority opinion (real) with slight uncertainty
            avg_others = sum(other_scores) / len(other_scores)
            ensemble_score = avg_others * 0.7 + max_score * 0.10  # Mostly trust majority
            strategy_used = "outlier_rejected"
        else:
            # Others aren't that confident it's real — partial override
            avg = sum(scores_list) / len(scores_list)
            ensemble_score = max_score * 0.45 + avg * 0.55
            strategy_used = "high_conf_contested"

    elif max_score > 0.95 and ai_votes == 1:
        # One model extremely confident, others uncertain
        avg = sum(scores_list) / len(scores_list)
        ensemble_score = max_score * 0.50 + avg * 0.50
        strategy_used = "high_conf_override"

    elif ai_votes >= 2:
        # Majority says AI → boost
        weighted = sum(
            model_scores[n] * _model_configs[n]["weight"]
            for n in model_scores
        ) / sum(_model_configs[n]["weight"] for n in model_scores)
        ensemble_score = max(weighted, 0.65)  # Floor at 65% if 2+ agree
        strategy_used = "majority_ai"

    elif real_votes >= 2:
        # Majority says real → dampen
        weighted = sum(
            model_scores[n] * _model_configs[n]["weight"]
            for n in model_scores
        ) / sum(_model_configs[n]["weight"] for n in model_scores)
        ensemble_score = min(weighted, 0.30)  # Cap at 30% if 2+ agree real
        strategy_used = "majority_real"

    else:
        # No clear consensus → weighted average
        weighted = sum(
            model_scores[n] * _model_configs[n]["weight"]
            for n in model_scores
        ) / sum(_model_configs[n]["weight"] for n in model_scores)
        ensemble_score = weighted
        strategy_used = "weighted_avg"

    ensemble_score = float(np.clip(ensemble_score, 0.0, 1.0))

    ms = int((time.perf_counter() - t0) * 1000)

    return {
        "score": ensemble_score,
        "execution_ms": ms,
        "details": {
            "model": "3-model-ensemble",
            "strategy": strategy_used,
            "models": model_details,
            "voting": {
                "ai_votes": ai_votes,
                "real_votes": real_votes,
                "uncertain": uncertain,
                "total_models": len(model_scores),
            },
        },
    }
