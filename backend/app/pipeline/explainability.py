"""Explainability Engine — GradCAM heatmaps + ELA overlay generation.

All heavy imports (torch) are lazy to allow the module to load
even when PyTorch is not installed — GradCAM simply becomes a no-op.
"""

from __future__ import annotations

import os
import uuid

import cv2
import numpy as np

from app.config import settings


def generate_gradcam(model, face_crop: np.ndarray, target_class: int = 1) -> np.ndarray:
    """Generate GradCAM heatmap for the given model and face crop.

    Returns heatmap overlay (BGR, same size as input).
    """
    import torch
    from torchvision import transforms

    transform = transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((380, 380)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])

    device = settings.resolved_device
    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)
    tensor = transform(rgb).unsqueeze(0).to(device)
    if settings.USE_FP16 and device == "cuda":
        tensor = tensor.half()

    # Hook to capture gradients and activations
    activations = []
    gradients = []

    target_layer = model.last_conv_layer

    def fwd_hook(module, inp, out):
        activations.append(out.detach())

    def bwd_hook(module, grad_in, grad_out):
        gradients.append(grad_out[0].detach())

    fwd_handle = target_layer.register_forward_hook(fwd_hook)
    bwd_handle = target_layer.register_full_backward_hook(bwd_hook)

    # Forward pass
    model.eval()
    if tensor.requires_grad is False:
        tensor.requires_grad_(True)
    output = model(tensor)
    score = output[0, target_class]

    # Backward pass
    model.zero_grad()
    score.backward()

    fwd_handle.remove()
    bwd_handle.remove()

    # Compute GradCAM
    act = activations[0].float()
    grad = gradients[0].float()

    weights = torch.mean(grad, dim=(2, 3), keepdim=True)
    cam = torch.sum(weights * act, dim=1).squeeze()
    cam = torch.relu(cam)
    cam = cam - cam.min()
    cam = cam / (cam.max() + 1e-8)

    # Resize to input size
    cam_np = cam.cpu().numpy()
    cam_resized = cv2.resize(cam_np, (face_crop.shape[1], face_crop.shape[0]))

    # Create color heatmap overlay
    heatmap = cv2.applyColorMap((cam_resized * 255).astype(np.uint8), cv2.COLORMAP_JET)
    overlay = cv2.addWeighted(face_crop, 0.55, heatmap, 0.45, 0)

    return overlay


def generate_ela_overlay(original: np.ndarray, ela_map: np.ndarray) -> np.ndarray:
    """Create ELA visualization overlay on original image."""
    ela_color = cv2.applyColorMap(ela_map, cv2.COLORMAP_HOT)
    ela_resized = cv2.resize(ela_color, (original.shape[1], original.shape[0]))
    overlay = cv2.addWeighted(original, 0.6, ela_resized, 0.4, 0)
    return overlay


def save_heatmap(image: np.ndarray, analysis_id: str, heatmap_type: str) -> str:
    """Save heatmap image to output directory. Returns file path."""
    output_dir = settings.OUTPUT_DIR / analysis_id
    output_dir.mkdir(parents=True, exist_ok=True)

    filename = f"{heatmap_type}_{uuid.uuid4().hex[:8]}.png"
    filepath = str(output_dir / filename)
    cv2.imwrite(filepath, image)
    return filepath


def generate_all_visuals(
    analysis_id: str,
    face_crop: np.ndarray | None,
    original: np.ndarray,
    results_raw: dict,
) -> list[dict]:
    """Generate all explainability visualizations. Returns list of {type, path}."""
    visuals = []

    # GradCAM for EfficientNet
    if face_crop is not None:
        try:
            from app.models.efficientnet import _load_model as load_eff
            eff_model = load_eff()
            gradcam = generate_gradcam(eff_model, face_crop, target_class=1)
            path = save_heatmap(gradcam, analysis_id, "gradcam_efficientnet")
            visuals.append({"type": "gradcam_efficientnet", "path": path})
        except Exception:
            pass

        try:
            from app.models.xception import _load_model as load_xcp
            xcp_model = load_xcp()
            gradcam_x = generate_gradcam(xcp_model, face_crop, target_class=1)
            path = save_heatmap(gradcam_x, analysis_id, "gradcam_xception")
            visuals.append({"type": "gradcam_xception", "path": path})
        except Exception:
            pass

    # ELA overlay
    if "ela" in results_raw:
        ela_details = results_raw["ela"].get("details", {})
        ela_map_data = ela_details.get("ela_map") if isinstance(ela_details, dict) else None
        if ela_map_data is None:
            ela_map_data = results_raw["ela"].get("ela_map")
        if ela_map_data is not None:
            try:
                ela_overlay = generate_ela_overlay(original, ela_map_data)
                path = save_heatmap(ela_overlay, analysis_id, "ela_overlay")
                visuals.append({"type": "ela", "path": path})
            except Exception:
                pass

    # Copy-move overlay
    if "copy_move" in results_raw:
        cm_details = results_raw["copy_move"].get("details", {})
        mask_data = cm_details.get("overlay_mask") if isinstance(cm_details, dict) else None
        if mask_data is None:
            mask_data = results_raw["copy_move"].get("overlay_mask")
        if mask_data is not None:
            try:
                mask_color = cv2.applyColorMap(mask_data, cv2.COLORMAP_AUTUMN)
                mask_resized = cv2.resize(mask_color, (original.shape[1], original.shape[0]))
                overlay = cv2.addWeighted(original, 0.7, mask_resized, 0.3, 0)
                path = save_heatmap(overlay, analysis_id, "copy_move_overlay")
                visuals.append({"type": "copy_move", "path": path})
            except Exception:
                pass

    return visuals
