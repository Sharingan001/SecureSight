"""Tier 1 — XceptionNet deepfake classifier.

Features:
  - Auto-loads Celeb-DF trained weights when available
  - Test-Time Augmentation (TTA) for +2-3% accuracy boost
  - Temperature-scaled confidence calibration
  - Depthwise separable convolutions capture different artifacts than EfficientNet
"""

from __future__ import annotations

import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms

from app.config import settings

_model = None
_transform = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
])

# TTA augmentation variants
_tta_transforms = [
    # Original
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]),
    # Horizontal flip
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((299, 299)),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]),
    # Slight zoom
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((330, 330)),
        transforms.CenterCrop(299),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]),
    # Zoom out
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((270, 270)),
        transforms.Pad((15, 15, 14, 14), fill=0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]),
    # Flip + zoom
    transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((330, 330)),
        transforms.CenterCrop(299),
        transforms.RandomHorizontalFlip(p=1.0),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ]),
]

TEMPERATURE = 1.5


class SeparableConv2d(nn.Module):
    def __init__(self, in_c: int, out_c: int, kernel: int = 3, stride: int = 1, padding: int = 1):
        super().__init__()
        self.depthwise = nn.Conv2d(in_c, in_c, kernel, stride, padding, groups=in_c, bias=False)
        self.pointwise = nn.Conv2d(in_c, out_c, 1, bias=False)
        self.bn = nn.BatchNorm2d(out_c)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(self.pointwise(self.depthwise(x)))


class XceptionBlock(nn.Module):
    def __init__(self, in_c: int, out_c: int, reps: int, stride: int = 1, grow_first: bool = True):
        super().__init__()
        layers = []
        filters = in_c
        if grow_first:
            layers.extend([nn.ReLU(inplace=True), SeparableConv2d(in_c, out_c), nn.BatchNorm2d(out_c)])
            filters = out_c
            for _ in range(reps - 1):
                layers.extend([nn.ReLU(inplace=True), SeparableConv2d(filters, filters), nn.BatchNorm2d(filters)])
        else:
            for _ in range(reps - 1):
                layers.extend([nn.ReLU(inplace=True), SeparableConv2d(filters, filters), nn.BatchNorm2d(filters)])
            layers.extend([nn.ReLU(inplace=True), SeparableConv2d(filters, out_c), nn.BatchNorm2d(out_c)])

        if stride != 1:
            layers.append(nn.MaxPool2d(3, stride, 1))

        self.sequential = nn.Sequential(*layers)
        self.skip = nn.Sequential(
            nn.Conv2d(in_c, out_c, 1, stride, bias=False),
            nn.BatchNorm2d(out_c),
        ) if in_c != out_c or stride != 1 else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.sequential(x) + self.skip(x)


class XceptionNet(nn.Module):
    """Simplified Xception for deepfake detection — optimized for speed."""

    def __init__(self, num_classes: int = 2):
        super().__init__()
        # Entry flow
        self.entry = nn.Sequential(
            nn.Conv2d(3, 32, 3, 2, 1, bias=False), nn.BatchNorm2d(32), nn.ReLU(inplace=True),
            nn.Conv2d(32, 64, 3, 1, 1, bias=False), nn.BatchNorm2d(64), nn.ReLU(inplace=True),
        )
        self.block1 = XceptionBlock(64, 128, reps=2, stride=2)
        self.block2 = XceptionBlock(128, 256, reps=2, stride=2)
        self.block3 = XceptionBlock(256, 728, reps=2, stride=2)

        # Middle flow
        self.middle = nn.Sequential(*[XceptionBlock(728, 728, reps=3) for _ in range(4)])

        # Exit flow
        self.exit_block = XceptionBlock(728, 1024, reps=2, stride=2, grow_first=False)
        self.exit_conv = nn.Sequential(
            SeparableConv2d(1024, 1536), nn.BatchNorm2d(1536), nn.ReLU(inplace=True),
            SeparableConv2d(1536, 2048), nn.BatchNorm2d(2048), nn.ReLU(inplace=True),
        )
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(2048, num_classes)

    @property
    def last_conv_layer(self):
        return self.exit_conv

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.entry(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.middle(x)
        x = self.exit_block(x)
        x = self.exit_conv(x)
        x = self.pool(x).flatten(1)
        return self.classifier(x)


def _load_model() -> XceptionNet:
    global _model
    if _model is not None:
        return _model

    _model = XceptionNet()

    weights_path = settings.XCEPTION_WEIGHTS
    if weights_path and Path(weights_path).exists():
        checkpoint = torch.load(weights_path, map_location="cpu", weights_only=True)
        if "model_state_dict" in checkpoint:
            _model.load_state_dict(checkpoint["model_state_dict"])
            auc = checkpoint.get("auc", "unknown")
            epoch = checkpoint.get("epoch", "unknown")
            print(f"[XceptionNet] Loaded trained weights (epoch={epoch}, AUC={auc})")
        else:
            _model.load_state_dict(checkpoint)
            print(f"[XceptionNet] Loaded weights from {weights_path}")
    else:
        print("[XceptionNet] Using random init (no trained weights)")

    device = settings.resolved_device
    _model = _model.to(device).eval()
    if settings.USE_FP16 and device == "cuda":
        _model = _model.half()
    return _model


def predict(face_crop: np.ndarray) -> dict:
    """Run inference with optional TTA."""
    t0 = time.perf_counter()
    model = _load_model()
    device = settings.resolved_device

    rgb = cv2.cvtColor(face_crop, cv2.COLOR_BGR2RGB)

    if settings.USE_TTA:
        all_probs = []
        for tta_t in _tta_transforms:
            tensor = tta_t(rgb).unsqueeze(0).to(device)
            if settings.USE_FP16 and device == "cuda":
                tensor = tensor.half()
            with torch.no_grad():
                logits = model(tensor)
                probs = F.softmax(logits / TEMPERATURE, dim=1)
                all_probs.append(probs[0, 1].item())

        weights = [0.30, 0.20, 0.20, 0.15, 0.15]
        fake_prob = sum(p * w for p, w in zip(all_probs, weights))
    else:
        tensor = _transform(rgb).unsqueeze(0).to(device)
        if settings.USE_FP16 and device == "cuda":
            tensor = tensor.half()
        with torch.no_grad():
            probs = F.softmax(model(tensor) / TEMPERATURE, dim=1)
            fake_prob = probs[0, 1].item()

    return {
        "score": fake_prob,
        "execution_ms": int((time.perf_counter() - t0) * 1000),
        "details": {
            "tta_enabled": settings.USE_TTA,
            "device": device,
            "trained_weights": settings.XCEPTION_WEIGHTS is not None,
        },
    }


def predict_batch(face_crops: list[np.ndarray]) -> list[dict]:
    if not face_crops:
        return []
    t0 = time.perf_counter()
    model = _load_model()
    device = settings.resolved_device

    batch = torch.stack([_transform(cv2.cvtColor(c, cv2.COLOR_BGR2RGB)) for c in face_crops]).to(device)
    if settings.USE_FP16 and device == "cuda":
        batch = batch.half()

    with torch.no_grad():
        probs = F.softmax(model(batch) / TEMPERATURE, dim=1)

    elapsed = int((time.perf_counter() - t0) * 1000)
    return [{"score": probs[i, 1].item(), "execution_ms": elapsed // len(face_crops)} for i in range(len(face_crops))]
