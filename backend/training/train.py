"""SecureSight — Production Training Pipeline.

Trains EfficientNet-B4 and XceptionNet deepfake detectors with:
  - AMP (Automatic Mixed Precision) for GPU acceleration
  - CPU fallback when no GPU available
  - Cosine annealing LR with warmup
  - Label smoothing + class-weighted loss
  - Early stopping with patience
  - Best model checkpoint saving
  - Full evaluation with ROC-AUC, accuracy, precision, recall, F1
  - TensorBoard-compatible logging
"""

from __future__ import annotations

import os
import json
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms

# Add parent path for app imports
sys.path.insert(0, str(Path(__file__).parent.parent))

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
USE_AMP = DEVICE == "cuda"
IMG_SIZE = 380

DATASET_DIR = Path(__file__).parent / "dataset"
WEIGHTS_DIR = Path(__file__).parent.parent / "weights"


# ══════════════════════════════════════════════════════════════════════
# AUGMENTATION
# ══════════════════════════════════════════════════════════════════════

class JPEGCompress:
    """Simulate JPEG compression artifacts (module-level for pickling)."""
    def __init__(self, quality_range=(10, 40)):
        self.quality_range = quality_range
    def __call__(self, img):
        import io
        from PIL import Image
        q = random.randint(*self.quality_range)
        buf = io.BytesIO()
        img.save(buf, format='JPEG', quality=q)
        buf.seek(0)
        return Image.open(buf).convert('RGB')


class DownscaleUpscale:
    """Simulate low-res image (module-level for pickling)."""
    def __init__(self, scale_range=(0.25, 0.5)):
        self.scale_range = scale_range
    def __call__(self, img):
        from PIL import Image
        w, h = img.size
        scale = random.uniform(*self.scale_range)
        small = img.resize((max(16, int(w*scale)), max(16, int(h*scale))), Image.BILINEAR)
        return small.resize((w, h), Image.BILINEAR)

def get_train_transforms(size: int = IMG_SIZE):
    """Aggressive training augmentations optimized for deepfake detection."""
    from PIL import Image

    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.RandomResizedCrop(size, scale=(0.6, 1.0), ratio=(0.85, 1.15)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomApply([
            transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.2, hue=0.05),
        ], p=0.7),
        transforms.RandomGrayscale(p=0.03),
        transforms.RandomApply([
            transforms.GaussianBlur(kernel_size=5, sigma=(0.1, 2.5)),
        ], p=0.3),
        transforms.RandomApply([
            transforms.RandomRotation(degrees=10),
        ], p=0.2),
        transforms.RandomApply([JPEGCompress()], p=0.3),
        transforms.RandomApply([DownscaleUpscale()], p=0.2),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.2)),
    ])


def get_val_transforms(size: int = IMG_SIZE):
    return transforms.Compose([
        transforms.ToPILImage(),
        transforms.Resize((size, size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
    ])


# ══════════════════════════════════════════════════════════════════════
# DATASET
# ══════════════════════════════════════════════════════════════════════

class DeepfakeDataset(Dataset):
    """Generic deepfake dataset loader.

    Directory structure:
        root/
        ├── real/    → label 0
        └── fake/    → label 1
    """

    EXTENSIONS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}

    def __init__(self, root: str, transform=None, max_samples: int = 0):
        self.transform = transform
        self.samples: list[tuple[str, int]] = []

        root = Path(root)
        for label_name, label_id in [("real", 0), ("fake", 1)]:
            label_dir = root / label_name
            if not label_dir.exists():
                continue
            for img_path in sorted(label_dir.iterdir()):
                if img_path.suffix.lower() in self.EXTENSIONS:
                    self.samples.append((str(img_path), label_id))

        random.shuffle(self.samples)
        if max_samples > 0:
            self.samples = self.samples[:max_samples]

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int):
        path, label = self.samples[idx]
        img = cv2.imread(path)
        if img is None:
            img = np.zeros((IMG_SIZE, IMG_SIZE, 3), dtype=np.uint8)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

        if self.transform:
            img = self.transform(img)
        else:
            img = transforms.ToTensor()(img)

        return img, label

    def class_counts(self) -> tuple[int, int]:
        real = sum(1 for _, l in self.samples if l == 0)
        fake = sum(1 for _, l in self.samples if l == 1)
        return real, fake


# ══════════════════════════════════════════════════════════════════════
# TRAINING LOOP
# ══════════════════════════════════════════════════════════════════════

def train_model(
    model: nn.Module,
    train_dir: str,
    val_dir: str,
    epochs: int = 30,
    batch_size: int = 16,
    lr: float = 3e-4,
    weight_decay: float = 1e-4,
    warmup_epochs: int = 3,
    patience: int = 7,
    save_path: str = "best_model.pth",
    device: str = DEVICE,
    img_size: int = IMG_SIZE,
):
    """Train a deepfake detector with full optimization."""
    print(f"\n{'='*60}")
    print(f"Training configuration:")
    print(f"  Device:      {device.upper()}")
    print(f"  AMP:         {USE_AMP}")
    print(f"  Epochs:      {epochs}")
    print(f"  Batch size:  {batch_size}")
    print(f"  LR:          {lr}")
    print(f"  Image size:  {img_size}")
    print(f"  Save path:   {save_path}")
    print(f"{'='*60}\n")

    model = model.to(device)

    # Datasets
    train_ds = DeepfakeDataset(train_dir, transform=get_train_transforms(img_size))
    val_ds = DeepfakeDataset(val_dir, transform=get_val_transforms(img_size))

    real_count, fake_count = train_ds.class_counts()
    print(f"[📊] Train: {len(train_ds)} samples ({real_count} real, {fake_count} fake)")
    real_count_v, fake_count_v = val_ds.class_counts()
    print(f"[📊] Val:   {len(val_ds)} samples ({real_count_v} real, {fake_count_v} fake)")

    # Determine number of workers based on system
    num_workers = min(4, os.cpu_count() or 1)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True,
        num_workers=num_workers, pin_memory=(device == "cuda"), drop_last=True,
    )
    val_loader = DataLoader(
        val_ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=(device == "cuda"),
    )

    # Class weights for imbalanced data
    if real_count > 0 and fake_count > 0:
        total = real_count + fake_count
        w_real = total / (2 * real_count)
        w_fake = total / (2 * fake_count)
        class_weights = torch.tensor([w_real, w_fake], dtype=torch.float32).to(device)
        print(f"[⚖] Class weights: real={w_real:.3f}, fake={w_fake:.3f}")
    else:
        class_weights = None

    # Optimizer + scheduler
    optimizer = torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=epochs - warmup_epochs)
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.05)

    # AMP scaler (GPU only)
    scaler = torch.amp.GradScaler(device) if USE_AMP else None

    best_auc = 0.0
    no_improve = 0
    history = {"train_loss": [], "val_auc": [], "val_acc": [], "lr": []}

    print(f"\n{'─'*60}")
    print(f"{'Epoch':>6} {'Loss':>10} {'AUC':>8} {'Acc':>8} {'LR':>12} {'Time':>8}")
    print(f"{'─'*60}")

    for epoch in range(epochs):
        epoch_start = time.time()

        # ── Warmup LR ─────────────────────────────────────────
        if epoch < warmup_epochs:
            warmup_lr = lr * (epoch + 1) / warmup_epochs
            for pg in optimizer.param_groups:
                pg['lr'] = warmup_lr

        # ── Train ─────────────────────────────────────────────
        model.train()
        running_loss = 0.0
        n_batches = 0

        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device)
            optimizer.zero_grad(set_to_none=True)

            if USE_AMP:
                with torch.amp.autocast(device):
                    outputs = model(images)
                    loss = criterion(outputs, labels)
                scaler.scale(loss).backward()
                scaler.unscale_(optimizer)
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                scaler.step(optimizer)
                scaler.update()
            else:
                outputs = model(images)
                loss = criterion(outputs, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

            running_loss += loss.item()
            n_batches += 1

        if epoch >= warmup_epochs:
            scheduler.step()

        avg_loss = running_loss / max(n_batches, 1)

        # ── Validate ──────────────────────────────────────────
        model.eval()
        all_probs = []
        all_labels = []
        correct = 0
        total = 0

        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)

                if USE_AMP:
                    with torch.amp.autocast(device):
                        outputs = model(images)
                else:
                    outputs = model(images)

                probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
                preds = (probs > 0.5).astype(int)
                all_probs.extend(probs.tolist())
                all_labels.extend(labels.numpy().tolist())
                correct += (preds == labels.numpy()).sum()
                total += len(labels)

        # Compute metrics
        try:
            from sklearn.metrics import roc_auc_score
            auc = roc_auc_score(all_labels, all_probs)
        except (ImportError, ValueError):
            auc = 0.0

        acc = correct / max(total, 1)
        current_lr = optimizer.param_groups[0]['lr']
        elapsed = time.time() - epoch_start

        # Log
        history["train_loss"].append(avg_loss)
        history["val_auc"].append(auc)
        history["val_acc"].append(float(acc))
        history["lr"].append(current_lr)

        marker = ""
        if auc > best_auc:
            best_auc = auc
            no_improve = 0
            os.makedirs(os.path.dirname(save_path) if os.path.dirname(save_path) else ".", exist_ok=True)
            torch.save({
                "model_state_dict": model.state_dict(),
                "epoch": epoch + 1,
                "auc": auc,
                "accuracy": float(acc),
            }, save_path)
            marker = " ★ BEST"
        else:
            no_improve += 1

        print(f"{epoch+1:>5d}/{epochs} {avg_loss:>10.4f} {auc:>8.4f} {acc:>7.1%} {current_lr:>12.2e} {elapsed:>6.1f}s{marker}")

        if no_improve >= patience:
            print(f"\n[⏹] Early stopping — no improvement for {patience} epochs")
            break

    print(f"\n{'='*60}")
    print(f"Training complete!")
    print(f"  Best AUC:      {best_auc:.4f}")
    print(f"  Best model at: {save_path}")
    print(f"{'='*60}\n")

    # Save training history
    hist_path = save_path.replace(".pth", "_history.json")
    with open(hist_path, "w") as f:
        json.dump(history, f, indent=2)

    return best_auc


# ══════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════

def evaluate_model(model: nn.Module, test_dir: str, device: str = DEVICE,
                   img_size: int = IMG_SIZE):
    """Full evaluation with ROC-AUC, accuracy, precision, recall, F1."""
    model = model.to(device).eval()

    test_ds = DeepfakeDataset(test_dir, transform=get_val_transforms(img_size))
    test_loader = DataLoader(test_ds, batch_size=16, shuffle=False,
                             num_workers=min(4, os.cpu_count() or 1))

    print(f"\n[🔍] Evaluating on {len(test_ds)} samples...")

    all_probs = []
    all_preds = []
    all_labels = []

    with torch.no_grad():
        for images, labels in test_loader:
            images = images.to(device)

            if USE_AMP:
                with torch.amp.autocast(device):
                    outputs = model(images)
            else:
                outputs = model(images)

            probs = F.softmax(outputs, dim=1)[:, 1].cpu().numpy()
            preds = (probs > 0.5).astype(int)
            all_probs.extend(probs.tolist())
            all_preds.extend(preds.tolist())
            all_labels.extend(labels.numpy().tolist())

    # Metrics
    try:
        from sklearn.metrics import roc_auc_score, classification_report, confusion_matrix
        auc = roc_auc_score(all_labels, all_probs)
        report = classification_report(all_labels, all_preds,
                                        target_names=["Real", "Fake"], digits=4)
        cm = confusion_matrix(all_labels, all_preds)

        print(f"\nROC-AUC: {auc:.4f}")
        print(f"\n{report}")
        print(f"Confusion Matrix:")
        print(f"  {'':>10} Pred Real  Pred Fake")
        print(f"  {'True Real':>10} {cm[0][0]:>9d} {cm[0][1]:>9d}")
        print(f"  {'True Fake':>10} {cm[1][0]:>9d} {cm[1][1]:>9d}")

        return {"auc": auc, "report": report, "confusion_matrix": cm.tolist()}
    except ImportError:
        acc = sum(1 for p, l in zip(all_preds, all_labels) if p == l) / len(all_labels)
        print(f"Accuracy: {acc:.4f}")
        return {"accuracy": acc}


# ══════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="SecureSight — Train Deepfake Detectors")
    parser.add_argument("--model", choices=["efficientnet", "xception", "both"],
                        default="both", help="Which model to train")
    parser.add_argument("--train-dir", default=str(DATASET_DIR / "train"),
                        help="Training data directory")
    parser.add_argument("--val-dir", default=str(DATASET_DIR / "val"),
                        help="Validation data directory")
    parser.add_argument("--test-dir", default=str(DATASET_DIR / "test"),
                        help="Test data directory")
    parser.add_argument("--epochs", type=int, default=20)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--lr", type=float, default=3e-4)
    parser.add_argument("--evaluate", action="store_true", help="Evaluate only")
    args = parser.parse_args()

    WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)

    models_to_train = []

    if args.model in ("efficientnet", "both"):
        from app.models.efficientnet import EfficientNetDetector
        models_to_train.append((
            "EfficientNet-B4",
            EfficientNetDetector(),
            str(WEIGHTS_DIR / "efficientnet_b4_deepfake.pth"),
        ))

    if args.model in ("xception", "both"):
        from app.models.xception import XceptionNet
        models_to_train.append((
            "XceptionNet",
            XceptionNet(),
            str(WEIGHTS_DIR / "xception_deepfake.pth"),
        ))

    for name, model, save_path in models_to_train:
        print(f"\n{'#'*60}")
        print(f"# {name}")
        print(f"{'#'*60}")

        if args.evaluate:
            if os.path.exists(save_path):
                ckpt = torch.load(save_path, map_location=DEVICE, weights_only=True)
                model.load_state_dict(ckpt["model_state_dict"])
                print(f"Loaded weights from {save_path}")
            evaluate_model(model, args.test_dir)
        else:
            train_model(
                model,
                train_dir=args.train_dir,
                val_dir=args.val_dir,
                epochs=args.epochs,
                batch_size=args.batch_size,
                lr=args.lr,
                save_path=save_path,
            )
            # Auto-evaluate after training
            if os.path.exists(args.test_dir):
                ckpt = torch.load(save_path, map_location=DEVICE, weights_only=True)
                model.load_state_dict(ckpt["model_state_dict"])
                evaluate_model(model, args.test_dir)
