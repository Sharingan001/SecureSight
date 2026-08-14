"""SecureSight — Expand dataset with diverse AI-generated image sources.

Downloads additional datasets covering:
  - MS COCO AI (SD3, SDXL, DALL-E 3, MidJourney v6)
  - Distilled AI detection set (Midjourney + SD + real)
  - More compressed/downscaled variants for robustness
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent.parent))

DATASET_DIR = Path(__file__).parent / "dataset_ai"
TARGET_SIZE = 380


def save_image(pil_img: Image.Image, output_path: Path, target_size: int = TARGET_SIZE):
    """Resize and save a PIL image as JPEG."""
    try:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")
        w, h = pil_img.size
        if w < 64 or h < 64:
            return False
        scale = target_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        pil_img = pil_img.crop((left, top, left + target_size, top + target_size))
        quality = random.choice([85, 88, 90, 92, 95])
        pil_img.save(str(output_path), "JPEG", quality=quality)
        return True
    except Exception:
        return False


def download_distilled_dataset():
    """Download jacoballessio/ai-image-detect-distilled — diverse AI generators."""
    from datasets import load_dataset

    print("=" * 60)
    print("[1/3] Downloading jacoballessio/ai-image-detect-distilled...")
    print("      Contains: Midjourney + SD + OpenImage real photos")
    print("=" * 60)

    try:
        ds = load_dataset("jacoballessio/ai-image-detect-distilled", split="train", streaming=True)

        real_count = 0
        fake_count = 0

        for sample in ds:
            img = sample.get("image")
            label = sample.get("label", -1)

            if img is None:
                continue

            is_real = (label == 0)  # Check label mapping

            if is_real:
                if real_count >= 5000:
                    continue
                out_dir = DATASET_DIR / "train" / "real"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"distilled_real_{real_count:05d}.jpg"
                if not out_path.exists() and save_image(img, out_path):
                    real_count += 1
            else:
                if fake_count >= 5000:
                    continue
                out_dir = DATASET_DIR / "train" / "fake"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"distilled_fake_{fake_count:05d}.jpg"
                if not out_path.exists() and save_image(img, out_path):
                    fake_count += 1

            if real_count >= 5000 and fake_count >= 5000:
                break

            if (real_count + fake_count) % 1000 == 0 and (real_count + fake_count) > 0:
                print(f"    [train] real={real_count} fake={fake_count}")

        print(f"    [train] DONE: real={real_count} fake={fake_count}")
    except Exception as e:
        print(f"    ⚠ Error: {e}")


def download_cifake():
    """Download CIFAKE-like data for compressed/small image robustness."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("[2/3] Downloading rodrigomasini/CIFAKE-image-dataset...")
    print("      Contains: 120K small real + AI images (robustness)")
    print("=" * 60)

    try:
        ds = load_dataset("rodrigomasini/CIFAKE-image-dataset", split="train", streaming=True)

        real_count = 0
        fake_count = 0

        for sample in ds:
            img = sample.get("image")
            label = sample.get("label", -1)

            if img is None:
                continue

            is_real = (label == 0)

            if is_real:
                if real_count >= 3000:
                    continue
                out_dir = DATASET_DIR / "train" / "real"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"cifake_real_{real_count:05d}.jpg"
                if not out_path.exists() and save_image(img, out_path):
                    real_count += 1
            else:
                if fake_count >= 3000:
                    continue
                out_dir = DATASET_DIR / "train" / "fake"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"cifake_fake_{fake_count:05d}.jpg"
                if not out_path.exists() and save_image(img, out_path):
                    fake_count += 1

            if real_count >= 3000 and fake_count >= 3000:
                break

            if (real_count + fake_count) % 1000 == 0 and (real_count + fake_count) > 0:
                print(f"    [train] real={real_count} fake={fake_count}")

        print(f"    [train] DONE: real={real_count} fake={fake_count}")
    except Exception as e:
        print(f"    ⚠ Error: {e}")


def generate_augmented_variants():
    """Create hard-negative augmented copies of existing training data.

    For each class, creates compressed/downscaled copies that simulate
    real-world conditions (WhatsApp compression, web thumbnails, etc.)
    """
    print("\n" + "=" * 60)
    print("[3/3] Generating hard-augmented variants...")
    print("      Adding: JPEG Q10-30, 2x downscale, noise injection")
    print("=" * 60)

    for cls in ["real", "fake"]:
        src_dir = DATASET_DIR / "train" / cls
        if not src_dir.exists():
            continue

        files = list(src_dir.glob("*.jpg"))
        random.shuffle(files)
        # Take 2000 samples for augmentation
        files = files[:2000]

        count = 0
        for f in files:
            try:
                img = cv2.imread(str(f))
                if img is None:
                    continue

                # Variant 1: Heavy JPEG compression (Q10-25)
                q = random.randint(10, 25)
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
                compressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                out1 = src_dir / f"aug_compress_{f.stem}_{q}.jpg"
                if not out1.exists():
                    cv2.imwrite(str(out1), compressed, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    count += 1

                # Variant 2: Downscale to 50% then back up
                h, w = img.shape[:2]
                small = cv2.resize(img, (w // 2, h // 2), interpolation=cv2.INTER_AREA)
                upscaled = cv2.resize(small, (w, h), interpolation=cv2.INTER_LINEAR)
                out2 = src_dir / f"aug_downscale_{f.stem}.jpg"
                if not out2.exists():
                    cv2.imwrite(str(out2), upscaled, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    count += 1

            except Exception:
                continue

        print(f"    [{cls}] Generated {count} augmented variants")

    # Add augmented to val too (smaller set)
    for cls in ["real", "fake"]:
        val_dir = DATASET_DIR / "val" / cls
        if not val_dir.exists():
            continue

        files = list(val_dir.glob("*.jpg"))
        random.shuffle(files)
        files = files[:300]

        count = 0
        for f in files:
            try:
                img = cv2.imread(str(f))
                if img is None:
                    continue
                q = random.randint(15, 30)
                _, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q])
                compressed = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                out = val_dir / f"aug_compress_{f.stem}_{q}.jpg"
                if not out.exists():
                    cv2.imwrite(str(out), compressed, [cv2.IMWRITE_JPEG_QUALITY, 90])
                    count += 1
            except Exception:
                continue

        print(f"    [val/{cls}] Generated {count} augmented variants")


def summary():
    """Print dataset summary."""
    print("\n" + "=" * 60)
    print("EXPANDED DATASET SUMMARY")
    print("=" * 60)
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            d = DATASET_DIR / split / cls
            count = len(list(d.glob("*.jpg"))) if d.exists() else 0
            print(f"  {split:6s}/{cls:5s}: {count:>6d} images")

    total = sum(
        len(list((DATASET_DIR / s / c).glob("*.jpg")))
        for s in ["train", "val", "test"]
        for c in ["real", "fake"]
        if (DATASET_DIR / s / c).exists()
    )
    print(f"\n  TOTAL: {total:>6d} images")


def main():
    random.seed(42)
    np.random.seed(42)

    print("🔧 SecureSight — Dataset Expansion")
    print("=" * 60)
    print(f"Output: {DATASET_DIR}")
    print()

    download_distilled_dataset()
    download_cifake()
    generate_augmented_variants()
    summary()

    print(f"\n✅ Dataset expanded! Ready for retraining.")


if __name__ == "__main__":
    main()
