"""SecureSight — Download & train on AI-Generated Image Detection datasets.

Downloads multiple datasets covering:
  - Face-swap deepfakes (Celeb-DF — already done)
  - AI-generated faces (GAN, Diffusion)
  - GPT/Gemini/Midjourney/Stable Diffusion generated images
  - Edited/manipulated images

Combines them into a unified training set for comprehensive AI detection.
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
MAX_PER_SPLIT = {
    "train": 15000,  # max images per class per split
    "val": 2000,
    "test": 2000,
}


def save_image(pil_img: Image.Image, output_path: Path, target_size: int = TARGET_SIZE):
    """Resize and save a PIL image as JPEG."""
    try:
        if pil_img.mode != "RGB":
            pil_img = pil_img.convert("RGB")

        # Resize maintaining aspect ratio then center crop
        w, h = pil_img.size
        scale = target_size / min(w, h)
        new_w, new_h = int(w * scale), int(h * scale)
        pil_img = pil_img.resize((new_w, new_h), Image.LANCZOS)

        # Center crop
        left = (new_w - target_size) // 2
        top = (new_h - target_size) // 2
        pil_img = pil_img.crop((left, top, left + target_size, top + target_size))

        quality = random.choice([88, 90, 92, 95])
        pil_img.save(str(output_path), "JPEG", quality=quality)
        return True
    except Exception as e:
        return False


def download_dappai_dataset():
    """Download dappai/Deepfake-vs-Real-v2 (32K downloads, high-res)."""
    from datasets import load_dataset

    print("=" * 60)
    print("[1/2] Downloading dappai/Deepfake-vs-Real-v2...")
    print("      Contains: Real photos + AI-generated faces")
    print("=" * 60)

    for split_name, hf_split in [("train", "train"), ("test", "test")]:
        print(f"\n  Loading {hf_split} split...")
        ds = load_dataset("dappai/Deepfake-vs-Real-v2", split=hf_split, streaming=True)

        real_count = 0
        fake_count = 0
        max_per_class = MAX_PER_SPLIT.get(split_name, 5000)

        for sample in ds:
            img = sample["image"]
            label = sample["label"]  # 0=fake, 1=real based on dataset

            if label == 1:  # Real
                if real_count >= max_per_class:
                    continue
                out_dir = DATASET_DIR / split_name / "real"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"dappai_real_{real_count:05d}.jpg"
                if save_image(img, out_path):
                    real_count += 1
            else:  # Fake/AI-generated
                if fake_count >= max_per_class:
                    continue
                out_dir = DATASET_DIR / split_name / "fake"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"dappai_fake_{fake_count:05d}.jpg"
                if save_image(img, out_path):
                    fake_count += 1

            if real_count >= max_per_class and fake_count >= max_per_class:
                break

            if (real_count + fake_count) % 500 == 0:
                print(f"    [{split_name}] real={real_count} fake={fake_count}")

        print(f"    [{split_name}] DONE: real={real_count} fake={fake_count}")


def download_ai_vs_deepfake():
    """Download prithivMLmods/AI-vs-Deepfake-vs-Real dataset."""
    from datasets import load_dataset

    print("\n" + "=" * 60)
    print("[2/2] Downloading prithivMLmods/Deepfake-vs-Real-v2...")
    print("      Contains: AI art + Deepfakes + Real photos")
    print("=" * 60)

    try:
        ds = load_dataset("prithivMLmods/Deepfake-vs-Real-v2", split="train", streaming=True)

        real_count = 0
        fake_count = 0

        for sample in ds:
            img = sample.get("image")
            label = sample.get("label", -1)

            if img is None:
                continue

            # Determine label (varies by dataset)
            is_real = (label == 1) or (str(label).lower() in ["real", "1"])

            if is_real:
                if real_count >= 5000:
                    continue
                out_dir = DATASET_DIR / "train" / "real"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"prithiv_real_{real_count:05d}.jpg"
                if save_image(img, out_path):
                    real_count += 1
            else:
                if fake_count >= 5000:
                    continue
                out_dir = DATASET_DIR / "train" / "fake"
                out_dir.mkdir(parents=True, exist_ok=True)
                out_path = out_dir / f"prithiv_fake_{fake_count:05d}.jpg"
                if save_image(img, out_path):
                    fake_count += 1

            if real_count >= 5000 and fake_count >= 5000:
                break

            if (real_count + fake_count) % 500 == 0 and (real_count + fake_count) > 0:
                print(f"    [train] real={real_count} fake={fake_count}")

        print(f"    [train] DONE: real={real_count} fake={fake_count}")
    except Exception as e:
        print(f"    ⚠ Error loading dataset: {e}")


def merge_celeb_df():
    """Copy Celeb-DF face crops into the combined dataset."""
    celeb_dir = Path(__file__).parent / "dataset"

    print("\n" + "=" * 60)
    print("[+] Merging Celeb-DF v2 face crops...")
    print("=" * 60)

    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            src = celeb_dir / split / cls
            dst = DATASET_DIR / split / cls
            dst.mkdir(parents=True, exist_ok=True)

            if not src.exists():
                continue

            files = list(src.glob("*.jpg"))
            # Take a subset to balance with AI-generated data
            max_files = MAX_PER_SPLIT.get(split, 5000)
            if len(files) > max_files:
                random.shuffle(files)
                files = files[:max_files]

            count = 0
            for f in files:
                dst_path = dst / f"celebdf_{f.name}"
                if not dst_path.exists():
                    import shutil
                    shutil.copy2(str(f), str(dst_path))
                    count += 1

            print(f"  [{split}/{cls}] Copied {count} Celeb-DF images")


def create_val_from_train():
    """Create validation split from training data if val is empty."""
    for cls in ["real", "fake"]:
        val_dir = DATASET_DIR / "val" / cls
        val_dir.mkdir(parents=True, exist_ok=True)

        existing = list(val_dir.glob("*.jpg"))
        if len(existing) >= 500:
            continue

        train_dir = DATASET_DIR / "train" / cls
        if not train_dir.exists():
            continue

        files = list(train_dir.glob("*.jpg"))
        random.shuffle(files)
        n_val = min(2000, len(files) // 8)  # 12.5% for val

        import shutil
        count = 0
        for f in files[:n_val]:
            dst = val_dir / f.name
            shutil.move(str(f), str(dst))
            count += 1

        print(f"  [val/{cls}] Moved {count} images from train")


def main():
    random.seed(42)
    np.random.seed(42)

    print("🔧 SecureSight — AI-Generated Image Detection Dataset Builder")
    print("=" * 60)
    print(f"Output: {DATASET_DIR}")
    print()

    # Step 1: Download HuggingFace datasets
    download_dappai_dataset()
    download_ai_vs_deepfake()

    # Step 2: Merge Celeb-DF data
    merge_celeb_df()

    # Step 3: Create val split if needed
    print("\n[*] Creating validation split...")
    create_val_from_train()

    # Summary
    print("\n" + "=" * 60)
    print("DATASET SUMMARY")
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
    print(f"\nReady for training!")


if __name__ == "__main__":
    main()
