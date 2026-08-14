"""SecureSight — Dataset downloader + synthetic fake generator.

Downloads LFW (Labeled Faces in the Wild) as real face images,
then generates high-quality synthetic fakes using forensic-grade
manipulation techniques for training deepfake detectors.

Manipulation techniques:
  1. Face blending (alpha composite with warped donor)
  2. Frequency-band noise injection (GAN-like artifacts)
  3. JPEG double-compression artifacts
  4. Color channel shifting (deepfake color bleed)
  5. Gaussian blur + sharpen inconsistency
  6. Face region warping (landmark distortion)
  7. Splicing (region swap between images)
"""

from __future__ import annotations

import os
import random
import shutil
import tarfile
import urllib.request
from pathlib import Path

import cv2
import numpy as np

# ── Configuration ─────────────────────────────────────────────────────

LFW_URL = "http://vis-www.cs.umass.edu/lfw/lfw-funneled.tgz"
DATASET_DIR = Path(__file__).parent / "dataset"
RAW_DIR = DATASET_DIR / "raw_lfw"
TRAIN_DIR = DATASET_DIR / "train"
VAL_DIR = DATASET_DIR / "val"
TEST_DIR = DATASET_DIR / "test"

TARGET_SIZE = 380
MIN_FACE_SIZE = 80

# How many fake variants to generate per real image
FAKES_PER_REAL = 3

# ── Download ──────────────────────────────────────────────────────────

def download_lfw():
    """Download and extract LFW dataset."""
    archive_path = DATASET_DIR / "lfw-funneled.tgz"
    DATASET_DIR.mkdir(parents=True, exist_ok=True)

    if RAW_DIR.exists() and any(RAW_DIR.iterdir()):
        print(f"[✓] LFW already exists at {RAW_DIR}")
        return

    if not archive_path.exists():
        print(f"[↓] Downloading LFW dataset ({LFW_URL})...")
        urllib.request.urlretrieve(LFW_URL, str(archive_path), _progress_hook)
        print()

    print("[⚙] Extracting archive...")
    with tarfile.open(str(archive_path), "r:gz") as tar:
        tar.extractall(str(DATASET_DIR))

    # lfw_funneled/ → raw_lfw/
    extracted = DATASET_DIR / "lfw_funneled"
    if extracted.exists():
        extracted.rename(RAW_DIR)

    # Cleanup
    archive_path.unlink(missing_ok=True)
    print(f"[✓] Extracted to {RAW_DIR}")


def _progress_hook(block_num, block_size, total_size):
    downloaded = block_num * block_size
    pct = min(downloaded / total_size * 100, 100) if total_size > 0 else 0
    mb = downloaded / 1024 / 1024
    total_mb = total_size / 1024 / 1024
    print(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)", end="", flush=True)


# ── Collect real face images ──────────────────────────────────────────

def collect_real_faces() -> list[str]:
    """Scan LFW directory and collect all face image paths."""
    faces = []
    for person_dir in sorted(RAW_DIR.iterdir()):
        if not person_dir.is_dir():
            continue
        for img_path in sorted(person_dir.glob("*.jpg")):
            faces.append(str(img_path))
    random.shuffle(faces)
    print(f"[✓] Found {len(faces)} real face images")
    return faces


# ── Manipulation techniques ───────────────────────────────────────────

def manipulate_blend(img: np.ndarray, donor_pool: list[str]) -> np.ndarray:
    """Blend face region with a warped donor face (simulates face swap)."""
    donor_path = random.choice(donor_pool)
    donor = cv2.imread(donor_path)
    if donor is None:
        return manipulate_color_shift(img)

    donor = cv2.resize(donor, (img.shape[1], img.shape[0]))

    # Create elliptical mask centered on face
    h, w = img.shape[:2]
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 11)

    # Random alpha for blend strength
    alpha = random.uniform(0.3, 0.7)
    mask = mask * alpha

    mask3 = np.stack([mask] * 3, axis=-1)
    result = (img.astype(np.float32) * (1 - mask3) + donor.astype(np.float32) * mask3)
    return np.clip(result, 0, 255).astype(np.uint8)


def manipulate_frequency_noise(img: np.ndarray) -> np.ndarray:
    """Inject frequency-band noise simulating GAN generation artifacts."""
    h, w = img.shape[:2]
    result = img.astype(np.float32)

    for c in range(3):
        # FFT
        f = np.fft.fft2(result[:, :, c])
        fshift = np.fft.fftshift(f)

        # Create band-pass noise (suppress high-freq like GANs do)
        rows, cols = h, w
        crow, ccol = rows // 2, cols // 2
        radius = random.randint(30, 80)

        # Suppress high frequencies
        mask = np.ones((rows, cols), dtype=np.float32)
        y, x = np.ogrid[-crow:rows - crow, -ccol:cols - ccol]
        outer = x * x + y * y > radius * radius
        suppress = random.uniform(0.3, 0.7)
        mask[outer] = suppress

        # Add periodic noise at specific frequency
        noise_freq = random.randint(10, 40)
        noise_strength = random.uniform(0.5, 2.0)
        fshift[crow + noise_freq, :] += noise_strength * np.abs(fshift[crow + noise_freq, :])
        fshift[crow - noise_freq, :] += noise_strength * np.abs(fshift[crow - noise_freq, :])

        fshift *= mask
        f_ishift = np.fft.ifftshift(fshift)
        result[:, :, c] = np.abs(np.fft.ifft2(f_ishift))

    return np.clip(result, 0, 255).astype(np.uint8)


def manipulate_double_jpeg(img: np.ndarray) -> np.ndarray:
    """Double JPEG compression at different quality levels."""
    q1 = random.randint(40, 70)
    q2 = random.randint(75, 95)

    # First compression
    _, buf1 = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, q1])
    img1 = cv2.imdecode(buf1, cv2.IMREAD_COLOR)

    # Second compression at different quality
    _, buf2 = cv2.imencode(".jpg", img1, [cv2.IMWRITE_JPEG_QUALITY, q2])
    result = cv2.imdecode(buf2, cv2.IMREAD_COLOR)

    return result


def manipulate_color_shift(img: np.ndarray) -> np.ndarray:
    """Shift color channels independently (deepfake color bleed artifact)."""
    result = img.astype(np.float32)

    # Random per-channel shift
    for c in range(3):
        shift_x = random.randint(-3, 3)
        shift_y = random.randint(-3, 3)
        M = np.float32([[1, 0, shift_x], [0, 1, shift_y]])
        result[:, :, c] = cv2.warpAffine(result[:, :, c], M,
                                          (img.shape[1], img.shape[0]),
                                          borderMode=cv2.BORDER_REFLECT)

    # Slight color temperature shift
    result[:, :, 0] *= random.uniform(0.92, 1.08)  # B
    result[:, :, 2] *= random.uniform(0.92, 1.08)  # R

    return np.clip(result, 0, 255).astype(np.uint8)


def manipulate_warp(img: np.ndarray) -> np.ndarray:
    """Subtle face region warping simulating landmark reconstruction errors."""
    h, w = img.shape[:2]

    # Random control points for thin-plate spline-like warp
    src_pts = np.float32([
        [w * 0.25, h * 0.25],
        [w * 0.75, h * 0.25],
        [w * 0.25, h * 0.75],
        [w * 0.75, h * 0.75],
    ])

    # Add random perturbation
    jitter = random.uniform(3, 12)
    dst_pts = src_pts + np.random.uniform(-jitter, jitter, src_pts.shape).astype(np.float32)

    M = cv2.getPerspectiveTransform(src_pts, dst_pts)
    warped = cv2.warpPerspective(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)

    # Blend warped face region only
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (21, 21), 7)
    mask3 = np.stack([mask] * 3, axis=-1)

    result = img.astype(np.float32) * (1 - mask3) + warped.astype(np.float32) * mask3
    return np.clip(result, 0, 255).astype(np.uint8)


def manipulate_blur_sharpen(img: np.ndarray) -> np.ndarray:
    """Inconsistent blur + sharpen (face region sharpened, background blurred)."""
    h, w = img.shape[:2]

    # Blur entire image
    blurred = cv2.GaussianBlur(img, (5, 5), 2)

    # Sharpen face region
    kernel = np.array([[-1, -1, -1], [-1, 9 + random.uniform(0, 2), -1], [-1, -1, -1]])
    sharpened = cv2.filter2D(img, -1, kernel)

    # Face mask
    mask = np.zeros((h, w), dtype=np.float32)
    cv2.ellipse(mask, (w // 2, h // 2), (w // 3, h // 3), 0, 0, 360, 1.0, -1)
    mask = cv2.GaussianBlur(mask, (31, 31), 11)
    mask3 = np.stack([mask] * 3, axis=-1)

    result = blurred.astype(np.float32) * (1 - mask3) + sharpened.astype(np.float32) * mask3
    return np.clip(result, 0, 255).astype(np.uint8)


def manipulate_splice(img: np.ndarray, donor_pool: list[str]) -> np.ndarray:
    """Region splicing from another image."""
    donor_path = random.choice(donor_pool)
    donor = cv2.imread(donor_path)
    if donor is None:
        return manipulate_warp(img)

    donor = cv2.resize(donor, (img.shape[1], img.shape[0]))
    h, w = img.shape[:2]

    # Random rectangular region
    rx = random.randint(w // 6, w // 2)
    ry = random.randint(h // 6, h // 2)
    rw = random.randint(w // 6, w // 3)
    rh = random.randint(h // 6, h // 3)

    # Feathered mask
    mask = np.zeros((h, w), dtype=np.float32)
    mask[ry:ry + rh, rx:rx + rw] = 1.0
    mask = cv2.GaussianBlur(mask, (21, 21), 7)
    mask3 = np.stack([mask] * 3, axis=-1)

    result = img.astype(np.float32) * (1 - mask3) + donor.astype(np.float32) * mask3
    return np.clip(result, 0, 255).astype(np.uint8)


# All manipulation functions
MANIPULATIONS = [
    manipulate_blend,
    manipulate_frequency_noise,
    manipulate_double_jpeg,
    manipulate_color_shift,
    manipulate_warp,
    manipulate_blur_sharpen,
    manipulate_splice,
]


def generate_fake(img: np.ndarray, donor_pool: list[str]) -> np.ndarray:
    """Apply 1-3 random manipulations to create a fake."""
    result = img.copy()
    n_ops = random.randint(1, 3)
    ops = random.sample(MANIPULATIONS, min(n_ops, len(MANIPULATIONS)))

    for op in ops:
        if op in (manipulate_blend, manipulate_splice):
            result = op(result, donor_pool)
        else:
            result = op(result)

    return result


# ── Build dataset ─────────────────────────────────────────────────────

def build_dataset(real_paths: list[str], output_dir: Path, donor_pool: list[str],
                  max_real: int = 0):
    """Build train/val/test split with real and generated fakes."""
    real_dir = output_dir / "real"
    fake_dir = output_dir / "fake"
    real_dir.mkdir(parents=True, exist_ok=True)
    fake_dir.mkdir(parents=True, exist_ok=True)

    paths = real_paths[:max_real] if max_real > 0 else real_paths
    total = len(paths)
    print(f"  Processing {total} real images → {output_dir.name}/")

    real_count = 0
    fake_count = 0

    for i, path in enumerate(paths):
        img = cv2.imread(path)
        if img is None:
            continue

        # Crop and resize to face region
        h, w = img.shape[:2]
        if h < MIN_FACE_SIZE or w < MIN_FACE_SIZE:
            continue

        # Center crop to square
        side = min(h, w)
        y0 = (h - side) // 2
        x0 = (w - side) // 2
        img = img[y0:y0 + side, x0:x0 + side]
        img = cv2.resize(img, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)

        # Save real
        real_path = real_dir / f"real_{i:06d}.jpg"
        cv2.imwrite(str(real_path), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        real_count += 1

        # Generate fakes
        for j in range(FAKES_PER_REAL):
            try:
                fake = generate_fake(img, donor_pool)
                fake_path = fake_dir / f"fake_{i:06d}_{j}.jpg"
                # Save at slightly different quality to add variation
                q = random.choice([85, 90, 92, 95])
                cv2.imwrite(str(fake_path), fake, [cv2.IMWRITE_JPEG_QUALITY, q])
                fake_count += 1
            except Exception:
                continue

        if (i + 1) % 200 == 0 or i == total - 1:
            print(f"    [{i + 1}/{total}] real={real_count} fake={fake_count}")

    print(f"  [✓] {output_dir.name}: {real_count} real, {fake_count} fake")


def main():
    """Download LFW and build training dataset."""
    print("=" * 60)
    print("SecureSight — Dataset Builder")
    print("=" * 60)

    # Step 1: Download LFW
    download_lfw()

    # Step 2: Collect real faces
    all_faces = collect_real_faces()
    if not all_faces:
        print("[✗] No images found!")
        return

    # Step 3: Split into train/val/test (70/15/15)
    random.seed(42)
    random.shuffle(all_faces)

    n = len(all_faces)
    n_train = int(n * 0.70)
    n_val = int(n * 0.15)

    train_faces = all_faces[:n_train]
    val_faces = all_faces[n_train:n_train + n_val]
    test_faces = all_faces[n_train + n_val:]

    print(f"\n[📊] Split: train={len(train_faces)} val={len(val_faces)} test={len(test_faces)}")

    # Step 4: Build datasets
    print("\n[🔨] Building training set...")
    build_dataset(train_faces, TRAIN_DIR, all_faces)

    print("\n[🔨] Building validation set...")
    build_dataset(val_faces, VAL_DIR, all_faces)

    print("\n[🔨] Building test set...")
    build_dataset(test_faces, TEST_DIR, all_faces)

    # Summary
    for split_dir in [TRAIN_DIR, VAL_DIR, TEST_DIR]:
        real_c = len(list((split_dir / "real").glob("*"))) if (split_dir / "real").exists() else 0
        fake_c = len(list((split_dir / "fake").glob("*"))) if (split_dir / "fake").exists() else 0
        print(f"  {split_dir.name:6s}: {real_c:6d} real | {fake_c:6d} fake | {real_c + fake_c:6d} total")

    print("\n[✓] Dataset ready for training!")
    print(f"    Train dir: {TRAIN_DIR}")
    print(f"    Val dir:   {VAL_DIR}")
    print(f"    Test dir:  {TEST_DIR}")


if __name__ == "__main__":
    main()
