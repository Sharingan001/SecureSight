"""SecureSight — Celeb-DF v2 Frame Extractor + Dataset Builder.

Extracts face frames from Celeb-DF v2 videos and builds
train/val/test splits for deepfake detector training.

Dataset: 590 real + 5639 fake celebrity videos + 300 YouTube real
"""

from __future__ import annotations

import os
import random
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

RAW_DIR = Path(__file__).parent / "dataset" / "raw"
DATASET_DIR = Path(__file__).parent / "dataset"
TARGET_SIZE = 380
FRAMES_PER_VIDEO = 15  # Extract N frames per video

# OpenCV face detector (always available)
_face_cascade = None
def get_face_detector():
    global _face_cascade
    if _face_cascade is None:
        _face_cascade = cv2.CascadeClassifier(
            cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
        )
    return _face_cascade


def extract_face_from_frame(frame: np.ndarray) -> np.ndarray | None:
    """Detect and crop the largest face from a frame."""
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.equalizeHist(gray)

    detector = get_face_detector()
    faces = detector.detectMultiScale(
        gray, scaleFactor=1.1, minNeighbors=5,
        minSize=(60, 60), flags=cv2.CASCADE_SCALE_IMAGE
    )

    if len(faces) == 0:
        return None

    # Pick largest face
    areas = [w * h for (x, y, w, h) in faces]
    idx = np.argmax(areas)
    x, y, w, h = faces[idx]

    # Add margin (25%)
    margin = int(max(w, h) * 0.25)
    fh, fw = frame.shape[:2]
    x1 = max(0, x - margin)
    y1 = max(0, y - margin)
    x2 = min(fw, x + w + margin)
    y2 = min(fh, y + h + margin)

    face = frame[y1:y2, x1:x2]
    if face.shape[0] < 60 or face.shape[1] < 60:
        return None

    face = cv2.resize(face, (TARGET_SIZE, TARGET_SIZE), interpolation=cv2.INTER_LANCZOS4)
    return face


def extract_frames_from_video(video_path: str, max_frames: int = FRAMES_PER_VIDEO) -> list[np.ndarray]:
    """Extract face crops from uniformly sampled frames of a video."""
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        return []

    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        cap.release()
        return []

    # Uniformly sample frame indices
    if total_frames <= max_frames:
        indices = list(range(total_frames))
    else:
        indices = np.linspace(0, total_frames - 1, max_frames, dtype=int).tolist()

    faces = []
    for frame_idx in indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
        ret, frame = cap.read()
        if not ret:
            continue

        face = extract_face_from_frame(frame)
        if face is not None:
            faces.append(face)

    cap.release()
    return faces


def process_video_folder(folder: Path, output_dir: Path, label: str, max_videos: int = 0):
    """Process all videos in a folder and save extracted face crops."""
    output_dir.mkdir(parents=True, exist_ok=True)

    videos = sorted([f for f in folder.iterdir() if f.suffix.lower() in {".mp4", ".avi", ".mov"}])
    if max_videos > 0:
        videos = videos[:max_videos]

    total_frames = 0
    total_videos = len(videos)

    for i, video_path in enumerate(videos):
        try:
            faces = extract_frames_from_video(str(video_path))
            for j, face in enumerate(faces):
                q = random.choice([90, 92, 95])
                out_path = output_dir / f"{label}_{video_path.stem}_f{j:03d}.jpg"
                cv2.imwrite(str(out_path), face, [cv2.IMWRITE_JPEG_QUALITY, q])
                total_frames += 1
        except Exception as e:
            print(f"  ⚠ Error on {video_path.name}: {e}")
            continue

        if (i + 1) % 50 == 0 or i == total_videos - 1:
            print(f"  [{label}] {i+1}/{total_videos} videos → {total_frames} faces extracted")

    return total_frames


def parse_test_list(list_path: Path) -> tuple[set[str], set[str]]:
    """Parse the official test split file. Returns (test_real_set, test_fake_set)."""
    test_real = set()
    test_fake = set()

    with open(list_path, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            parts = line.split()
            if len(parts) < 2:
                continue
            label = parts[0]
            path = parts[1]

            video_name = Path(path).stem
            folder = Path(path).parent.name

            if label == "1":  # real
                test_real.add(f"{folder}/{video_name}")
            else:  # fake
                test_fake.add(f"{folder}/{video_name}")

    return test_real, test_fake


def main():
    print("=" * 60)
    print("SecureSight — Celeb-DF v2 Dataset Builder")
    print("=" * 60)
    print()

    # Verify raw data exists
    celeb_real = RAW_DIR / "Celeb-real"
    celeb_fake = RAW_DIR / "Celeb-synthesis"
    youtube_real = RAW_DIR / "YouTube-real"
    test_list = RAW_DIR / "List_of_testing_videos.txt"

    for d in [celeb_real, celeb_fake, youtube_real]:
        if not d.exists():
            print(f"[✗] Missing: {d}")
            return
        print(f"[✓] Found: {d.name} ({len(list(d.glob('*.mp4')))} videos)")

    # Parse official test split
    test_real_set, test_fake_set = set(), set()
    if test_list.exists():
        test_real_set, test_fake_set = parse_test_list(test_list)
        print(f"[✓] Test split: {len(test_real_set)} real + {len(test_fake_set)} fake videos")

    # Create output dirs
    for split in ["train", "val", "test"]:
        for cls in ["real", "fake"]:
            (DATASET_DIR / split / cls).mkdir(parents=True, exist_ok=True)

    random.seed(42)
    np.random.seed(42)

    # ── Process REAL videos ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[1/3] Extracting faces from REAL videos...")
    print(f"{'─'*60}")

    # Get all real videos
    real_videos_celeb = sorted(celeb_real.glob("*.mp4"))
    real_videos_yt = sorted(youtube_real.glob("*.mp4"))

    # Split into test / train+val based on official list
    test_real_videos = []
    trainval_real_videos = []

    for v in real_videos_celeb:
        key = f"Celeb-real/{v.stem}"
        if key in test_real_set:
            test_real_videos.append(v)
        else:
            trainval_real_videos.append(v)

    for v in real_videos_yt:
        key = f"YouTube-real/{v.stem}"
        if key in test_real_set:
            test_real_videos.append(v)
        else:
            trainval_real_videos.append(v)

    # Split train+val → 85% train, 15% val
    random.shuffle(trainval_real_videos)
    n_val_real = max(1, int(len(trainval_real_videos) * 0.15))
    val_real_videos = trainval_real_videos[:n_val_real]
    train_real_videos = trainval_real_videos[n_val_real:]

    print(f"  Real split: train={len(train_real_videos)} val={len(val_real_videos)} test={len(test_real_videos)}")

    # Extract frames
    print("\n  Extracting TRAIN real faces...")
    train_real_count = 0
    for i, v in enumerate(train_real_videos):
        faces = extract_frames_from_video(str(v))
        for j, face in enumerate(faces):
            cv2.imwrite(str(DATASET_DIR / "train" / "real" / f"real_{v.stem}_f{j:03d}.jpg"),
                        face, [cv2.IMWRITE_JPEG_QUALITY, random.choice([90, 92, 95])])
            train_real_count += 1
        if (i + 1) % 100 == 0:
            print(f"    [{i+1}/{len(train_real_videos)}] → {train_real_count} faces")
    print(f"    Total train real: {train_real_count}")

    print("\n  Extracting VAL real faces...")
    val_real_count = 0
    for v in val_real_videos:
        faces = extract_frames_from_video(str(v))
        for j, face in enumerate(faces):
            cv2.imwrite(str(DATASET_DIR / "val" / "real" / f"real_{v.stem}_f{j:03d}.jpg"),
                        face, [cv2.IMWRITE_JPEG_QUALITY, random.choice([90, 92, 95])])
            val_real_count += 1
    print(f"    Total val real: {val_real_count}")

    print("\n  Extracting TEST real faces...")
    test_real_count = 0
    for v in test_real_videos:
        faces = extract_frames_from_video(str(v))
        for j, face in enumerate(faces):
            cv2.imwrite(str(DATASET_DIR / "test" / "real" / f"real_{v.stem}_f{j:03d}.jpg"),
                        face, [cv2.IMWRITE_JPEG_QUALITY, random.choice([90, 92, 95])])
            test_real_count += 1
    print(f"    Total test real: {test_real_count}")

    # ── Process FAKE videos ──────────────────────────────────────────
    print(f"\n{'─'*60}")
    print("[2/3] Extracting faces from DEEPFAKE videos...")
    print(f"{'─'*60}")

    fake_videos = sorted(celeb_fake.glob("*.mp4"))

    # Split based on official test list
    test_fake_videos = []
    trainval_fake_videos = []

    for v in fake_videos:
        key = f"Celeb-synthesis/{v.stem}"
        if key in test_fake_set:
            test_fake_videos.append(v)
        else:
            trainval_fake_videos.append(v)

    random.shuffle(trainval_fake_videos)
    n_val_fake = max(1, int(len(trainval_fake_videos) * 0.15))
    val_fake_videos = trainval_fake_videos[:n_val_fake]
    train_fake_videos = trainval_fake_videos[n_val_fake:]

    print(f"  Fake split: train={len(train_fake_videos)} val={len(val_fake_videos)} test={len(test_fake_videos)}")

    print("\n  Extracting TRAIN fake faces...")
    train_fake_count = 0
    for i, v in enumerate(train_fake_videos):
        faces = extract_frames_from_video(str(v))
        for j, face in enumerate(faces):
            cv2.imwrite(str(DATASET_DIR / "train" / "fake" / f"fake_{v.stem}_f{j:03d}.jpg"),
                        face, [cv2.IMWRITE_JPEG_QUALITY, random.choice([85, 88, 90, 92])])
            train_fake_count += 1
        if (i + 1) % 200 == 0:
            print(f"    [{i+1}/{len(train_fake_videos)}] → {train_fake_count} faces")
    print(f"    Total train fake: {train_fake_count}")

    print("\n  Extracting VAL fake faces...")
    val_fake_count = 0
    for v in val_fake_videos:
        faces = extract_frames_from_video(str(v))
        for j, face in enumerate(faces):
            cv2.imwrite(str(DATASET_DIR / "val" / "fake" / f"fake_{v.stem}_f{j:03d}.jpg"),
                        face, [cv2.IMWRITE_JPEG_QUALITY, random.choice([85, 88, 90, 92])])
            val_fake_count += 1
    print(f"    Total val fake: {val_fake_count}")

    print("\n  Extracting TEST fake faces...")
    test_fake_count = 0
    for v in test_fake_videos:
        faces = extract_frames_from_video(str(v))
        for j, face in enumerate(faces):
            cv2.imwrite(str(DATASET_DIR / "test" / "fake" / f"fake_{v.stem}_f{j:03d}.jpg"),
                        face, [cv2.IMWRITE_JPEG_QUALITY, random.choice([85, 88, 90, 92])])
            test_fake_count += 1
    print(f"    Total test fake: {test_fake_count}")

    # ── Summary ──────────────────────────────────────────────────────
    print(f"\n{'='*60}")
    print("Dataset Summary (Celeb-DF v2):")
    print(f"{'='*60}")
    for split in ["train", "val", "test"]:
        r = len(list((DATASET_DIR / split / "real").glob("*.jpg")))
        f = len(list((DATASET_DIR / split / "fake").glob("*.jpg")))
        print(f"  {split:6s}: {r:6d} real | {f:6d} fake | {r+f:6d} total")

    total_r = sum(len(list((DATASET_DIR / s / "real").glob("*.jpg"))) for s in ["train","val","test"])
    total_f = sum(len(list((DATASET_DIR / s / "fake").glob("*.jpg"))) for s in ["train","val","test"])
    print(f"  {'TOTAL':6s}: {total_r:6d} real | {total_f:6d} fake | {total_r+total_f:6d} total")
    print(f"\nDataset location: {DATASET_DIR}")
    print("Ready for training! Run: python training/train.py")


if __name__ == "__main__":
    main()
