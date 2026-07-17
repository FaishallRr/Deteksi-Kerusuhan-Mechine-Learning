"""
Extract S3D features for all videos in data/splits/.

For each video:
  - Sample frames at ~5 FPS
  - Group into 16-frame segments (stride 8)
  - Extract 1024-d feature per segment via S3D
  - Save as .npy in features/final_dataset/

Usage:
    python preprocessing/extract_all_features.py

Output:
    features/final_dataset/
    ├── train/
    │   ├── demo_rusuh_videoname.npy
    │   ├── demo_damai_videoname.npy
    │   └── normal_videoname.npy
    ├── val/
    ├── test/
    └── metadata.json
"""
import sys, os, json, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import numpy as np
from pathlib import Path
from tqdm import tqdm

from preprocessing.feature_extractor import TemporalFeatureExtractor
from utils.config_loader import load_config

config = load_config("config.yaml")
device = config["general"]["device"]

DATA_DIR = Path("data")
SPLITS_DIR = DATA_DIR / "splits"
FEAT_DIR = Path("features") / "final_dataset"

STRIDE = 8
SEGMENT_LEN = 16
TARGET_FPS = 4  # sample frames at ~4 FPS

extractor = TemporalFeatureExtractor(architecture="s3d", device=device)


def sample_frames(video_path, max_segments=200):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30

    # Sample at TARGET_FPS
    step = max(1, int(fps / TARGET_FPS))
    frame_indices = list(range(0, total, step))
    if len(frame_indices) < SEGMENT_LEN:
        # Pad by duplicating if video too short
        frame_indices = frame_indices * ((SEGMENT_LEN // len(frame_indices)) + 1)
        frame_indices = frame_indices[:SEGMENT_LEN]

    frames = []
    for idx in frame_indices:
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()

    if len(frames) < SEGMENT_LEN:
        return None

    # Build segments: sliding window with STRIDE
    segments = []
    for start in range(0, len(frames) - SEGMENT_LEN + 1, STRIDE):
        seg = frames[start:start + SEGMENT_LEN]
        segments.append(seg)
        if len(segments) >= max_segments:
            break

    return segments


def extract_video(video_path, out_path):
    if out_path.exists():
        return  # already extracted

    segments = sample_frames(video_path)
    if segments is None:
        print(f"  Skipping {video_path.name}: too short")
        return

    # Batch extract
    features = extractor.extract_batch(segments)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(out_path), features)


def main():
    print(f"Device: {device}")
    print(f"Feature dir: {FEAT_DIR}")

    metadata = []
    total_skipped = 0

    for split_name in ("train", "val", "test"):
        csv_path = SPLITS_DIR / f"{split_name}.csv"
        if not csv_path.exists():
            print(f"  Skipping {split_name}: no CSV found")
            continue

        # Read CSV
        import csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        print(f"\n{split_name}: {len(rows)} videos")

        for row in tqdm(rows, desc=f"Extracting {split_name}"):
            rel_path = row["path"]
            label = row["label"]
            video_path = DATA_DIR / rel_path
            if not video_path.exists():
                total_skipped += 1
                continue

            # Output: features/final_dataset/{split}/{label}_{filename}.npy
            stem = f"{label}_{video_path.stem}"
            out_path = FEAT_DIR / split_name / f"{stem}.npy"

            extract_video(video_path, out_path)

            if out_path.exists():
                feat = np.load(str(out_path))
                metadata.append({
                    "path": str(out_path),
                    "label": 1 if label == "demo_rusuh" else 0,
                    "label_name": label,
                    "split": split_name,
                    "source": row.get("source", ""),
                    "segments": feat.shape[0],
                })

    # Save metadata
    meta_path = FEAT_DIR / "metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)

    print(f"\n{'='*50}")
    print(f"Done!")
    print(f"  Total videos processed: {len(metadata)}")
    print(f"  Skipped (missing): {total_skipped}")
    if metadata:
        rusuh = sum(1 for m in metadata if m["label"] == 1)
        normal = sum(1 for m in metadata if m["label"] == 0)
        print(f"  demo_rusuh (anomaly): {rusuh} | demo_damai+normal: {normal}")
        print(f"  Total segments: {sum(m['segments'] for m in metadata)}")
    print(f"  Metadata: {meta_path}")


if __name__ == "__main__":
    main()
