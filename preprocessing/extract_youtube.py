"""Extract S3D features from YouTube-scraped demo_damai videos."""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
from pathlib import Path
from preprocessing.feature_extractor import TemporalFeatureExtractor

YT_DIR = Path("data/demo_damai/youtube_scraped")
OUT_DIR = Path("features/final_dataset/train")
SEGMENT_LEN = 16
STRIDE = 8
TARGET_FPS = 4

extractor = TemporalFeatureExtractor(architecture="s3d", device="cpu")


def sample_frames(video_path, max_segments=200):
    import cv2
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30
    step = max(1, int(fps / TARGET_FPS))
    frame_indices = list(range(0, total, step))
    if len(frame_indices) < SEGMENT_LEN:
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
    segments = []
    for start in range(0, len(frames) - SEGMENT_LEN + 1, STRIDE):
        seg = frames[start:start + SEGMENT_LEN]
        segments.append(seg)
        if len(segments) >= max_segments:
            break
    return segments


def main():
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    # Already extracted
    extracted = {f.stem.replace("demo_damai_", "") for f in OUT_DIR.glob("demo_damai_*.npy")}
    videos = sorted(YT_DIR.glob("*.mp4"))
    total, skipped = 0, 0

    for vpath in videos:
        vid = vpath.stem
        out_path = OUT_DIR / f"demo_damai_{vid}.npy"
        if vid in extracted or out_path.exists():
            skipped += 1
            continue
        try:
            segments = sample_frames(vpath)
            if segments is None:
                print(f"  Skip (short): {vid}")
                continue
            features = extractor.extract_batch(segments)
            np.save(str(out_path), features)
            total += 1
            print(f"  OK ({len(segments)} seg): {vid}")
        except Exception as e:
            print(f"  FAIL: {vid} - {e}")

    print(f"\nDone. Extracted: {total}, Skipped: {skipped}")


if __name__ == "__main__":
    main()
