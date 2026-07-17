"""
Extract S3D features from demo videos - matching training pipeline exactly.
"""
import sys, os, cv2, time, numpy as np, json
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))
from preprocessing.feature_extractor import TemporalFeatureExtractor

VIDEO_DIR = Path("test_videos")
OUTPUT_DIR = Path("features/demo_videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_FPS = 4
SEGMENT_LEN = 16
STRIDE = 8
MAX_SEGMENTS = 200

RUSUH_VIDEOS = [
    ("fight_sample_1.mp4", "Rusuh - tawuran (CCTV)"),
    ("fight_sample_2.mp4", "Rusuh - tawuran (CCTV #2)"),
    ("u3EyoqRiEyI.mp4", "Rusuh - tawuran YouTube #2"),
    ("yt_tawuran_hQ98fJ-KITQ.mp4", "Rusuh - tawuran YouTube"),
]
NORMAL_VIDEOS = []
DEMO_VIDEOS = [(f, 1, d) for f, d in RUSUH_VIDEOS] + [(f, 0, d) for f, d in NORMAL_VIDEOS]


def sample_frames(video_path):
    """Match the training pipeline exactly."""
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
        if len(segments) >= MAX_SEGMENTS:
            break

    return segments


def main():
    extractor = TemporalFeatureExtractor(architecture="s3d", device="cpu")
    metadata = []

    for fname, label, display_name in DEMO_VIDEOS:
        vpath = VIDEO_DIR / fname
        if not vpath.exists():
            print("  SKIP (not found): %s" % fname)
            continue

        print("Processing: %s (label=%d)" % (fname, label))
        t0 = time.time()

        segments = sample_frames(str(vpath))
        if segments is None or len(segments) == 0:
            print("  -> FAILED: not enough frames")
            continue

        all_feats = []
        for seg in segments:
            feats = extractor.extract(np.array(seg))
            all_feats.append(feats.squeeze())

        feat_arr = np.array(all_feats)
        out_path = OUTPUT_DIR / ("%s.npy" % Path(fname).stem)
        np.save(out_path, feat_arr)

        meta = {
            "path": str(out_path),
            "label": label,
            "label_name": "rusuh" if label else "normal",
            "video_file": fname,
            "display_name": display_name,
            "segments": len(segments),
            "extraction_time": round(time.time() - t0, 1),
        }
        metadata.append(meta)
        print("  -> %d segments, %.1fs, saved to %s" % (len(segments), meta["extraction_time"], out_path.name))

    with open(OUTPUT_DIR / "demo_metadata.json", "w") as f:
        json.dump(metadata, f, indent=2)
    print("\n[OK] %d videos processed" % len(metadata))


if __name__ == "__main__":
    main()
