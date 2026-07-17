"""
Extract S3D features from demo videos for Streamlit app.
Saves features + metadata so the app can show video + prediction.
"""
import sys, os, json, cv2, time, numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from preprocessing.feature_extractor import TemporalFeatureExtractor

VIDEO_DIR = Path("test_videos")
OUTPUT_DIR = Path("features/demo_videos")
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

# Only use H.264 videos (verified browser-playable)
# Test results: fight_*.mp4, tawuran_grogol.mp4, yt_*.mp4 work
# normal_*.mp4, *_detected.mp4, anomaly_*.mp4 use FMP4 codec (not playable in browser)
RUSUH_VIDEOS = [
    ("fight_sample_1.mp4", "🔴 Rusuh - tawuran (CCTV)"),
    ("fight_sample_2.mp4", "🔴 Rusuh - tawuran (CCTV #2)"),
    ("tawuran_grogol.mp4", "🔴 Rusuh - tawuran Grogol"),
    ("yt_tawuran_hQ98fJ-KITQ.mp4", "🔴 Rusuh - tawuran YouTube"),
]
NORMAL_VIDEOS = [
    # Normal H.264 videos - none available, use feature demo instead
]
DEMO_VIDEOS = [(f, 1, d) for f, d in RUSUH_VIDEOS] + [(f, 0, d) for f, d in NORMAL_VIDEOS]

def main():
    print(f"Found {len(DEMO_VIDEOS)} videos to process")
    extractor = TemporalFeatureExtractor(architecture="s3d", device="cpu")

    metadata = []
    for fname, label, display_name in DEMO_VIDEOS:
        vpath = VIDEO_DIR / fname
        if not vpath.exists():
            print(f"  SKIP (not found): {fname}")
            continue

        print(f"\n  Processing: {fname} (label={label})")
        t0 = time.time()

        cap = cv2.VideoCapture(str(vpath))
        frames = []
        all_segments = []
        segment_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame_rgb)

            if len(frames) >= 16:
                feats = extractor.extract(np.array(frames[-16:]))
                seg_feat = feats.squeeze()
                all_segments.append(seg_feat.cpu().numpy() if hasattr(seg_feat, 'cpu') else seg_feat)
                segment_count += 1
                frames = frames[-8:]  # 8-frame overlap

                if segment_count >= 16:
                    break

        cap.release()
        elapsed = time.time() - t0

        if segment_count > 0:
            feat_arr = np.array(all_segments)
            out_path = OUTPUT_DIR / f"{Path(fname).stem}.npy"
            np.save(out_path, feat_arr)

            meta = {
                "path": str(out_path),
                "label": label,
                "label_name": "rusuh" if label else "normal",
                "video_file": fname,
                "display_name": display_name,
                "segments": segment_count,
                "extraction_time": round(elapsed, 1),
            }
            metadata.append(meta)
            print(f"    -> {segment_count} segments, {elapsed:.1f}s, saved to {out_path.name}")
        else:
            print(f"    -> FAILED: no segments extracted")

    # Save metadata
    meta_path = OUTPUT_DIR / "demo_metadata.json"
    with open(meta_path, "w") as f:
        json.dump(metadata, f, indent=2)
    print(f"\n[OK] {len(metadata)} videos processed. Metadata saved to {meta_path}")


if __name__ == "__main__":
    main()
