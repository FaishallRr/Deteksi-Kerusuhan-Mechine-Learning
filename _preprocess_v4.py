import sys; sys.path.insert(0, ".")
from preprocessing.feature_extractor import TemporalFeatureExtractor
import cv2, torch, numpy as np, json, os, glob
from pathlib import Path

device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Using device: {device}")

extractor = TemporalFeatureExtractor(architecture="s3d", device=device)

videos = sorted(glob.glob("sample_videos/indonesia_v4/*.mp4"))
print(f"Found {len(videos)} videos to process")

meta_path = "features/indonesia_v4/metadata.json"
os.makedirs("features/indonesia_v4", exist_ok=True)

existing = {}
if os.path.exists(meta_path):
    with open(meta_path) as f:
        existing = json.load(f)

for v in videos:
    basename = Path(v).name
    if basename in existing:
        print(f"  SKIP {basename} (already processed)")
        continue

    print(f"  Processing {basename}...", end=" ")

    cap = cv2.VideoCapture(v)
    frames = []
    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frames.append(frame)
    cap.release()

    if len(frames) < 16:
        print(f"TOO SHORT ({len(frames)} frames)")
        continue

    all_features = []
    for i in range(0, len(frames) - 15, 16):
        segment = frames[i : i + 16]
        feat = extractor.extract(segment)
        feat_np = feat.cpu().numpy() if torch.is_tensor(feat) else feat
        if feat_np.ndim > 2:
            feat_np = feat_np.reshape(feat_np.shape[0], -1)
        all_features.append(feat_np)

    if all_features:
        features = np.stack(all_features)
        feat_file = f"features/indonesia_v4/{basename}".replace(".mp4", "") + ".npy"
        np.save(feat_file, features)

        existing[basename] = {
            "feature_file": feat_file,
            "num_segments": features.shape[0],
            "feature_dim": features.shape[1],
            "label": 1,
            "source": "indonesia_v4",
        }
        print(f"{len(all_features)} segments, shape {features.shape}")
    else:
        print("NO valid segments")

with open(meta_path, "w") as f:
    json.dump(existing, f, indent=2)

total_segments = sum(v["num_segments"] for v in existing.values())
print(f"\nTotal: {len(existing)} videos, {total_segments} segments")
