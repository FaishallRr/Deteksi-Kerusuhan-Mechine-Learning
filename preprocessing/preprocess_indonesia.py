import sys
sys.path.insert(0, ".")

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import csv
import json
import yaml


class IndonesiaVideoProcessor:
    def __init__(
        self,
        metadata_path: str = "sample_videos/indonesia/metadata.csv",
        output_dir: str = "features/indonesia",
        target_size: Tuple[int, int] = (224, 224),
        temporal_window: int = 16,
        fps_sampling: int = 2,
        device: str = "cpu",
    ):
        self.metadata_path = Path(metadata_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_size = target_size
        self.temporal_window = temporal_window
        self.fps_sampling = fps_sampling
        self.device = device
        self._init_extractor()

    def _init_extractor(self):
        from preprocessing.feature_extractor import TemporalFeatureExtractor
        self.extractor = TemporalFeatureExtractor("s3d", self.device)

    def load_video_frames(self, video_path: Path) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        frames = []
        count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if count % self.fps_sampling == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.target_size)
                frames.append(frame)
            count += 1

        cap.release()
        if len(frames) == 0:
            return None
        return np.array(frames)

    def extract_features(self, frames: np.ndarray) -> np.ndarray:
        segments = []
        for i in range(0, len(frames) - self.temporal_window + 1, self.temporal_window):
            seg = frames[i:i + self.temporal_window]
            segments.append(seg)
        features = []
        for seg in tqdm(segments, desc="Extracting features", leave=False):
            feat = self.extractor.extract(list(seg))
            features.append(feat)
        if len(features) == 0:
            return np.array([])
        return np.concatenate(features, axis=0) if features[0].ndim > 1 else np.array(features)

    def run(self):
        with open(self.metadata_path, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            rows = list(reader)

        all_metadata = []
        for row in tqdm(rows, desc="Processing Indonesian videos"):
            video_path = Path(row["file_path"])
            if not video_path.exists():
                print(f"  SKIP (not found): {video_path}")
                continue

            video_id = row["video_id"]
            category = row["category"]
            label = int(row["label"])

            out_dir = self.output_dir / category
            out_dir.mkdir(parents=True, exist_ok=True)
            out_path = out_dir / f"{video_id}.npy"

            if out_path.exists():
                print(f"  SKIP (exists): {video_id}")
                try:
                    features = np.load(out_path)
                except Exception:
                    features = None
                if features is not None:
                    all_metadata.append({
                        "video_id": video_id,
                        "path": str(out_path),
                        "category": category,
                        "label": label,
                        "segments": len(features) if features.ndim > 1 else 1,
                        "feature_dim": features.shape[-1],
                    })
                    continue

            frames = self.load_video_frames(video_path)
            if frames is None or len(frames) < self.temporal_window:
                print(f"  SKIP (too short/no frames): {video_id}")
                continue

            features = self.extract_features(frames)
            if len(features) == 0:
                print(f"  SKIP (no features): {video_id}")
                continue

            np.save(out_path, features)
            all_metadata.append({
                "video_id": video_id,
                "path": str(out_path),
                "category": category,
                "label": label,
                "segments": len(features) if features.ndim > 1 else 1,
                "feature_dim": features.shape[-1],
                "total_frames": len(frames),
            })

        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(all_metadata, f, indent=2)

        print(f"\n{'='*50}")
        print(f"INDONESIA PREPROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"Total videos processed: {len(all_metadata)}")
        labels = [m["label"] for m in all_metadata]
        print(f"Normal: {sum(1 for l in labels if l == 0)} | Anomaly: {sum(1 for l in labels if l == 1)}")

        cats = set(m["category"] for m in all_metadata)
        for cat in sorted(cats):
            cat_meta = [m for m in all_metadata if m["category"] == cat]
            total_segs = sum(m["segments"] for m in cat_meta)
            print(f"  {cat}: {len(cat_meta)} videos, {total_segs} segments")

        print(f"Metadata: {meta_path}")
        return all_metadata


if __name__ == "__main__":
    import torch
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"Using device: {device}")
    processor = IndonesiaVideoProcessor(device=device)
    processor.run()
