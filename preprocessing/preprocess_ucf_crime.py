import sys
sys.path.insert(0, ".")

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import json
import re


class UCFCrimeProcessor:
    def __init__(
        self,
        source_dir: str = "ucf_crime_raw",
        output_dir: str = "features/ucf_crime",
        target_size: Tuple[int, int] = (224, 224),
        temporal_window: int = 16,
        device: str = "cpu",
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_size = target_size
        self.temporal_window = temporal_window
        self.device = device
        self._init_extractor()

    def _init_extractor(self):
        from preprocessing.feature_extractor import TemporalFeatureExtractor
        self.extractor = TemporalFeatureExtractor("s3d", self.device)

    def get_video_ids(self, class_dir: Path) -> List[str]:
        ids = set()
        for f in class_dir.iterdir():
            m = re.match(r"(.+)_\d+\.png", f.name)
            if m:
                ids.add(m.group(1))
        return sorted(ids)

    def load_frames(self, class_dir: Path, video_id: str) -> Optional[np.ndarray]:
        frames = sorted(class_dir.glob(f"{video_id}_*.png"),
                        key=lambda f: int(re.search(r'_(\d+)\.png', f.name).group(1)))
        if len(frames) < self.temporal_window:
            return None
        images = []
        for f in frames:
            img = cv2.imread(str(f))
            if img is None:
                continue
            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            img = cv2.resize(img, self.target_size)
            images.append(img)
        return np.array(images)

    def extract_features(self, frames: np.ndarray) -> np.ndarray:
        segments = []
        for i in range(0, len(frames) - self.temporal_window + 1, self.temporal_window):
            seg = frames[i:i + self.temporal_window]
            segments.append(seg)
        features = []
        for seg in segments:
            feat = self.extractor.extract(list(seg))
            features.append(feat)
        return np.array(features)

    def process_class(self, class_dir: Path, label: int, split: str, max_videos: int = None):
        class_name = class_dir.name
        out_dir = self.output_dir / split / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        video_ids = self.get_video_ids(class_dir)
        if max_videos:
            import random
            random.seed(42)
            video_ids = random.sample(video_ids, min(max_videos, len(video_ids)))

        metadata = []
        for vid_id in tqdm(video_ids, desc=f"{split}/{class_name} ({len(video_ids)} videos)"):
            frames = self.load_frames(class_dir, vid_id)
            if frames is None:
                continue

            features = self.extract_features(frames)
            if len(features) == 0:
                continue

            out_path = out_dir / f"{vid_id}.npy"
            np.save(out_path, features)

            metadata.append({
                "video_id": vid_id,
                "path": str(out_path),
                "split": split,
                "category": class_name,
                "label": label,
                "segments": len(features),
                "feature_dim": features.shape[1],
                "total_frames": len(frames),
            })

        return metadata

    def run(
        self,
        anomaly_classes: List[str] = None,
        normal_classes: List[str] = None,
        videos_per_anomaly: int = 15,
        normal_count: int = 25,
    ):
        if anomaly_classes is None:
            anomaly_classes = ["Fighting", "Assault", "Robbery", "Shooting", "Abuse", "Arson"]
        if normal_classes is None:
            normal_classes = ["NormalVideos"]

        all_metadata = []

        for split in ["Train", "Test"]:
            split_source = self.source_dir / split
            if not split_source.exists():
                continue

            for cls in anomaly_classes:
                cls_dir = split_source / cls
                if not cls_dir.exists():
                    continue
                n = max(3, videos_per_anomaly // 2) if split == "Test" else videos_per_anomaly
                meta = self.process_class(cls_dir, label=1, split=split, max_videos=n)
                all_metadata.extend(meta)

            for cls in normal_classes:
                cls_dir = split_source / cls
                if not cls_dir.exists():
                    continue
                n = max(5, normal_count // 2) if split == "Test" else normal_count
                meta = self.process_class(cls_dir, label=0, split=split, max_videos=n)
                all_metadata.extend(meta)

        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(all_metadata, f, indent=2)

        print(f"\n{'='*50}")
        print(f"PREPROCESSING COMPLETE")
        print(f"{'='*50}")
        print(f"Total videos processed: {len(all_metadata)}")
        labels = [m["label"] for m in all_metadata]
        print(f"Normal: {sum(1 for l in labels if l == 0)} | Anomaly: {sum(1 for l in labels if l == 1)}")
        print(f"Metadata: {meta_path}")

        for split in ["Train", "Test"]:
            split_meta = [m for m in all_metadata if m["split"] == split]
            print(f"\n{split}:")
            cats = set(m["category"] for m in split_meta)
            for cat in sorted(cats):
                cat_meta = [m for m in split_meta if m["category"] == cat]
                total_segs = sum(m["segments"] for m in cat_meta)
                total_frames = sum(m["total_frames"] for m in cat_meta)
                print(f"  {cat}: {len(cat_meta)} videos, {total_segs} segments, {total_frames} frames")

        return all_metadata


if __name__ == "__main__":
    processor = UCFCrimeProcessor(device="cpu")
    processor.run(videos_per_anomaly=15, normal_count=25)
