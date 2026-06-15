import sys
sys.path.insert(0, ".")

import cv2
import torch
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Dict
from tqdm import tqdm
import json
from PIL import Image
from torchvision.transforms import functional as F


class IndonesiaV3Processor:
    def __init__(
        self,
        source_dir: str = "sample_videos/indonesia_v3",
        output_dir: str = "features/indonesia_v3",
        target_size: Tuple[int, int] = (224, 224),
        temporal_window: int = 16,
        stride: int = 16,
        sample_rate: int = 4,
        max_frames: int = 300,
        batch_size: int = 32,
        device: str = "cuda",
    ):
        self.source_dir = Path(source_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.target_size = target_size
        self.temporal_window = temporal_window
        self.stride = stride
        self.sample_rate = sample_rate
        self.max_frames = max_frames
        self.batch_size = batch_size
        self.device = device
        self._init_extractor()

    def _init_extractor(self):
        from preprocessing.feature_extractor import TemporalFeatureExtractor
        self.extractor = TemporalFeatureExtractor("s3d", self.device)
        self.extractor.model = self.extractor.model.to(self.device)

    def load_video_frames(self, video_path: Path) -> Optional[np.ndarray]:
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            return None

        frames = []
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            if frame_count % self.sample_rate == 0:
                frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                frame = cv2.resize(frame, self.target_size)
                frames.append(frame)
                if len(frames) >= self.max_frames:
                    break
            frame_count += 1

        cap.release()

        if len(frames) < self.temporal_window:
            return None

        return np.array(frames)

    def extract_features(self, frames: np.ndarray) -> np.ndarray:
        segments = []
        for i in range(0, len(frames) - self.temporal_window + 1, self.stride):
            seg = frames[i:i + self.temporal_window]
            segments.append(seg)

        if len(segments) == 0:
            return np.array([])

        all_features = []
        for batch_start in range(0, len(segments), self.batch_size):
            batch_segs = segments[batch_start:batch_start + self.batch_size]
            batch_tensors = []
            for seg in batch_segs:
                processed = []
                for frame in seg:
                    img = Image.fromarray(frame)
                    img = F.resize(img, [224, 224])
                    img = F.to_tensor(img)
                    img = F.normalize(img, mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
                    processed.append(img)
                tensor = torch.stack(processed, dim=1)
                batch_tensors.append(tensor)
            batch_input = torch.stack(batch_tensors, dim=0).to(self.device)
            with torch.no_grad():
                batch_output = self.extractor.model(batch_input)
            all_features.append(batch_output.cpu().numpy())

        return np.concatenate(all_features, axis=0)

    def process_class(self, class_dir: Path, label: int) -> List[Dict]:
        class_name = class_dir.name
        out_dir = self.output_dir / class_name
        out_dir.mkdir(parents=True, exist_ok=True)

        video_files = sorted(class_dir.glob("*.mp4"))
        if not video_files:
            video_files = sorted(class_dir.glob("*.avi"))

        metadata = []
        for video_path in tqdm(video_files, desc=f"{class_name} ({label})"):
            video_id = video_path.stem
            out_path = out_dir / f"{video_id}.npy"

            if out_path.exists():
                existing = np.load(str(out_path))
                metadata.append({
                    "video_id": video_id,
                    "path": str(out_path),
                    "split": "indonesia",
                    "category": class_name,
                    "label": label,
                    "segments": existing.shape[0],
                    "feature_dim": existing.shape[1],
                })
                continue

            frames = self.load_video_frames(video_path)
            if frames is None:
                tqdm.write(f"  Skipping {video_id}: too few frames")
                continue

            features = self.extract_features(frames)
            if len(features) == 0:
                continue

            np.save(out_path, features)
            metadata.append({
                "video_id": video_id,
                "path": str(out_path),
                "split": "indonesia",
                "category": class_name,
                "label": label,
                "segments": len(features),
                "feature_dim": features.shape[1],
                "total_frames": len(frames),
            })

        return metadata

    def run(self):
        all_metadata = []
        for class_name in ["anomaly", "normal"]:
            class_dir = self.source_dir / class_name
            if not class_dir.exists():
                continue
            label = 1 if class_name == "anomaly" else 0
            print(f"Processing {class_name} (label={label})...")
            meta = self.process_class(class_dir, label)
            all_metadata.extend(meta)

        meta_path = self.output_dir / "metadata.json"
        with open(meta_path, "w") as f:
            json.dump(all_metadata, f, indent=2)

        print(f"\n{'='*50}")
        print(f"PREPROCESSING INDONESIA V3 COMPLETE")
        print(f"{'='*50}")
        print(f"Total videos processed: {len(all_metadata)}")
        labels = [m["label"] for m in all_metadata]
        print(f"Normal: {sum(1 for l in labels if l == 0)} | Anomaly: {sum(1 for l in labels if l == 1)}")
        print(f"Metadata: {meta_path}")
        return all_metadata


if __name__ == "__main__":
    processor = IndonesiaV3Processor(device="cuda")
    processor.run()
