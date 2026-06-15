from pathlib import Path
from typing import List, Tuple, Optional
import numpy as np
import cv2
from tqdm import tqdm

from preprocessing.extract_frames import FrameExtractor
from preprocessing.feature_extractor import TemporalFeatureExtractor


class PreprocessingPipeline:
    def __init__(
        self,
        target_size: Tuple[int, int] = (640, 640),
        fps: int = 1,
        temporal_window: int = 16,
        device: str = "cpu",
    ):
        self.target_size = target_size
        self.fps = fps
        self.temporal_window = temporal_window
        self.frame_extractor = FrameExtractor(target_size, fps)
        self.feature_extractor = TemporalFeatureExtractor("s3d", device)

    def process_video(self, video_path: str) -> Optional[List[np.ndarray]]:
        if not Path(video_path).exists():
            raise FileNotFoundError(f"Video not found: {video_path}")
        if not self.frame_extractor.validate_duration(video_path, self.temporal_window):
            print(f"Skipping {video_path}: too short (< {self.temporal_window} frames)")
            return None
        frames = self.frame_extractor.extract(video_path)
        return frames

    def extract_features(self, frames: List[np.ndarray]) -> np.ndarray:
        segments = []
        for i in range(0, len(frames) - self.temporal_window + 1, self.temporal_window):
            segment = frames[i : i + self.temporal_window]
            segments.append(segment)
        if len(segments) == 0:
            return np.array([])
        features = []
        for seg in tqdm(segments, desc="Extracting features"):
            feat = self.feature_extractor.extract(seg)
            features.append(feat)
        return np.array(features)

    def run(self, video_path: str) -> Optional[np.ndarray]:
        print(f"[Pipeline] Processing: {video_path}")
        frames = self.process_video(video_path)
        if frames is None or len(frames) < self.temporal_window:
            print(f"[Pipeline] Insufficient frames ({len(frames) if frames else 0})")
            return None
        print(f"[Pipeline] Extracted {len(frames)} frames")
        features = self.extract_features(frames)
        print(f"[Pipeline] Extracted {len(features)} feature vectors (dim={features.shape[1] if features.ndim > 1 else 'N/A'})")
        return features


if __name__ == "__main__":
    pipeline = PreprocessingPipeline()
    for vid in ["sample_videos/test_normal.mp4", "sample_videos/test_anomaly.mp4"]:
        features = pipeline.run(vid)
        print(f"  Features shape: {features.shape}")
        print()
