import cv2
import numpy as np
from pathlib import Path
from typing import List, Optional


class FrameExtractor:
    def __init__(self, target_size: tuple = (640, 640), fps: int = 1):
        self.target_size = target_size
        self.fps = fps

    def extract(self, video_path: str) -> Optional[List[np.ndarray]]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            return None

        frames = []
        frame_count = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_count % self.fps == 0:
                resized = cv2.resize(frame, self.target_size)
                frames.append(resized)

            frame_count += 1

        cap.release()
        return frames

    def validate_duration(self, video_path: str, min_frames: int = 16) -> bool:
        cap = cv2.VideoCapture(video_path)
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        cap.release()
        return total >= min_frames
