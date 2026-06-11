import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
import json


class EvidenceManager:
    def __init__(self, output_dir: str = "./evidence"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)

    def save_screenshot(self, frame: np.ndarray, report_id: str) -> str:
        filename = f"{report_id}_screenshot.jpg"
        path = self.output_dir / filename
        cv2.imwrite(str(path), frame)
        return str(path)

    def save_video_clip(
        self, frames: list, report_id: str, fps: int = 1
    ) -> str:
        filename = f"{report_id}_clip.mp4"
        path = self.output_dir / filename
        height, width = frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        for frame in frames:
            out.write(frame)
        out.release()
        return str(path)

    def save_face_crop(self, face_img: np.ndarray, report_id: str, idx: int) -> str:
        filename = f"{report_id}_face_{idx}.jpg"
        path = self.output_dir / filename
        cv2.imwrite(str(path), face_img)
        return str(path)

    def save_plate_crop(self, plate_img: np.ndarray, report_id: str) -> str:
        filename = f"{report_id}_plate.jpg"
        path = self.output_dir / filename
        cv2.imwrite(str(path), plate_img)
        return str(path)

    def save_metadata(self, report_id: str, data: dict):
        filename = f"{report_id}_metadata.json"
        path = self.output_dir / filename
        with open(path, "w") as f:
            json.dump(data, f, indent=2)
        return str(path)
