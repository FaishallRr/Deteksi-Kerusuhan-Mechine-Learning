import cv2
import numpy as np
from pathlib import Path


def generate_dataset(
    output_dir: str = "sample_videos",
    num_normal: int = 5,
    num_anomaly: int = 5,
    duration_seconds: int = 20,
    fps: int = 1,
    width: int = 448,
    height: int = 448,
):
    output_dir = Path(output_dir)
    output_dir.mkdir(exist_ok=True)
    total_frames = duration_seconds * fps

    for i in range(num_normal):
        path = output_dir / f"normal_{i:02d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        for f in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            x = 50 + f * 3
            cv2.rectangle(frame, (x % 300 + 50, 50), (x % 300 + 150, 150), (80, 80, 80), -1)
            cv2.putText(frame, "NORMAL", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
            out.write(frame)
        out.release()
        print(f"Generated {path}")

    for i in range(num_anomaly):
        path = output_dir / f"anomaly_{i:02d}.mp4"
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out = cv2.VideoWriter(str(path), fourcc, fps, (width, height))
        for f in range(total_frames):
            frame = np.zeros((height, width, 3), dtype=np.uint8)
            cv2.rectangle(frame, (100, 80), (200, 180), (0, 0, 200), -1)
            cv2.rectangle(frame, (250, 100), (350, 200), (0, 0, 200), -1)
            cv2.circle(frame, (150, 130), 15, (0, 0, 255), -1)
            cv2.circle(frame, (300, 150), 15, (0, 0, 255), -1)
            cv2.line(frame, (200, 130), (250, 150), (0, 0, 255), 2)
            cv2.putText(frame, "ANOMALY", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)
            out.write(frame)
        out.release()
        print(f"Generated {path}")


if __name__ == "__main__":
    generate_dataset(num_normal=5, num_anomaly=5)
    print("\nDataset generated: 5 normal + 5 anomaly videos")
