import cv2
import numpy as np
from pathlib import Path


def generate_test_video(
    output_path: str = "sample_videos/test_normal.mp4",
    duration_seconds: int = 30,
    fps: int = 1,
    width: int = 448,
    height: int = 448,
    scenario: str = "normal",
):
    output_path = Path(output_path)
    output_path.parent.mkdir(exist_ok=True)

    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    out = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    total_frames = duration_seconds * fps

    for i in range(total_frames):
        frame = np.zeros((height, width, 3), dtype=np.uint8)

        if scenario == "normal":
            cv2.rectangle(frame, (50, 50), (150, 150), (100, 100, 100), -1)
            cv2.rectangle(frame, (250, 100), (350, 200), (100, 100, 100), -1)
            cv2.putText(
                frame,
                f"Frame {i+1}/{total_frames} - NORMAL",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (255, 255, 255),
                1,
            )

        elif scenario == "anomaly":
            if i > 10:
                cv2.rectangle(frame, (50, 50), (150, 150), (0, 0, 200), -1)
                cv2.circle(frame, (100, 100), 30, (0, 0, 255), -1)
            if i > 15:
                cv2.rectangle(frame, (250, 100), (350, 200), (0, 0, 200), -1)
                cv2.line(frame, (300, 150), (200, 250), (0, 0, 255), 3)
            cv2.putText(
                frame,
                f"Frame {i+1}/{total_frames} - ANOMALY",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                1,
            )

        out.write(frame)

    out.release()
    print(f"Generated: {output_path} ({duration_seconds}s, {fps} fps)")
    return str(output_path)


if __name__ == "__main__":
    generate_test_video("sample_videos/test_normal.mp4", scenario="normal")
    generate_test_video("sample_videos/test_anomaly.mp4", scenario="anomaly")
    print("Test videos created successfully")
