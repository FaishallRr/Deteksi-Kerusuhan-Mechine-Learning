import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
from pathlib import Path
from collections import deque
from inference import AnomalyDetector


def run_headless(detector, video_path: str):
    cap = cv2.VideoCapture(video_path)
    frames_buffer = []
    score_buffer = deque(maxlen=10)
    total_frames = 0
    alert_triggered = False

    print(f"Processing: {video_path}")

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        resized = cv2.resize(frame, (640, 640))
        frames_buffer.append(resized)
        total_frames += 1

        if len(frames_buffer) >= 16:
            mil_features = detector.extractor.extract(frames_buffer[-16:])
            mil_score = detector.mil.predict_bag(mil_features)

            yolo_objects = detector.yolo.detect(resized)
            faces = detector.yolo.detect_faces(resized)
            if faces:
                yolo_objects["faces"] = faces
            plate = detector.yolo.detect_plate(resized)
            yolo_objects["plate"] = plate

            fused_score = detector._compute_fused_score(mil_score, yolo_objects)
            score_buffer.append(fused_score)

            if (
                fused_score >= 0.8
                and len(score_buffer) >= 10
                and all(s >= 0.8 for s in score_buffer)
            ):
                if not alert_triggered:
                    weapons = len(yolo_objects.get("weapons", []))
                    persons = len(yolo_objects.get("persons", []))
                    plate_info = plate if plate and plate.get("plate") else None
                    print(
                        f"  ALERT at frame {total_frames} | "
                        f"Score: {fused_score:.2f} | "
                        f"MIL: {mil_score:.2f} | "
                        f"Persons: {persons} | "
                        f"Weapons: {weapons}"
                    )
                    if plate_info:
                        print(f"  Plate: {plate_info['plate']} ({plate_info['confidence']:.0%})")
                    alert_triggered = True

    cap.release()
    return {
        "total_frames": total_frames,
        "alert_triggered": alert_triggered,
        "avg_score": np.mean(score_buffer) if score_buffer else 0,
    }


if __name__ == "__main__":
    detector = AnomalyDetector("config.yaml")

    for vid_type, label in [
        ("sample_videos/test_normal.mp4", "NORMAL"),
        ("sample_videos/test_anomaly.mp4", "ANOMALY"),
        ("sample_videos/normal_00.mp4", "NORMAL"),
        ("sample_videos/anomaly_00.mp4", "ANOMALY"),
    ]:
        if not Path(vid_type).exists():
            print(f"Skipping {vid_type} (not found)")
            continue
        result = run_headless(detector, vid_type)
        status = "✅ ALERT" if result["alert_triggered"] else "⛔ NO ALERT"
        print(f"  [{label}] {vid_type}: {result['total_frames']} frames, avg score={result['avg_score']:.2f} {status}")
        print()
