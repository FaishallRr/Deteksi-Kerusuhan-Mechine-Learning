"""
Extract sajam weapon crops from existing anomaly videos using our trained YOLO model.
More reliable than Google Images scraping — gives real-world weapon images from tawuran/riot footage.
"""
import sys
sys.path.insert(0, ".")

import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO
import time

OUTPUT_DIR = Path("dataset/sajam_extracted_crops")
INDO_MODEL_PATH = "models/yolo11n_indo.pt"
CONF_THRESHOLD = 0.25

# Source directories to extract from
SOURCE_DIRS = [
    Path("sample_videos/indonesia_v4"),
    Path("sample_videos/indonesia_v5"),
    Path("sample_videos/indonesia_v6/anomaly"),
]

WEAPON_CLASSES = {
    0: "celurit", 1: "golok", 2: "kapak",
    3: "pedang", 4: "pisau", 5: "pistol", 6: "senapan",
}


def extract_crops():
    model = YOLO(INDO_MODEL_PATH)
    print(f"[SAJAM CROPS] Loaded model: {INDO_MODEL_PATH}")
    print(f"[SAJAM CROPS] Classes: {model.names}")
    print()

    total_crops = 0
    total_frames = 0

    for src_dir in SOURCE_DIRS:
        if not src_dir.exists():
            print(f"  SKIP (not found): {src_dir}")
            continue

        videos = list(src_dir.glob("*.mp4")) + list(src_dir.glob("*.avi"))
        print(f"\n[SAJAM CROPS] Processing {src_dir}/ — {len(videos)} videos")

        for vpath in videos:
            cap = cv2.VideoCapture(str(vpath))
            fps = cap.get(cv2.CAP_PROP_FPS)
            total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            if total < 10:
                cap.release()
                continue

            print(f"    {vpath.name}: {total}f @ {fps:.1f}fps", end=" ")

            # Sample every 15 frames
            frame_count = 0
            video_crops = 0
            sampled_frames = 0

            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame_count += 1

                if frame_count % 15 != 0:
                    continue

                sampled_frames += 1
                results = model(frame, device="cuda", verbose=False)[0]

                for box in results.boxes:
                    cls_id = int(box.cls[0])
                    conf = float(box.conf[0])
                    if conf < CONF_THRESHOLD:
                        continue
                    if cls_id not in WEAPON_CLASSES:
                        continue

                    x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    if x2 - x1 < 20 or y2 - y1 < 20:
                        continue

                    crop = frame[y1:y2, x1:x2]
                    class_name = WEAPON_CLASSES[cls_id]
                    class_dir = OUTPUT_DIR / class_name
                    class_dir.mkdir(parents=True, exist_ok=True)

                    crop_path = class_dir / f"{vpath.stem}_f{frame_count}_c{conf:.2f}.jpg"
                    cv2.imwrite(str(crop_path), crop)
                    video_crops += 1
                    total_crops += 1

            cap.release()
            print(f"→ {video_crops} crops")

    print(f"\n{'='*50}")
    print(f"EXTRACTION COMPLETE")
    print(f"{'='*50}")
    print(f"  Total crops: {total_crops}")
    print(f"  Output: {OUTPUT_DIR}")
    print(f"  Classes: {list(WEAPON_CLASSES.values())}")


if __name__ == "__main__":
    extract_crops()
