import cv2
import numpy as np
from pathlib import Path
from ultralytics import YOLO

INDO_MODEL = "models/yolo11n_indo.pt"
CROP_SIZE = 64
OUTPUT = Path("dataset/sajam_crops")
OUTPUT.mkdir(parents=True, exist_ok=True)

def extract_crops_from_videos(video_dir: Path, label: str, conf_thresh: float = 0.3):
    model = YOLO(INDO_MODEL)
    videos = list(video_dir.glob("*.mp4")) + list(video_dir.glob("*.mkv")) + list(video_dir.glob("*.webm"))
    print(f"[{label}] {len(videos)} videos in {video_dir}")

    out_dir = OUTPUT / label
    out_dir.mkdir(parents=True, exist_ok=True)

    total_crops = 0
    for vpath in videos:
        cap = cv2.VideoCapture(str(vpath))
        frame_count = 0
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frame_count += 1
            if frame_count % 30 != 0:
                continue

            results = model(frame, verbose=False)[0]
            for box in results.boxes:
                conf = float(box.conf[0])
                if conf < conf_thresh:
                    continue
                x1, y1, x2, y2 = map(int, box.xyxy[0].tolist())
                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                if x2 - x1 < 10 or y2 - y1 < 10:
                    continue

                crop = frame[y1:y2, x1:x2]
                crop = cv2.resize(crop, (CROP_SIZE, CROP_SIZE))
                cls_name = results.names[int(box.cls[0])]
                fname = f"{vpath.stem}_f{frame_count}_{cls_name}_{conf:.2f}.jpg"
                cv2.imwrite(str(out_dir / fname), crop)
                total_crops += 1

        cap.release()
        print(f"  {vpath.name}: {total_crops} crops so far")

    print(f"[{label}] Total: {total_crops} crops -> {out_dir}")
    return total_crops

if __name__ == "__main__":
    import sys

    base = Path("sample_videos")

    if len(sys.argv) > 1 and sys.argv[1] == "positive":
        extract_crops_from_videos(base / "indonesia_v4", "sajam_pos")
        extract_crops_from_videos(base / "indonesia_v5", "sajam_pos")
    elif len(sys.argv) > 1 and sys.argv[1] == "negative":
        extract_crops_from_videos(base / "cctv_indonesia", "sajam_neg")
        extract_crops_from_videos(base / "cctv_indonesia_scraped", "sajam_neg")
        extract_crops_from_videos(base / "fp_test", "sajam_neg")
        extract_crops_from_videos(base / "demo_indonesia", "sajam_neg")
    else:
        print("Usage: python core/generate_sajam_crops.py [positive|negative]")
