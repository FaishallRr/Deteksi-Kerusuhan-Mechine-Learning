import sys, os, time
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from core.yolo_detector import YOLODetector
from utils.config_loader import load_config

config = load_config("config.yaml")
_device = config["general"]["device"]
detector = YOLODetector(
    config["model"]["yolo"]["model_path"],
    confidence_threshold=config["model"]["yolo"]["confidence_threshold"],
    device=_device,
    indo_model_path=config["model"]["yolo"]["indo_model_path"],
    sajam_verifier_threshold=config["model"]["yolo"].get("sajam_verifier_threshold", 0.65),
    smoothing_config=config.get("smoothing", {}),
)

video_path = sys.argv[1] if len(sys.argv) > 1 else "test_videos/fight_sample_1.mp4"
cap = cv2.VideoCapture(video_path)
fps = cap.get(cv2.CAP_PROP_FPS)
w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
out_name = os.path.splitext(os.path.basename(video_path))[0] + "_detected.mp4"
out_path = os.path.join("test_videos", out_name)
fourcc = cv2.VideoWriter_fourcc(*"mp4v")
out = cv2.VideoWriter(out_path, fourcc, fps, (w, h))

frame_idx = 0
t0 = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    yolo_objects = detector.detect(frame)
    weapons = yolo_objects.get("weapons", [])
    persons = yolo_objects.get("persons", [])
    vehicles = yolo_objects.get("vehicles", [])

    for p in persons:
        x1, y1, x2, y2 = map(int, p["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        cv2.putText(frame, f'Person {p["confidence"]:.2f}', (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for w in weapons:
        x1, y1, x2, y2 = map(int, w["bbox"])
        bw, bh = x2 - x1, y2 - y1
        shrink = 0.20
        dx, dy = int(bw * shrink), int(bh * shrink)
        x1, y1 = x1 + dx, y1 + dy
        x2, y2 = x2 - dx, y2 - dy
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
        wtype = w.get("class", "SAJAM")
        wconf = w.get("sajam_conf", w.get("confidence", 0))
        label = f'{wtype} {wconf:.2f}'
        (tw, th), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 2)
        cv2.rectangle(frame, (x1, y1-th-6), (x1+tw+6, y1), (0, 0, 255), -1)
        cv2.putText(frame, label, (x1+3, y1-3), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 2)

    for v in vehicles:
        x1, y1, x2, y2 = map(int, v["bbox"])
        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 165, 0), 2)
        cv2.putText(frame, v.get("class", "vehicle"), (x1, y1-5),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 165, 0), 2)

    info = f"Frame:{frame_idx}/{total} P:{len(persons)} W:{len(weapons)} V:{len(vehicles)}"
    cv2.putText(frame, info, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    out.write(frame)

    if frame_idx % 100 == 0:
        print(f"  Progress: {frame_idx}/{total} ({100*frame_idx/total:.0f}%)")

elapsed = time.time() - t0
cap.release()
out.release()
print(f"\n✅ Selesai! Video output: {out_path}")
print(f"   {frame_idx} frames processed in {elapsed:.1f}s ({frame_idx/elapsed:.1f} fps)")
print(f"   Buka file {out_path} untuk lihat bounding box")
