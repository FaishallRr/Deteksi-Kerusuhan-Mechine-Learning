import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2, json, time
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

video_path = sys.argv[1] if len(sys.argv) > 1 else "fight_sample_1.mp4"
cap = cv2.VideoCapture(video_path)
total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
fps = cap.get(cv2.CAP_PROP_FPS)
print(f"Video: {video_path}")
print(f"Frames: {total}, FPS: {fps:.1f}")
print(f"{'Frame':>6} | {'P':>2} {'W':>2} {'V':>2} | {'Score':>6} | {'Persons':>2} {'Weapons':>2} {'Sajam':>6}")
print("-" * 60)

frame_idx = 0
t0 = time.time()
while True:
    ret, frame = cap.read()
    if not ret:
        break
    frame_idx += 1
    if frame_idx % 30 != 0 and frame_idx < total:
        continue
    yolo_objects = detector.detect(frame, smooth_weapons=False)
    weapons = yolo_objects.get("weapons", [])
    persons = yolo_objects.get("persons", [])
    vehicles = yolo_objects.get("vehicles", [])
    n_w = len(weapons)
    n_p = len(persons)
    n_v = len(vehicles)
    sajam_confs = [f"{w.get('sajam_conf', 0):.3f}" for w in weapons]
    score = min(n_w / 3.0, 1.0) * 0.30
    print(f"{frame_idx:>6}/{total:<6} | {n_p:>2} {n_w:>2} {n_v:>2} | {score:>6.3f} | {n_p:>2} {n_w:>2} {str(sajam_confs):>20}")

elapsed = time.time() - t0
cfps = frame_idx / elapsed if elapsed > 0 else 0
print(f"\nProcessed {frame_idx} frames in {elapsed:.1f}s ({cfps:.1f} fps)")
cap.release()
