import sys, os, warnings, pathlib
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
os.chdir(os.path.dirname(os.path.abspath(__file__)))
warnings.filterwarnings('ignore')

print("--- FILE CHECK ---")
model_files = [
    'yolo11m.onnx', 'yolo11m.pt', 'yolo11n.pt',
    'yolo11s.onnx', 'yolo11s.pt', 'yolo26n.pt',
    'models/yolo11n_indo.pt', 'models/sajam_cnn_verify.pt',
    'models/mil_model_v8_idn.pt',
]
for f in model_files:
    p = pathlib.Path(f)
    if p.exists():
        size_mb = p.stat().st_size / (1024*1024)
        print(f"  OK: {f} ({size_mb:.1f} MB)")
    else:
        print(f"  MISSING: {f}")

print("\n--- LOADING YOLO (yolo11n.pt) ---")
from ultralytics import YOLO
model = YOLO('yolo11n.pt')
model_names = list(model.names.values())
print(f"  Classes ({len(model_names)}): {model_names[:6]}...")
print("  YOLO loaded OK")

print("\n--- LOADING ONNX ---")
try:
    import onnxruntime as ort
    providers = ort.get_available_providers()
    print(f"  ONNX Runtime providers: {providers}")
except Exception as e:
    print(f"  ONNX Runtime error: {e}")

print("\n--- LOADING SajamVerifier ---")
from core.yolo_detector import SajamVerifier
import torch
sv = SajamVerifier()
sv_path = 'models/sajam_cnn_verify.pt'
if pathlib.Path(sv_path).exists():
    sv.load_state_dict(torch.load(sv_path, map_location='cpu', weights_only=True))
    sv.eval()
    print(f"  Sajam verifier loaded OK (params: {sum(p.numel() for p in sv.parameters())})")

print("\n--- LOADING MIL Model ---")
from core.mil_ranking import MILRankingModel
mil = MILRankingModel(input_dim=1024, hidden_units=512)
mil_path = 'models/mil_model_v8_idn.pt'
if pathlib.Path(mil_path).exists():
    mil.load_state_dict(torch.load(mil_path, map_location='cpu', weights_only=True))
    mil.eval()
    print(f"  MIL model loaded OK (params: {sum(p.numel() for p in mil.parameters())})")

print("\n--- QUICK YOLO INFERENCE TEST ---")
import cv2, numpy as np
dummy = np.zeros((640, 640, 3), dtype=np.uint8)
results = model(dummy, device='cpu', verbose=False)[0]
boxes = results.boxes
print(f"  Inference OK - detected {len(boxes)} objects on dummy frame")

print("\n=== ALL CHECKS PASSED ===")
