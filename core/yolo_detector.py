import cv2
import numpy as np
import torch
import torch.nn as nn
from pathlib import Path
from typing import List, Dict, Any


class SajamVerifier(nn.Module):
    def __init__(self, input_size=64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(3, 16, 3, padding=1), nn.BatchNorm2d(16), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(16, 32, 3, padding=1), nn.BatchNorm2d(32), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(32, 64, 3, padding=1), nn.BatchNorm2d(64), nn.ReLU(), nn.MaxPool2d(2),
            nn.Conv2d(64, 128, 3, padding=1), nn.BatchNorm2d(128), nn.ReLU(), nn.MaxPool2d(2),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Dropout(0.3),
            nn.Linear(128, 2),
        )
        self.input_size = input_size

    def forward(self, x):
        return self.net(x)

    def verify(self, frame: np.ndarray, bbox, device="cpu", threshold=0.5):
        x1, y1, x2, y2 = map(int, bbox)
        x1, y1 = max(0, x1), max(0, y1)
        x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
        if x2 - x1 < 10 or y2 - y1 < 10:
            return False, 0.0
        crop = frame[y1:y2, x1:x2]
        crop = cv2.resize(crop, (self.input_size, self.input_size))
        img = crop.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        tensor = torch.tensor(img, dtype=torch.float32).unsqueeze(0).to(device)
        with torch.no_grad():
            out = self(tensor)
            prob = torch.softmax(out, dim=1)
            sajam_prob = prob[0, 1].item()
        return sajam_prob >= threshold, sajam_prob


class YOLODetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, device: str = "cpu"):
        from ultralytics import YOLO
        self.base_model = YOLO("yolo11m.onnx")
        self.confidence_threshold = confidence_threshold
        self.device = device

        self.indo_model_path = "models/yolo11n_indo.pt"
        import os
        self.indo_model = YOLO(self.indo_model_path) if os.path.exists(self.indo_model_path) else None
        if self.indo_model:
            print(f"[YOLO] Loaded Indo weapon model ({list(self.indo_model.names.values())})")

        verifier_path = Path("models/sajam_cnn_verify.pt")
        self.sajam_verifier = None
        self.sajam_verifier_threshold = 0.5
        if verifier_path.exists():
            self.sajam_verifier = SajamVerifier().to(device)
            self.sajam_verifier.load_state_dict(torch.load(verifier_path, map_location=device, weights_only=True))
            self.sajam_verifier.eval()
            print(f"[YOLO] Loaded sajam verifier from {verifier_path}")

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        detections = {
            "persons": [],
            "weapons": [],
            "vehicles": [],
            "faces": [],
            "plates": [],
        }

        try:
            results = self.base_model(frame, device=self.device, verbose=False, iou=0.45, half=True, imgsz=800)[0]
        except Exception:
            results = self.base_model(frame, device=self.device, verbose=False, iou=0.45, half=True, imgsz=640)[0]
        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue
            xyxy = box.xyxy[0].tolist()
            label = results.names[cls_id]

            if label in ["car", "motorcycle", "bus", "truck", "bicycle"]:
                bw = xyxy[2] - xyxy[0]
                bh = xyxy[3] - xyxy[1]
                ratio = bw / bh if bh > 0 else 1
                area = bw * bh

                if label == "bicycle":
                    if ratio > 1.2 or area > 3000:
                        label = "motorcycle"
                    else:
                        label = "motorcycle" if conf > 0.3 else "vehicle"

                elif label == "motorcycle":
                    if area > 12000 and ratio < 1.8:
                        label = "car"
                    elif area > 25000 and ratio < 2.2:
                        label = "truck"

                elif label == "car":
                    if area < 500 and ratio < 1.6:
                        label = "motorcycle"
                    elif area > 20000 and ratio > 2.0:
                        label = "truck"

                elif label == "truck":
                    if area < 1500:
                        label = "car"
                    elif area < 3000 and conf < 0.4:
                        label = "car"

                elif label == "bus":
                    if area < 3000:
                        label = "truck"
                    elif area < 5000 and conf < 0.4:
                        label = "truck"

            detection = {"bbox": xyxy, "confidence": conf, "class": label}

            if label == "person":
                detections["persons"].append(detection)
            elif label in ["car", "motorcycle", "bus", "truck", "vehicle"]:
                detections["vehicles"].append(detection)

        if self.indo_model and len(detections["persons"]) > 0:
            try:
                w_results = self.indo_model(frame, device=self.device, verbose=False, iou=0.45, half=True, imgsz=800)[0]
            except Exception:
                w_results = self.indo_model(frame, device=self.device, verbose=False, iou=0.45, half=True, imgsz=640)[0]
            for box in w_results.boxes:
                cls_id = int(box.cls[0])
                conf = float(box.conf[0])
                if conf < self.confidence_threshold:
                    continue
                xyxy = box.xyxy[0].tolist()
                label = w_results.names[cls_id]
                detection = {"bbox": xyxy, "confidence": conf, "class": label}

                if self.sajam_verifier is not None:
                    verified, sajam_prob = self.sajam_verifier.verify(
                        frame, xyxy, self.device, self.sajam_verifier_threshold
                    )
                    detection["sajam_conf"] = round(sajam_prob, 3)
                    if verified:
                        detections["weapons"].append(detection)
                else:
                    detections["weapons"].append(detection)

        return detections

    def detect_faces(self, frame: np.ndarray) -> List[Dict]:
        faces = []
        try:
            from retinaface import RetinaFace
            resp = RetinaFace.detect_faces(frame)
            if isinstance(resp, dict):
                for key, val in resp.items():
                    facial_area = val.get("facial_area", [])
                    confidence = val.get("score", 0)
                    if confidence > self.confidence_threshold:
                        x1, y1, x2, y2 = facial_area
                        faces.append({
                            "bbox": [x1, y1, x2, y2],
                            "confidence": confidence,
                        })
        except ImportError:
            pass
        return faces

    def detect_plate(self, frame: np.ndarray) -> Dict:
        try:
            import easyocr
            reader = easyocr.Reader(["id"])
            results = reader.readtext(frame)
            for bbox, text, conf in results:
                if conf > 0.6 and len(text) > 3:
                    coords = [bbox[0][0], bbox[0][1], bbox[2][0], bbox[2][1]]
                    return {"plate": text, "confidence": conf, "bbox": coords}
        except ImportError:
            pass
        return {"plate": None, "confidence": 0.0, "bbox": None}
