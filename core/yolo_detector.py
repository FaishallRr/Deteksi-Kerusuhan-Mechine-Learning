import cv2
import numpy as np
from typing import List, Dict, Any


class YOLODetector:
    def __init__(self, model_path: str, confidence_threshold: float = 0.5, device: str = "cpu"):
        from ultralytics import YOLO
        self.model = YOLO(model_path)
        self.confidence_threshold = confidence_threshold
        self.device = device

    def detect(self, frame: np.ndarray) -> Dict[str, Any]:
        results = self.model(frame, device=self.device, verbose=False)[0]
        detections = {
            "persons": [],
            "weapons": [],
            "vehicles": [],
            "faces": [],
            "plates": [],
        }

        for box in results.boxes:
            cls_id = int(box.cls[0])
            conf = float(box.conf[0])
            if conf < self.confidence_threshold:
                continue
            xyxy = box.xyxy[0].tolist()
            label = results.names[cls_id]
            detection = {"bbox": xyxy, "confidence": conf}

            if label == "person":
                detections["persons"].append(detection)
            elif label in ["knife", "gun", "weapon"]:
                detections["weapons"].append(detection)
            elif label in ["car", "motorcycle", "bus", "truck"]:
                detections["vehicles"].append(detection)

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
