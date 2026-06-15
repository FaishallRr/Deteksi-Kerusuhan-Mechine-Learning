import cv2
import numpy as np
import json
from http.server import ThreadingHTTPServer, BaseHTTPRequestHandler
from datetime import datetime
from collections import deque
from core.yolo_detector import YOLODetector
import threading

detector = YOLODetector("yolo11m.pt", confidence_threshold=0.4, device="cuda")
score_buffer = deque(maxlen=10)
buffer_lock = threading.Lock()


class DetectHandler(BaseHTTPRequestHandler):
    def do_OPTIONS(self):
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")
        self.end_headers()

    def do_POST(self):
        global detector, score_buffer
        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length)

        nparr = np.frombuffer(body, np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            self._send_json({"error": "invalid image"})
            return

        yolo_objects = detector.detect(frame)

        weapons = yolo_objects.get("weapons", [])
        persons = yolo_objects.get("persons", [])
        n_w = len(weapons)
        n_p = len(persons)
        n_v = len(yolo_objects.get("vehicles", []))

        # Fight detection: IoU-based proximity
        fight = 0
        if n_p >= 2:
            close_pairs = 0
            for i in range(n_p):
                for j in range(i + 1, n_p):
                    bi = persons[i]["bbox"]
                    bj = persons[j]["bbox"]
                    ix1 = max(bi[0], bj[0]); iy1 = max(bi[1], bj[1])
                    ix2 = min(bi[2], bj[2]); iy2 = min(bi[3], bj[3])
                    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                    inter = iw * ih
                    ai = (bi[2]-bi[0]) * (bi[3]-bi[1])
                    aj = (bj[2]-bj[0]) * (bj[3]-bj[1])
                    min_area = min(ai, aj)
                    if min_area > 0 and (inter / min_area) > 0.15:
                        close_pairs += 1
            fight = min(close_pairs / max(n_p - 1, 1), 1.0)

        fused_score = min(0.35 * min(n_w, 3) + 0.2 * fight + 0.1 * min(n_p / 8, 1), 1.0)

        with buffer_lock:
            score_buffer.append(fused_score)
            is_alert = fused_score >= 0.6 and len(score_buffer) >= 10 and all(s >= 0.6 for s in score_buffer)

        # Build boxes array
        boxes = []
        for w in weapons:
            x1, y1, x2, y2 = map(int, w["bbox"])
            boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "t": w.get("class", "weapon"), "c": w["confidence"]})
        for p in persons:
            x1, y1, x2, y2 = map(int, p["bbox"])
            boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "t": "person", "c": p["confidence"]})
        for v in yolo_objects.get("vehicles", []):
            x1, y1, x2, y2 = map(int, v["bbox"])
            boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "t": v.get("class", "vehicle"), "c": v["confidence"]})
        for f in yolo_objects.get("faces", []):
            x1, y1, x2, y2 = map(int, f["bbox"])
            boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "t": "face", "c": f["confidence"]})

        # NMS
        if len(boxes) > 1:
            boxes.sort(key=lambda b: -b["c"])
            keep = [True] * len(boxes)
            for i in range(len(boxes)):
                if not keep[i]:
                    continue
                bi = boxes[i]
                for j in range(i + 1, len(boxes)):
                    if not keep[j]:
                        continue
                    bj = boxes[j]
                    ix1 = max(bi["x"], bj["x"]); iy1 = max(bi["y"], bj["y"])
                    ix2 = min(bi["x"] + bi["w"], bj["x"] + bj["w"]); iy2 = min(bi["y"] + bi["h"], bj["y"] + bj["h"])
                    iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                    inter = iw * ih
                    union = bi["w"] * bi["h"] + bj["w"] * bj["h"] - inter
                    iou = inter / union if union > 0 else 0
                    if iou > 0.35:
                        keep[j] = False
            boxes = [b for i, b in enumerate(boxes) if keep[i]]

        response = {
            "type": "detection",
            "s": round(fused_score, 3),
            "w": n_w, "p": n_p, "v": n_v,
            "a": is_alert,
            "t": datetime.now().strftime("%d %b %Y %H:%M:%S"),
            "boxes": boxes,
        }
        self._send_json(response)

    def _send_json(self, data):
        body = json.dumps(data).encode()
        self.send_response(200)
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def log_message(self, format, *args):
        pass


def start_detect_server(host="0.0.0.0", port=8765):
    server = ThreadingHTTPServer((host, port), DetectHandler)
    print(f"[DetectServer] Listening on {host}:{port}")
    server.serve_forever()
