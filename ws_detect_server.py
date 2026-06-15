import asyncio
import websockets
import cv2
import numpy as np
import json
import time
import os
import psutil
from datetime import datetime
from collections import deque
from core.yolo_detector import YOLODetector
import threading

detector = YOLODetector("yolo11m.onnx", confidence_threshold=0.15, device="cuda")
score_buffer = deque(maxlen=10)
buffer_lock = threading.Lock()

_server_start = time.time()
_total_frames = 0
_client_count = 0

_person_history = {}
_next_person_id = 0
_PERSON_MATCH_IOU = 0.3
_PERSON_EXPIRE_FRAMES = 15


def _match_persons(current_persons):
    global _next_person_id, _person_history
    matched = {}
    used = set()
    for pid, hist in list(_person_history.items()):
        if hist["missed"] >= _PERSON_EXPIRE_FRAMES:
            del _person_history[pid]
            continue
        best_idx = -1
        best_iou = _PERSON_MATCH_IOU
        for j, p in enumerate(current_persons):
            if j in used:
                continue
            bx = p["bbox"]
            ph = hist["bbox"]
            ix1 = max(bx[0], ph[0]); iy1 = max(bx[1], ph[1])
            ix2 = min(bx[2], ph[2]); iy2 = min(bx[3], ph[3])
            iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
            inter = iw * ih
            union = (bx[2]-bx[0])*(bx[3]-bx[1]) + (ph[2]-ph[0])*(ph[3]-ph[1]) - inter
            iou = inter / union if union > 0 else 0
            if iou > best_iou:
                best_iou = iou
                best_idx = j
        if best_idx >= 0:
            matched[pid] = best_idx
            used.add(best_idx)
    for j, p in enumerate(current_persons):
        if j in used:
            continue
        pid = _next_person_id
        _next_person_id += 1
        cx = (p["bbox"][0] + p["bbox"][2]) / 2
        cy = (p["bbox"][1] + p["bbox"][3]) / 2
        _person_history[pid] = {
            "bbox": p["bbox"],
            "cx": cx, "cy": cy,
            "positions": deque([(cx, cy)], maxlen=10),
            "missed": 0,
            "velocities": deque(maxlen=10),
        }
        matched[pid] = j
    return matched


def _update_person_tracks(matched, current_persons):
    for pid, idx in matched.items():
        p = current_persons[idx]
        hist = _person_history[pid]
        bx = p["bbox"]
        cx = (bx[0] + bx[2]) / 2
        cy = (bx[1] + bx[3]) / 2
        hist["bbox"] = bx
        if len(hist["positions"]) > 0:
            prev = hist["positions"][-1]
            dx = cx - prev[0]
            dy = cy - prev[1]
            vel = (dx * dx + dy * dy) ** 0.5
            hist["velocities"].append(vel)
        hist["positions"].append((cx, cy))
        hist["cx"] = cx
        hist["cy"] = cy
        hist["missed"] = 0
    for hist in _person_history.values():
        if hist["missed"] > 0:
            hist["velocities"].append(0)
        hist["missed"] += 1


def _compute_dynamic_score(weapons, persons, vehicles, person_velocities, close_pairs, n_p):
    components = {}
    w_score = min(len(weapons) / 3.0, 1.0)
    components["weapon"] = w_score

    crowd_score = 0.0
    if n_p >= 3:
        crowd_density = close_pairs / max(n_p - 1, 1)
        crowd_score = min(crowd_density * 1.2, 1.0)
    components["crowd"] = crowd_score

    speed_score = 0.0
    if person_velocities:
        avg_vel = sum(person_velocities) / len(person_velocities)
        frame_area = 640 * 360
        norm_vel = avg_vel / (frame_area ** 0.5) * 100
        speed_score = min(norm_vel / 3.0, 1.0)
    components["speed"] = speed_score

    weapon_person_proximity = 0.0
    if weapons and n_p > 0:
        close_count = 0
        for w in weapons:
            wx = (w["bbox"][0] + w["bbox"][2]) / 2
            wy = (w["bbox"][1] + w["bbox"][3]) / 2
            for p in persons:
                px = (p["bbox"][0] + p["bbox"][2]) / 2
                py = (p["bbox"][1] + p["bbox"][3]) / 2
                dist = ((wx - px) ** 2 + (wy - py) ** 2) ** 0.5
                if dist < 200:
                    close_count += 1 / max(len(weapons), 1)
        weapon_person_proximity = min(close_count / max(len(weapons), 1), 1.0)
    components["proximity"] = weapon_person_proximity

    vehicle_anomaly = 0.0
    if vehicles and n_p > 0:
        for v in vehicles:
            vx = (v["bbox"][0] + v["bbox"][2]) / 2
            vy = (v["bbox"][1] + v["bbox"][3]) / 2
            for p in persons:
                px = (p["bbox"][0] + p["bbox"][2]) / 2
                py = (p["bbox"][1] + p["bbox"][3]) / 2
                dist = ((vx - px) ** 2 + (vy - py) ** 2) ** 0.5
                if dist < 150:
                    vehicle_anomaly = max(vehicle_anomaly, 0.4)
    components["vehicle"] = vehicle_anomaly

    weights = {
        "weapon": 0.30,
        "crowd": 0.20,
        "speed": 0.15,
        "proximity": 0.20,
        "vehicle": 0.15,
    }
    score = sum(components[k] * weights[k] for k in weights)
    score = min(score, 1.0)

    return score, components


async def monitor(interval=30):
    global _total_frames
    proc = psutil.Process(os.getpid())
    while True:
        await asyncio.sleep(interval)
        elapsed = time.time() - _server_start
        fps = _total_frames / elapsed if elapsed > 0 else 0
        mem_mb = proc.memory_info().rss / (1024 * 1024)
        cpu_pct = proc.cpu_percent()
        print(f"[Monitor] {elapsed:.0f}s | Frames: {_total_frames} | FPS: {fps:.1f} | "
              f"Mem: {mem_mb:.0f}MB | CPU: {cpu_pct:.0f}% | Clients: {_client_count}", flush=True)


async def handler(websocket):
    global _total_frames, _client_count
    _client_count += 1
    cid = _client_count
    addr = websocket.remote_address
    print(f"[WS] Client#{cid} connected: {addr}")
    frames = 0
    t0 = time.time()
    try:
        async for message in websocket:
            if not isinstance(message, (bytes, bytearray)):
                continue

            nparr = np.frombuffer(message, np.uint8)
            frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
            if frame is None:
                continue

            if cv2.mean(frame)[0] < 5:
                continue

            yolo_objects = detector.detect(frame)

            weapons = yolo_objects.get("weapons", [])
            persons = yolo_objects.get("persons", [])
            n_w = len(weapons)
            n_p = len(persons)
            n_v = len(yolo_objects.get("vehicles", []))

            fight = 0
            close_pairs = 0
            if n_p >= 2:
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

            matched = _match_persons(persons)
            _update_person_tracks(matched, persons)
            person_velocities = []
            for pid, idx in matched.items():
                hist = _person_history[pid]
                if hist["velocities"]:
                    person_velocities.extend(hist["velocities"])

            fused_score, components = _compute_dynamic_score(
                weapons, persons, yolo_objects.get("vehicles", []),
                person_velocities, close_pairs, n_p
            )

            with buffer_lock:
                score_buffer.append(fused_score)
                recent = list(score_buffer)
                is_alert = fused_score >= 0.6 and len(recent) >= 8 and all(s >= 0.55 for s in recent[-8:])

            if fused_score >= 0.7:
                status = "bahaya"
            elif fused_score >= 0.4:
                status = "mencurigakan"
            else:
                status = "normal"

            boxes = []
            for w in weapons:
                x1, y1, x2, y2 = map(int, w["bbox"])
                if (x2 - x1) * (y2 - y1) < 100: continue
                boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "t": w.get("class", "weapon"), "c": w["confidence"]})
            for p in persons:
                x1, y1, x2, y2 = map(int, p["bbox"])
                if (x2 - x1) * (y2 - y1) < 100: continue
                boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "t": "person", "c": p["confidence"]})
            for v in yolo_objects.get("vehicles", []):
                x1, y1, x2, y2 = map(int, v["bbox"])
                if (x2 - x1) * (y2 - y1) < 100: continue
                boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "t": v.get("class", "vehicle"), "c": v["confidence"]})
            for f in yolo_objects.get("faces", []):
                x1, y1, x2, y2 = map(int, f["bbox"])
                if (x2 - x1) * (y2 - y1) < 100: continue
                boxes.append({"x": x1, "y": y1, "w": x2 - x1, "h": y2 - y1, "t": "face", "c": f["confidence"]})

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
                        if iou > 0.25 or (bi["t"] == bj["t"] and iou > 0.15):
                            keep[j] = False
                boxes = [b for i, b in enumerate(boxes) if keep[i]]

            response = {
                "type": "detection",
                "s": round(fused_score, 3),
                "w": n_w, "p": n_p, "v": n_v,
                "a": is_alert,
                "st": status,
                "t": datetime.now().strftime("%d %b %Y %H:%M:%S"),
                "boxes": boxes,
            }
            await websocket.send(json.dumps(response))
            frames += 1
            _total_frames += 1
    except websockets.exceptions.ConnectionClosed:
        pass
    except Exception as e:
        print(f"[WS] Client#{cid} error: {e}")
    finally:
        _client_count -= 1
        elapsed = time.time() - t0
        cfps = frames / elapsed if elapsed > 0 else 0
        print(f"[WS] Client#{cid} disconnected. Frames: {frames}, Duration: {elapsed:.0f}s, FPS: {cfps:.1f}")


def start_ws_server(host="0.0.0.0", port=8765):
    async def serve():
        async with websockets.serve(handler, host, port, ping_interval=20):
            print(f"[DetectServer] WebSocket on ws://{host}:{port}", flush=True)
            asyncio.create_task(monitor(30))
            await asyncio.Event().wait()

    try:
        asyncio.run(serve())
    except OSError:
        print(f"[DetectServer] Port {port} busy, retry in 3s...", flush=True)
        time.sleep(3)
        asyncio.run(serve())
