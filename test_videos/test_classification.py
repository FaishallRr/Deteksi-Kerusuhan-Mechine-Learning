import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
import cv2
import numpy as np
from pathlib import Path
from collections import deque

from core.yolo_detector import YOLODetector
from core.crowd_analyzer import CrowdAnalyzer
from utils.config_loader import load_config

config = load_config("config.yaml")
_device = config["general"]["device"]
detector = YOLODetector(
    config["model"]["yolo"]["model_path"],
    confidence_threshold=config["model"]["yolo"]["confidence_threshold"],
    device=_device,
)
crowd_analyzer = CrowdAnalyzer()

_SAMPLE_EVERY = 10

# Warmup
print("[Test] Warming up model...", flush=True)
_dummy = np.zeros((540, 960, 3), dtype=np.uint8)
detector.detect(_dummy)
print("[Test] Model ready.", flush=True)


def process_video(video_path):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    crowd_analyzer.reset()

    frame_idx = 0
    scene_counts = {"normal": 0, "demo_damai": 0, "demo_rusuh": 0}
    total_motion = 0.0
    motion_frames = 0
    total_persons = 0
    sample_count = 0
    prev_gray = None

    # Person tracking
    _person_tracks = {}
    _next_pid = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % _SAMPLE_EVERY != 0:
            continue
        if frame_idx % 100 == 0:
            print(f"  [Progress] {os.path.basename(video_path)} {frame_idx}/{total} ({100*frame_idx/total:.0f}%)", flush=True)

        # Motion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        motion_pct = 0.0
        if prev_gray is not None:
            diff = cv2.absdiff(prev_gray, gray)
            motion_pct = float(np.mean(diff > 25))
        prev_gray = gray
        total_motion += motion_pct
        motion_frames += 1

        # YOLO detection
        yolo_objects = detector.detect(frame, min_area=200)
        persons = yolo_objects.get("persons", [])
        n_p = len(persons)
        sample_count += 1
        total_persons += n_p

        # Person tracking for velocities (used by CrowdAnalyzer internally)
        matched = {}
        used = set()
        for pid, hist in list(_person_tracks.items()):
            if hist["missed"] >= 15:
                del _person_tracks[pid]
                continue
            best_idx = -1
            best_iou = 0.35
            for j, p in enumerate(persons):
                if j in used:
                    continue
                ix1 = max(hist["bbox"][0], p["bbox"][0]); iy1 = max(hist["bbox"][1], p["bbox"][1])
                ix2 = min(hist["bbox"][2], p["bbox"][2]); iy2 = min(hist["bbox"][3], p["bbox"][3])
                iw = max(0, ix2 - ix1); ih = max(0, iy2 - iy1)
                inter = iw * ih
                ua = (hist["bbox"][2]-hist["bbox"][0])*(hist["bbox"][3]-hist["bbox"][1]) + (p["bbox"][2]-p["bbox"][0])*(p["bbox"][3]-p["bbox"][1]) - inter
                iou = inter / ua if ua > 0 else 0
                if iou > best_iou:
                    best_iou = iou
                    best_idx = j
            if best_idx >= 0:
                matched[pid] = best_idx
                used.add(best_idx)
                bx = persons[best_idx]["bbox"]
                cx = (bx[0] + bx[2]) / 2
                cy = (bx[1] + bx[3]) / 2
                if len(hist["positions"]) > 0:
                    prev = hist["positions"][-1]
                    dx = cx - prev[0]
                    dy = cy - prev[1]
                    vel = (dx*dx + dy*dy)**0.5
                    hist["velocities"].append(vel)
                hist["positions"].append((cx, cy))
                hist["bbox"] = bx
                hist["missed"] = 0
        for j, p in enumerate(persons):
            if j in used:
                continue
            pid = _next_pid
            _next_pid += 1
            bx = p["bbox"]
            cx = (bx[0] + bx[2]) / 2
            cy = (bx[1] + bx[3]) / 2
            _person_tracks[pid] = {
                "bbox": bx,
                "positions": deque([(cx, cy)], maxlen=15),
                "velocities": deque(maxlen=15),
                "missed": 0,
            }
            matched[pid] = j
        for hist in _person_tracks.values():
            if hist["missed"] > 0:
                hist["velocities"].append(0)
            hist["missed"] += 1

        # Compute running ratio from tracked velocities
        velocities = []
        for hist in _person_tracks.values():
            if hist["velocities"]:
                velocities.extend(hist["velocities"])
        running_thresh = 10.0
        running_count = sum(1 for v in velocities if v > running_thresh) if velocities else 0
        running_ratio = running_count / max(len(velocities), 1)

        # Compute close pair ratio from person positions
        close_count = 0
        pair_count = 0
        positions = [(p["bbox"][0] + p["bbox"][2]) / 2 for p in persons]
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                dist = abs(positions[i] - positions[j])
                if dist < 50:
                    close_count += 1
                pair_count += 1
        close_pair_ratio = close_count / max(pair_count, 1)

        # CrowdAnalyzer: chaos index + scene classification
        chaos_value, chaos_details = crowd_analyzer.compute_chaos_index(frame, n_p, running_ratio, close_pair_ratio)
        scene = crowd_analyzer.classify_scene(n_p, chaos_value, running_ratio)
        scene_counts[scene] = scene_counts.get(scene, 0) + 1

    cap.release()
    total_samples = max(sum(scene_counts.values()), 1)
    avg_motion = (total_motion / max(motion_frames, 1)) * 100
    avg_p = total_persons / max(sample_count, 1)

    dominant_scene = max(scene_counts, key=scene_counts.get)
    pct_rusuh = scene_counts.get("demo_rusuh", 0) / total_samples * 100
    pct_damai = scene_counts.get("demo_damai", 0) / total_samples * 100
    pct_normal = scene_counts.get("normal", 0) / total_samples * 100

    return {
        "file": os.path.basename(video_path),
        "frames": total,
        "samples": total_samples,
        "dominant": dominant_scene,
        "%rusuh": pct_rusuh,
        "%damai": pct_damai,
        "%normal": pct_normal,
        "avg_persons": avg_p,
        "avg_motion_pct": avg_motion,
    }


def main():
    base = "sample_videos/indonesia_v7"
    is_full = "--full" in sys.argv
    all_rusuh = sorted(Path(f"{base}/demo_rusuh").glob("*.mp4"))
    all_damai = sorted(Path(f"{base}/demo_damai").glob("*.mp4"))
    categories = {
        "demo_rusuh": all_rusuh if is_full else all_rusuh[:2],
        "demo_damai": all_damai if is_full else all_damai[:2],
    }
    if not is_full:
        print(f"[Test] QUICK mode: {len(categories['demo_rusuh'])} rusuh + {len(categories['demo_damai'])} damai videos")
        print(f"[Test] Use --full to run all")

    HDR = f"{'Kategori':12s} | {'Video':32s} | {'Dom':6s} | {'%Rusuh':>7s} | {'%Damai':>7s} | {'%Normal':>7s} | {'AvgP':>4s} | {'AvgMot':>6s}"
    print(HDR)
    print("-" * len(HDR))

    all_results = []
    for cat_name, files in categories.items():
        for fpath in files:
            r = process_video(str(fpath))
            all_results.append(r)
            verdict = "✅" if (
                (cat_name == "demo_rusuh" and r["dominant"] == "demo_rusuh")
                or (cat_name == "demo_damai" and r["dominant"] in ("demo_damai", "normal"))
            ) else "❌"
            flagged = " 🏴" if (cat_name == "demo_damai" and r["%rusuh"] > 20) else ""
            print(f"{cat_name:12s} | {r['file']:32s} | {r['dominant']:6s} | {r['%rusuh']:6.1f}% | {r['%damai']:6.1f}% | {r['%normal']:6.1f}% | {r['avg_persons']:3.1f} | {r['avg_motion_pct']:5.2f}% {verdict}{flagged}")

    print(f"\n--- SUMMARY ---")
    for cat_name in categories:
        cat_results = []
        for r in all_results:
            rpath = Path(f"{base}/{cat_name}/{r['file']}")
            if rpath.exists():
                cat_results.append(r)
        if not cat_results:
            continue
        avg_rusuh = np.mean([r["%rusuh"] for r in cat_results])
        avg_damai = np.mean([r["%damai"] for r in cat_results])
        avg_normal = np.mean([r["%normal"] for r in cat_results])
        avg_mot = np.mean([r["avg_motion_pct"] for r in cat_results])
        correct = sum(1 for r in cat_results if (
            (cat_name == "demo_rusuh" and r["dominant"] == "demo_rusuh")
            or (cat_name == "demo_damai" and r["dominant"] in ("demo_damai", "normal"))
        ))
        wrong = len(cat_results) - correct
        print(f"{cat_name:12s}: avg_rusuh={avg_rusuh:5.1f}% avg_damai={avg_damai:5.1f}% avg_normal={avg_normal:5.1f}% | avg_mot={avg_mot:.2f}% | {correct}/{len(cat_results)} ✅  {wrong} ❌")


if __name__ == "__main__":
    main()
