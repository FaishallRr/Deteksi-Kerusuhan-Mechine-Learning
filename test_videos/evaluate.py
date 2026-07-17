"""
Evaluasi pipeline lengkap: MIL model + CrowdAnalyzer pada test videos.
Usage:
    python test_videos/evaluate.py                         # quick (2 per class)
    python test_videos/evaluate.py --full                  # semua video
    python test_videos/evaluate.py --video path/video.mp4  # single video
"""

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import cv2
import torch
import numpy as np
from pathlib import Path
from collections import deque

from core.yolo_detector import YOLODetector
from core.crowd_analyzer import CrowdAnalyzer
from preprocessing.feature_extractor import TemporalFeatureExtractor
from core.mil_attention import AttentionMILModel
from utils.config_loader import load_config

config = load_config("config.yaml")
device = config["general"]["device"]

# Init models
print("[Init] Loading models...")
detector = YOLODetector(
    config["model"]["yolo"]["model_path"],
    confidence_threshold=config["model"]["yolo"]["confidence_threshold"],
    device=device,
)
extractor = TemporalFeatureExtractor(architecture="s3d", device=device)
mil_model = AttentionMILModel(input_dim=1024, hidden_units=256, dropout=0.3)
mil_weights = "models/mil_final.pt"
if Path(mil_weights).exists():
    mil_model.load_state_dict(torch.load(mil_weights, map_location=device, weights_only=True))
mil_model.eval()
crowd_analyzer = CrowdAnalyzer()
print("[Init] Ready.")

SAMPLE_EVERY = 30  # process every 30th frame (~1 fps)


def evaluate_video(video_path):
    cap = cv2.VideoCapture(str(video_path))
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total == 0:
        return None

    cap.release()
    cap = cv2.VideoCapture(str(video_path))
    crowd_analyzer.reset()

    frame_idx = 0
    scene_counts = {"normal": 0, "demo_damai": 0, "demo_rusuh": 0}
    fusion_counts = {"normal": 0, "demo_damai": 0, "demo_rusuh": 0}
    mil_scores = []
    mil_rusuh_votes = 0
    mil_total_votes = 0
    total_persons = 0
    sample_count = 0
    frames_buffer = []
    feat_buffer = []
    prev_gray = None

    _person_tracks = {}
    _next_pid = 0

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1
        if frame_idx % SAMPLE_EVERY != 0:
            continue

        resized = cv2.resize(frame, (640, 640))
        frames_buffer.append(resized)

        # Motion
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        prev_gray = gray

        # YOLO
        yolo_objects = detector.detect(frame, min_area=200)
        persons = yolo_objects.get("persons", [])
        n_p = len(persons)
        sample_count += 1
        total_persons += n_p

        # Person tracking
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
        for hist in _person_tracks.values():
            if hist["missed"] > 0:
                hist["velocities"].append(0)
            hist["missed"] += 1

        velocities = []
        for hist in _person_tracks.values():
            if hist["velocities"]:
                velocities.extend(hist["velocities"])
        running_count = sum(1 for v in velocities if v > 10.0) if velocities else 0
        running_ratio = running_count / max(len(velocities), 1)

        close_count = 0
        pair_count = 0
        positions = [(p["bbox"][0] + p["bbox"][2]) / 2 for p in persons]
        for i in range(len(positions)):
            for j in range(i + 1, len(positions)):
                if abs(positions[i] - positions[j]) < 50:
                    close_count += 1
                pair_count += 1
        close_pair_ratio = close_count / max(pair_count, 1)

        # CrowdAnalyzer
        chaos_value, _ = crowd_analyzer.compute_chaos_index(frame, n_p, running_ratio, close_pair_ratio)
        ca_scene = crowd_analyzer.classify_scene(n_p, chaos_value, running_ratio)
        scene_counts[ca_scene] = scene_counts.get(ca_scene, 0) + 1

        # MIL model + adaptive fusion
        mil_score = 0.0
        if len(frames_buffer) >= 16:
            seg = extractor.extract(frames_buffer[-16:])
            feat_buffer.append(seg.squeeze())
            min_seg = min(16, len(feat_buffer))
            if min_seg >= 4:
                bag = torch.FloatTensor(np.array(feat_buffer[-min_seg:])).unsqueeze(0)
                with torch.no_grad():
                    logits = mil_model(bag)
                    mil_score = torch.sigmoid(logits).item()
                mil_scores.append(mil_score)
                mil_total_votes += 1
                if mil_score >= 0.5:
                    mil_rusuh_votes += 1

                # Adaptive fusion scene
                n_person = n_p
                if mil_score > 0.8:
                    fused = mil_score
                elif n_person < 3:
                    fused = 0.9 * mil_score + 0.1 * chaos_value
                else:
                    fused = 0.6 * mil_score + 0.4 * chaos_value
                if fused > 0.5:
                    fusion_scene = "demo_rusuh"
                elif mil_score > 0.3 or ca_scene == "demo_damai":
                    fusion_scene = "demo_damai"
                else:
                    fusion_scene = ca_scene
                fusion_counts[fusion_scene] = fusion_counts.get(fusion_scene, 0) + 1
            frames_buffer = frames_buffer[-8:]

    cap.release()

    total_samples = max(sum(scene_counts.values()), 1)
    dominant_ca = max(scene_counts, key=scene_counts.get)
    dominant_fusion = max(fusion_counts, key=fusion_counts.get) if fusion_counts else "normal"
    avg_mil = np.mean(mil_scores) if mil_scores else 0.0
    max_mil = np.max(mil_scores) if mil_scores else 0.0
    mil_pct_rusuh = (mil_rusuh_votes / max(mil_total_votes, 1)) * 100

    return {
        "file": os.path.basename(video_path),
        "frames": total,
        "samples": total_samples,
        "dominant": dominant_ca,
        "dominant_fusion": dominant_fusion,
        "%rusuh_ca": scene_counts.get("demo_rusuh", 0) / total_samples * 100,
        "%damai_ca": scene_counts.get("demo_damai", 0) / total_samples * 100,
        "%normal_ca": scene_counts.get("normal", 0) / total_samples * 100,
        "avg_mil": avg_mil,
        "max_mil": max_mil,
        "%mil_rusuh": mil_pct_rusuh,
        "avg_persons": total_persons / max(sample_count, 1),
    }


def main():
    if "--video" in sys.argv:
        idx = sys.argv.index("--video")
        vpath = sys.argv[idx + 1]
        r = evaluate_video(vpath)
        if r is None:
            print(f"Error: cannot open {vpath}")
            return
        results = [("video", r)]
    else:
        is_full = "--full" in sys.argv
        base = "sample_videos/indonesia_v7"
        all_rusuh = sorted(Path(f"{base}/demo_rusuh").glob("*.mp4"))
        all_damai = sorted(Path(f"{base}/demo_damai").glob("*.mp4"))
        categories = {
            "demo_rusuh": all_rusuh if is_full else all_rusuh[:2],
            "demo_damai": all_damai if is_full else all_damai[:2],
        }
        if not is_full:
            print("[Test] QUICK mode. Use --full untuk semua video")

        results = []
        for cat_name, files in categories.items():
            for fpath in files:
                r = evaluate_video(str(fpath))
                if r:
                    results.append((cat_name, r))

    # Print results
    HDR = (f"{'Kategori':12s} | {'Video':32s} | {'CA-Dom':>6s} | {'Fusion':>6s} | "
           f"{'%Rusuh':>6s} | {'AvgMIL':>6s} | {'MaxMIL':>6s} | "
           f"{'%MIL-R':>6s} | {'AvgP':>4s}")
    print(f"\n{'='*120}")
    print(HDR)
    print('-' * len(HDR))

    for cat_name, r in results:
        fusion_ok = (
            (cat_name == "demo_rusuh" and r["dominant_fusion"] == "demo_rusuh")
            or (cat_name == "demo_damai" and r["dominant_fusion"] in ("demo_damai", "normal"))
            or (cat_name == "video")
        )
        mil_flag = " ⚠" if (fusion_ok and cat_name == "demo_damai" and r["%mil_rusuh"] > 30) else ""
        ca_flag = " 🏴" if (cat_name == "demo_damai" and r["%rusuh_ca"] > 20) else ""
        verdict = "✅" if fusion_ok else "❌"
        print(f"{cat_name:12s} | {r['file']:32s} | {r['dominant']:6s} | {r['dominant_fusion']:6s} | "
              f"{r['%rusuh_ca']:5.1f}% | "
              f"{r['avg_mil']:.4f} | {r['max_mil']:.4f} | {r['%mil_rusuh']:5.1f}% | "
              f"{r['avg_persons']:3.1f} {verdict}{ca_flag}{mil_flag}")

    print(f"\n{'='*120}")
    print("LEGENDA:")
    print("  CA-Dom = klasifikasi CrowdAnalyzer (rule-based)")
    print("  Fusion = adaptive MIL+CA (MIL primary, CA fallback)")
    print("  MIL score > 0.5 = terindikasi kerusuhan oleh MIL model")
    print("  %MIL-R = persentase frame MIL yang score > 0.5")
    print("  🏴 = CA false positive (damai terdeteksi rusuh oleh CA)")
    print("  ⚠ = MIL false positive (damai terdeteksi rusuh oleh MIL)")


if __name__ == "__main__":
    main()
