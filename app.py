import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
import tempfile

from inference import AnomalyDetector
from utils.config_loader import load_config

st.set_page_config(
    page_title="Sistem Deteksi Kerusuhan",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 Sistem Deteksi Kerusuhan & Anomali")
st.markdown("---")

config = load_config()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Status", "🟢 Active")
mode = config["general"]["mode"].upper()
col2.metric("Mode", "🎬 " + ("Simulasi" if mode == "FILE" else "CCTV"))
col3.metric("Threshold Alert", config["thresholds"]["alert"])
col4.metric("Threshold Warning", config["thresholds"]["warning"])

st.sidebar.header("Konfigurasi")

video_source = st.sidebar.radio("Sumber Video", ["Sample Videos", "Upload File"])

video_path = None
if video_source == "Sample Videos":
    video_files = list(Path("sample_videos").glob("*.mp4"))
    video_names = [f.name for f in video_files] if video_files else ["Tidak ada video"]
    selected = st.sidebar.selectbox("Pilih Video", video_names)
    video_path = str(Path("sample_videos") / selected) if video_files else None
else:
    uploaded = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
    if uploaded:
        tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
        tfile.write(uploaded.read())
        video_path = tfile.name

threshold_override = st.sidebar.slider(
    "Alert Threshold (override)", 0.5, 0.95, config["thresholds"]["alert"], 0.05
)

main_col, info_col = st.columns([3, 1])

alert_history = []

with main_col:
    st.subheader("📹 Live Detection Feed")
    feed_placeholder = st.empty()
    video_placeholder = st.empty()

    detect_btn = st.button("▶️ Mulai Deteksi", type="primary", use_container_width=True)

    if detect_btn and video_path and Path(video_path).exists():
        try:
            detector = AnomalyDetector("config.yaml")
            detector.config["thresholds"]["alert"] = threshold_override

            cap = cv2.VideoCapture(video_path)
            frames_buffer = []
            score_buffer = deque(maxlen=10)

            progress = st.progress(0)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

            frame_count = 0
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                resized = cv2.resize(frame, (448, 448))
                frames_buffer.append(resized)
                frame_count += 1

                progress.progress(min(frame_count / total_frames, 1.0))

                if len(frames_buffer) >= 16:
                    mil_features = detector.extractor.extract(frames_buffer[-16:])
                    mil_score = detector.mil.predict_bag(mil_features)
                    yolo_objects = detector.yolo.detect(resized)

                    faces = detector.yolo.detect_faces(resized)
                    if faces:
                        yolo_objects["faces"] = faces
                    plate = detector.yolo.detect_plate(resized)
                    yolo_objects["plate"] = plate

                    fused_score = detector._compute_fused_score(mil_score, yolo_objects)
                    score_buffer.append(fused_score)

                    display = resized.copy()
                    for obj_type, items in yolo_objects.items():
                        if obj_type == "plate":
                            continue
                        for item in items:
                            x1, y1, x2, y2 = map(int, item["bbox"])
                            conf = item["confidence"]
                            label = f"{obj_type} {conf:.0%}"
                            color = (0, 255, 0) if obj_type == "persons" else (0, 0, 255)
                            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(display, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    color = (0, 255, 0) if fused_score < 0.6 else (0, 255, 255) if fused_score < 0.8 else (0, 0, 255)
                    cv2.putText(display, f"Anomaly Score: {fused_score:.2f}", (10, 30),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

                    if fused_score >= threshold_override and len(score_buffer) >= 10 and all(s >= threshold_override for s in score_buffer):
                        cv2.putText(display, "!!! ANOMALI TERDETEKSI !!!", (50, 400),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)
                        alert_history.append({
                            "time": datetime.now().strftime("%H:%M:%S"),
                            "score": fused_score,
                            "objects": len(yolo_objects.get("weapons", [])) + len(yolo_objects.get("persons", [])),
                        })

                    display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(display_rgb, channels="RGB", use_container_width=True)

            cap.release()
            progress.progress(1.0)
            st.success(f"✅ Selesai: {frame_count} frame diproses")
            st.balloons()

        except Exception as e:
            st.error(f"Error: {e}")

with info_col:
    st.subheader("🚨 Riwayat Alert")
    if alert_history:
        for a in reversed(alert_history[-10:]):
            st.warning(f"**{a['time']}** - Score: {a['score']:.2f}")
    else:
        st.info("Belum ada alert")

    st.subheader("📍 Lokasi")
    st.caption(config["location"]["address"])
    st.markdown(f"[🗺️ Buka Google Maps]({config['location']['maps_link']})")

    st.subheader("⚙️ Info Sistem")
    st.caption(f"Waktu: {datetime.now().strftime('%d %B %Y %H:%M:%S')}")
    st.caption(f"Device: {config['general']['device'].upper()}")
    st.caption(f"Model: S3D + YOLOv11 + MIL")
    st.caption(f"Alert via: Telegram & WhatsApp")

st.markdown("---")
st.caption(
    "🚨 **Sistem Deteksi Kerusuhan & Anomali** | "
    "YOLOv11 + S3D + MIL Ranking | "
    "Laporan dikirim ke Telegram & WhatsApp | "
    "© 2026"
)
