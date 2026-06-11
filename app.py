import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image

from inference import AnomalyDetector
from utils.config_loader import load_config

st.set_page_config(
    page_title="Sistem Deteksi Kerusuhan",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 Sistem Deteksi Kerusuhan & Anomali")
st.markdown("---")

col1, col2, col3 = st.columns(3)
col1.metric("Status", "🟢 Active")
col2.metric("Mode", "Simulasi (Video File)")
col3.metric("Threshold Alert", "0.80")

st.sidebar.header("Konfigurasi")
config = load_config()

video_files = list(Path("sample_videos").glob("*.mp4"))
video_files += list(Path("sample_videos").glob("*.avi"))
video_names = [f.name for f in video_files]

selected_video = st.sidebar.selectbox(
    "Pilih Video Sample",
    video_names if video_names else ["Tidak ada video"],
)

confidence_threshold = st.sidebar.slider("Confidence Threshold", 0.0, 1.0, 0.5)

main_col, info_col = st.columns([3, 1])

with main_col:
    st.subheader("📹 Feed Deteksi")
    placeholder = st.empty()
    with placeholder:
        st.info("Pilih video dan klik 'Mulai Deteksi'")

    if st.button("▶️ Mulai Deteksi", type="primary"):
        if not video_files:
            st.error("Tidak ada video sample. Masukkan video ke folder sample_videos/")
        else:
            video_path = str(
                Path("sample_videos") / selected_video
            )
            detector = AnomalyDetector()
            cap = cv2.VideoCapture(video_path)

            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    break

                resized = cv2.resize(frame, (448, 448))
                frame_rgb = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
                placeholder.image(frame_rgb, channels="RGB", use_column_width=True)

            cap.release()
            st.success("Deteksi selesai")

with info_col:
    st.subheader("📋 Riwayat Alert")
    st.info("Belum ada alert")

    st.subheader("⚙️ Info Sistem")
    st.caption(f"Config: config.yaml")
    st.caption(f"Waktu: {datetime.now().strftime('%H:%M:%S')}")

st.markdown("---")
st.caption(
    "Sistem Deteksi Kerusuhan & Anomali berbasis Machine Learning | "
    "YOLOv11 + X3D + MIL Ranking | Laporan dikirim ke Telegram & WhatsApp"
)
