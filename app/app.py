"""
Streamlit App - Deteksi Kerusuhan Menggunakan AttentionMIL
UAS Machine Learning - Universitas Dian Nuswantoro
"""

import sys, os, json, random, warnings
warnings.filterwarnings("ignore")
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
import plotly.graph_objects as go
from pathlib import Path
from collections import Counter
from PIL import Image

import torch
from sklearn.metrics import roc_auc_score, roc_curve, confusion_matrix, classification_report
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from core.mil_attention import AttentionMILModel

sns.set_style("whitegrid")

# ---- CONFIG ----
st.set_page_config(
    page_title="Deteksi Kerusuhan - AttentionMIL",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

MODEL_PATH = "models/mil_final.pt"
META_PATH = "features/final_dataset/metadata.json"
FEAT_DIR = Path("features/final_dataset")
EVAL_DIR = Path("reports/evaluation")
INTERP_DIR = Path("reports/interpretation")
EDA_DIR = Path("reports/eda")
DEVICE = "cpu"
LABEL_NAMES = {0: "Normal/Damai", 1: "Rusuh"}


# ---- CACHE ----
@st.cache_resource
def load_model():
    model = AttentionMILModel(input_dim=1024, hidden_units=256, dropout=0.3)
    model.load_state_dict(torch.load(MODEL_PATH, map_location=DEVICE, weights_only=True))
    model.eval()
    return model


@st.cache_data
def load_metadata():
    with open(META_PATH) as f:
        meta = json.load(f)
    BASE = Path.cwd().resolve()
    for m in meta:
        raw = m["path"]
        raw_norm = raw.replace("\\", "/")
        if ":" in raw_norm.split("/")[0]:
            parts = raw_norm.split("/")
            try:
                idx = next(i for i, p in enumerate(parts) if p == "features")
                m["path"] = "/".join(parts[idx:])
            except StopIteration:
                m["path"] = parts[-1]
        else:
            p = Path(raw_norm)
            if p.is_absolute():
                try:
                    rel = p.relative_to(BASE)
                    m["path"] = str(rel)
                except ValueError:
                    m["path"] = p.name
    df = pd.DataFrame(meta)
    df["label_display"] = df["label"].map(LABEL_NAMES)
    return df, meta


@st.cache_data
def load_eval_metrics():
    metrics_path = EVAL_DIR / "metrics.json"
    if metrics_path.exists():
        with open(metrics_path) as f:
            return json.load(f)
    return None


@st.cache_data
def get_test_features():
    """Cache test set features for fast demo."""
    _, meta = load_metadata()
    test_items = [m for m in meta if m["split"] == "test"]
    features, labels, paths = [], [], []
    for item in test_items[:200]:  # limit to 200 for speed
        try:
            feat = np.load(item["path"])
            features.append(feat[:16].mean(axis=0))
            labels.append(item["label"])
            paths.append(Path(item["path"]).name)
        except (FileNotFoundError, OSError):
            continue
    return np.array(features), np.array(labels), paths


def predict_features(model, features):
    """Predict on feature array."""
    features_t = torch.FloatTensor(features).unsqueeze(0)
    with torch.no_grad():
        logits = model(features_t)
    return torch.sigmoid(logits).item()


@st.cache_resource
def load_yolo():
    from ultralytics import YOLO
    return YOLO("yolov8n.pt")

@st.cache_resource
def load_s3d_extractor():
    from preprocessing.feature_extractor import TemporalFeatureExtractor
    return TemporalFeatureExtractor(architecture="s3d", device=DEVICE)

def extract_frames_cap(video_path, max_frames=150):
    import cv2
    cap = cv2.VideoCapture(video_path)
    frames = []
    while len(frames) < max_frames:
        ret, frame = cap.read()
        if not ret:
            break
        frames.append(frame)
    cap.release()
    return frames

def detect_persons_yolo(frame, model, conf=0.4):
    results = model(frame, conf=conf, verbose=False)[0]
    persons = []
    if results.boxes is not None:
        for box in results.boxes:
            if int(box.cls[0]) == 0:
                x1, y1, x2, y2 = map(int, box.xyxy[0])
                persons.append((x1, y1, x2, y2, float(box.conf[0])))
    return persons

def draw_person_boxes(frame, boxes):
    import cv2
    for x1, y1, x2, y2, conf in boxes:
        cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
        label = f"Person {conf:.2f}"
        cv2.putText(frame, label, (x1, y1 - 8),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    return frame

def predict_upload_video(video_path, model, s3d, yolo):
    import cv2
    cap = cv2.VideoCapture(video_path)
    orig_fps = cap.get(cv2.CAP_PROP_FPS)
    if orig_fps <= 0:
        orig_fps = 15
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / orig_fps if orig_fps else 0

    frames_orig = []
    while True:
        ret, f = cap.read()
        if not ret:
            break
        frames_orig.append(f)
        if len(frames_orig) >= 300:
            break
    cap.release()

    if len(frames_orig) < 16:
        return None, frames_orig, orig_fps, "Video terlalu pendek (< 16 frame)"

    sample_interval = max(1, round(orig_fps / 4))
    frames_4fps = frames_orig[::sample_interval]

    stride = 8
    segments = []
    for i in range(0, len(frames_4fps) - 16 + 1, stride):
        seg = frames_4fps[i:i + 16]
        segments.append(seg)
        if len(segments) >= 16:
            break

    if not segments:
        return None, frames_orig, orig_fps, "Tidak bisa membuat segmen"

    features = s3d.extract_batch(segments)
    n_seg = min(16, features.shape[0])
    feat_t = torch.FloatTensor(features[:n_seg]).unsqueeze(0)

    with torch.no_grad():
        logits = model(feat_t)
    score = torch.sigmoid(logits).item()

    yolo_max = min(len(frames_orig), 150)
    for idx in range(0, yolo_max, 2):
        persons = detect_persons_yolo(frames_orig[idx], yolo)
        frames_orig[idx] = draw_person_boxes(frames_orig[idx], persons)

    return score, frames_orig[:yolo_max], orig_fps, None


# ---- SIDEBAR ----
st.sidebar.title("☰ Menu Navigasi")
page = st.sidebar.radio(
    "Pilih Halaman",
    ["Beranda", "Exploratory Data Analysis", "Demo Model", "Evaluasi & Interpretasi", "Dokumentasi"],
)

st.sidebar.markdown("---")
st.sidebar.markdown("**UAS Machine Learning**")
st.sidebar.markdown("Universitas Dian Nuswantoro")
st.sidebar.markdown("---")
st.sidebar.markdown("**Anggota:**")
st.sidebar.markdown("- Faishal Rasyid Rusianto (A11.2024.15869)")

# ---- PAGES ----
if page == "Beranda":
    # ---- HOME ----
    st.title(" Deteksi Kerusuhan Menggunakan AttentionMIL")
    st.markdown("### UAS Machine Learning - Program Studi Teknik Informatika")
    st.markdown("---")

    col1, col2, col3, col4 = st.columns(4)
    metrics = load_eval_metrics()

    if metrics:
        col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
        col2.metric("AUC", f"{metrics['auc']:.4f}")
        col3.metric("F1 Score", f"{metrics['f1']:.4f}")
        col4.metric("MCC", f"{metrics['mcc']:.4f}")

    st.markdown("---")

    col_a, col_b = st.columns(2)
    with col_a:
        st.markdown("### 📊 Dataset Overview")
        df, _ = load_metadata()
        st.markdown(f"- **Total Video:** {len(df):,}")
        st.markdown(f"- **Kelas:** 3 (demo_rusuh, demo_damai, normal)")
        st.markdown(f"- **Split:** Train {df[df['split']=='train'].shape[0]:,} | "
                    f"Val {df[df['split']=='val'].shape[0]:,} | "
                    f"Test {df[df['split']=='test'].shape[0]:,}")
        st.markdown(f"- **Sumber:** YouTube, Kaggle, SCVD, MSV-PG")
        st.markdown(f"- **Fitur:** S3D 1024-dimensional")

    with col_b:
        st.markdown("### 🤖 Model Architecture")
        st.markdown("- **Model:** Attention-based MIL")
        st.markdown("- **Input:** 16 segments × 1024-d S3D features")
        st.markdown("- **Attention:** Learnable segment weighting")
        st.markdown("- **Classifier:** 2-layer MLP (256→128→1)")
        st.markdown("- **Dropout:** 0.3")

    st.markdown("---")
    st.markdown("### 🎯 Tujuan")
    st.markdown(
        "Mendeteksi kerusuhan dari video CCTV/video amatir menggunakan "
        "pendekatan Multiple Instance Learning dengan attention mechanism. "
        "Model mampu mengidentifikasi segmen video yang mengandung aktivitas "
        "kerusuhan dan memberikan skor anomali secara real-time."
    )

    st.markdown("### 📈 Model Performance")
    col_p1, col_p2 = st.columns(2)
    with col_p1:
        roc_path = EVAL_DIR / "roc_curve.png"
        if roc_path.exists():
            st.image(str(roc_path), caption="ROC Curve")
    with col_p2:
        cm_path = EVAL_DIR / "confusion_matrix.png"
        if cm_path.exists():
            st.image(str(cm_path), caption="Confusion Matrix")

    st.markdown("### 🔍 Attention Weights")
    attn_path = INTERP_DIR / "attention_weights.png"
    if attn_path.exists():
        st.image(str(attn_path), caption="Attention weights per segment for sample videos")


elif page == "Exploratory Data Analysis":
    # ---- EDA PAGE ----
    st.title("📊 Exploratory Data Analysis")
    df, meta = load_metadata()

    tab1, tab2, tab3, tab4, tab5 = st.tabs([
        "Label Distribution", "Source Analysis", "Split Analysis",
        "PCA Visualization", "t-SNE Visualization"
    ])

    with tab1:
        st.subheader("Distribusi Label")
        col1, col2 = st.columns(2)

        label_counts = df["label_name"].value_counts()
        colors = ["#2ecc71", "#e74c3c", "#3498db"]

        fig1, ax = plt.subplots(figsize=(8, 5))
        bars = ax.bar(label_counts.index, label_counts.values, color=colors, edgecolor="white")
        ax.set_xlabel("Kelas"); ax.set_ylabel("Jumlah Video")
        ax.set_title("Distribusi Label", fontweight="bold")
        for bar in bars:
            h = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2, h + 20, str(h), ha="center", fontweight="bold")
        col1.pyplot(fig1)

        fig2, ax = plt.subplots(figsize=(6, 6))
        ax.pie(label_counts.values, labels=label_counts.index, autopct="%1.1f%%",
               colors=colors, startangle=90, explode=[0.03]*3)
        ax.set_title("Proporsi Label", fontweight="bold")
        col2.pyplot(fig2)

        st.dataframe(label_counts.reset_index().rename(
            columns={"index": "Label", "label_name": "Jumlah"}))

    with tab2:
        st.subheader("Distribusi Sumber Data")
        src_counts = df["source"].value_counts()
        fig, ax = plt.subplots(figsize=(10, 5))
        bars = ax.barh(src_counts.index, src_counts.values,
                       color=sns.color_palette("Set2"), edgecolor="white")
        ax.set_xlabel("Jumlah Video"); ax.set_ylabel("Sumber Data")
        ax.set_title("Distribusi Sumber Data", fontweight="bold")
        for bar in bars:
            w = bar.get_width()
            ax.text(w + 10, bar.get_y() + bar.get_height()/2, str(w), ha="left", va="center")
        st.pyplot(fig)

        # Source per label
        st.subheader("Sumber Data per Label")
        ct_src = pd.crosstab(df["source"], df["label_name"])
        fig, ax = plt.subplots(figsize=(10, 6))
        ct_src.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
        ax.set_xlabel("Source"); ax.set_ylabel("Count")
        ax.set_title("Source per Label", fontweight="bold")
        ax.legend(title="Label")
        st.pyplot(fig)

    with tab3:
        st.subheader("Distribusi Train/Val/Test")
        split_counts = df["split"].value_counts()
        split_colors = {"train": "#2ecc71", "val": "#f39c12", "test": "#e74c3c"}

        col1, col2 = st.columns(2)
        fig1, ax = plt.subplots(figsize=(8, 5))
        ax.bar(split_counts.index, split_counts.values,
               color=[split_colors[s] for s in split_counts.index], edgecolor="white")
        ax.set_xlabel("Split"); ax.set_ylabel("Jumlah Video")
        ax.set_title("Distribusi Split", fontweight="bold")
        for i, v in enumerate(split_counts.values):
            ax.text(i, v + 20, str(v), ha="center", fontweight="bold")
        col1.pyplot(fig1)

        fig2, ax = plt.subplots(figsize=(6, 6))
        ax.pie(split_counts.values, labels=split_counts.index,
               autopct="%1.1f%%",
               colors=[split_colors[s] for s in split_counts.index],
               startangle=90, explode=[0.03]*3)
        ax.set_title("Proporsi Split", fontweight="bold")
        col2.pyplot(fig2)

        # Label per split
        st.subheader("Label Distribution per Split")
        ct = pd.crosstab(df["split"], df["label_name"])
        fig, ax = plt.subplots(figsize=(8, 5))
        ct.plot(kind="bar", ax=ax, color=colors, edgecolor="white")
        ax.set_xlabel("Split"); ax.set_ylabel("Count")
        ax.set_title("Label per Split", fontweight="bold")
        ax.legend(title="Label")
        ax.set_xticklabels(ax.get_xticklabels(), rotation=0)
        st.pyplot(fig)

    with tab4:
        st.subheader("PCA - Feature Space Visualization")
        st.info("Menampilkan 2.000 sample acak dari dataset dengan PCA 2 komponen.")

        n_samples = min(2000, len(df))
        np.random.seed(42)
        sample_idx = np.random.choice(len(df), n_samples, replace=False)
        sample_df = df.iloc[sample_idx]

        features_pca = []
        labels_pca = []
        progress = st.progress(0)
        for i, (_, row) in enumerate(sample_df.iterrows()):
            try:
                feat = np.load(row["path"])
                features_pca.append(feat.mean(axis=0))
                labels_pca.append(row["label"])
            except (FileNotFoundError, OSError):
                pass
            progress.progress((i + 1) / n_samples)

        if not features_pca:
            st.warning("Tidak ada file fitur yang tersedia untuk PCA.")
            st.stop()
        features_pca = np.array(features_pca)
        labels_pca = np.array(labels_pca)

        pca = PCA(n_components=2)
        coords = pca.fit_transform(features_pca)

        fig = px.scatter(
            x=coords[:, 0], y=coords[:, 1],
            color=labels_pca, color_continuous_scale="RdYlGn",
            title=f"PCA (explained variance: {pca.explained_variance_ratio_.sum():.2%})",
            labels={"x": "PC1", "y": "PC2", "color": "Label"},
        )
        st.plotly_chart(fig, width='stretch')

    with tab5:
        st.subheader("t-SNE - Feature Space Visualization")
        st.info("Menampilkan 1.000 sample dengan t-SNE (perplexity=30).")

        n_tsne = min(1000, len(df))
        np.random.seed(42)
        tsne_idx = np.random.choice(len(df), n_tsne, replace=False)
        tsne_df = df.iloc[tsne_idx]

        features_tsne = []
        labels_tsne = []
        progress = st.progress(0)
        for i, (_, row) in enumerate(tsne_df.iterrows()):
            try:
                feat = np.load(row["path"])
                features_tsne.append(feat.mean(axis=0))
                labels_tsne.append(row["label"])
            except (FileNotFoundError, OSError):
                pass
            progress.progress((i + 1) / n_tsne)

        if not features_tsne:
            st.warning("Tidak ada file fitur yang tersedia untuk t-SNE.")
            st.stop()
        features_tsne = np.array(features_tsne)
        labels_tsne = np.array(labels_tsne)

        tsne = TSNE(n_components=2, random_state=42, perplexity=30)
        coords_tsne = tsne.fit_transform(features_tsne)

        fig = px.scatter(
            x=coords_tsne[:, 0], y=coords_tsne[:, 1],
            color=labels_tsne, color_continuous_scale="RdYlGn",
            title="t-SNE Visualization",
            labels={"x": "t-SNE 1", "y": "t-SNE 2", "color": "Label"},
        )
        st.plotly_chart(fig, width='stretch')


elif page == "Demo Model":
    # ---- DEMO PAGE ----
    st.title("🧪 Demo Model - Deteksi Kerusuhan")

    model = load_model()

    tab_video, tab_feature, tab_batch, tab_upload, tab_webcam = st.tabs([
        "🎬 Video Demo (Asli)", "📊 Feature Demo", "📈 Batch Test Set",
        "📤 Upload Video", "🎥 CCTV Live"
    ])

    # ===== VIDEO DEMO TAB (with actual video playback) =====
    with tab_video:
        st.markdown("### Demo dengan Video Asli")
        st.markdown(
            "Video diputar langsung, fitur S3D sudah diekstrak sebelumnya. "
            "Model memprediksi dalam **< 1 detik**."
        )

        DEMO_META_PATH = Path("features/demo_videos/demo_metadata.json")
        VIDEO_DIR = Path("test_videos")

        if DEMO_META_PATH.exists():
            with open(DEMO_META_PATH) as f:
                demo_meta = json.load(f)

            # Split by label
            normal_demos = [m for m in demo_meta if m["label"] == 0]
            rusuh_demos = [m for m in demo_meta if m["label"] == 1]

            if normal_demos:
                demo_type = st.radio("Pilih tipe video:", ["Rusuh (dengan video)", "Normal/Damai (fitur)"], horizontal=True, key="demo_type")
                if demo_type == "Normal/Damai (fitur)":
                    demo_pool = normal_demos
                else:
                    demo_pool = rusuh_demos
            else:
                st.info("Untuk video **Normal/Damai**, gunakan tab **Feature Demo** di atas.")
                demo_pool = rusuh_demos

            if demo_pool:
                selected_demo = st.selectbox(
                    "Pilih video:",
                    demo_pool,
                    format_func=lambda x: x["display_name"],
                    key="demo_video_select",
                )

                col_vid, col_pred = st.columns([1, 1.2])

                with col_vid:
                    video_path = VIDEO_DIR / selected_demo["video_file"]
                    show_bbox = st.checkbox("Tampilkan Bounding Box (YOLO)", key="bbox_demo")

                    if show_bbox and video_path.exists():
                        yolo = load_yolo()
                        cap = cv2.VideoCapture(str(video_path))
                        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                        total = min(total, 150)
                        frames_bb = []
                        bprog = st.progress(0)
                        for fi in range(total):
                            ret, fr = cap.read()
                            if not ret:
                                break
                            persons = detect_persons_yolo(fr, yolo)
                            fr = draw_person_boxes(fr, persons)
                            frames_bb.append(fr)
                            bprog.progress((fi + 1) / total)
                        cap.release()

                        if frames_bb:
                            out_p = str(video_path).replace(".mp4", "_bbox.mp4")
                            h, w = frames_bb[0].shape[:2]
                            out = cv2.VideoWriter(out_p, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 15, (w, h))
                            for f in frames_bb:
                                out.write(f)
                            out.release()
                            st.video(out_p)
                            os.unlink(out_p)
                        else:
                            st.video(str(video_path))
                    elif video_path.exists():
                        st.video(str(video_path))
                    else:
                        st.warning(f"Video file tidak ditemukan: {video_path}")

                    st.caption(f"Sumber: `{selected_demo['video_file']}` | "
                               f"Segmen: {selected_demo['segments']}")

                with col_pred:
                    if st.button("🔍 Predict Video Ini", type="primary", width='stretch'):
                        feat_path = Path(selected_demo["path"])
                        if not feat_path.exists():
                            st.error(f"File fitur tidak ditemukan: {feat_path.name}. Gunakan video lain.")
                            st.stop()
                        feat = np.load(str(feat_path))
                        n_seg = min(16, feat.shape[0])
                        feat_t = torch.FloatTensor(feat[:n_seg]).unsqueeze(0)

                        with torch.no_grad():
                            logits = model(feat_t)
                        score = torch.sigmoid(logits).item()

                        pred_label = "🔴 RUSUH" if score >= 0.5 else "🟢 NORMAL/DAMAI"
                        confidence = max(score, 1 - score)
                        true_label = "RUSUH" if selected_demo["label"] == 1 else "NORMAL/DAMAI"
                        correct = (score >= 0.5) == (selected_demo["label"] == 1)

                        fig = go.Figure(go.Indicator(
                            mode="gauge+number+delta",
                            value=score,
                            title={"text": "Anomaly Score", "font": {"size": 18}},
                            delta={"reference": 0.5, "increasing": {"color": "red"},
                                   "decreasing": {"color": "green"}},
                            gauge={
                                "axis": {"range": [0, 1]},
                                "bar": {"color": "red" if score >= 0.5 else "green"},
                                "steps": [
                                    {"range": [0, 0.5], "color": "lightgreen"},
                                    {"range": [0.5, 1], "color": "lightcoral"},
                                ],
                                "threshold": {
                                    "line": {"color": "black", "width": 4},
                                    "thickness": 0.75, "value": 0.5
                                }
                            }
                        ))
                        fig.update_layout(height=250)
                        st.plotly_chart(fig, width='stretch')

                        col1, col2 = st.columns(2)
                        col1.metric("Prediction", pred_label)
                        col2.metric("Confidence", f"{confidence:.2%}")
                        col3, col4 = st.columns(2)
                        col3.metric("True Label", true_label)
                        col4.metric("Status", "✅ Correct" if correct else "❌ Wrong",
                                    delta_color="off")

                        # Segment scores
                        if selected_demo["segments"] > 1:
                            seg_scores = []
                            for i in range(n_seg):
                                f_t = torch.FloatTensor(feat[i:i+1]).unsqueeze(0)
                                with torch.no_grad():
                                    s = torch.sigmoid(model(f_t)).item()
                                seg_scores.append(s)

                            st.subheader("Segment Scores")
                            fig2, ax2 = plt.subplots(figsize=(8, 2.2))
                            colors = ["#e74c3c" if s >= 0.5 else "#2ecc71" for s in seg_scores]
                            ax2.bar(range(len(seg_scores)), seg_scores, color=colors, alpha=0.85)
                            ax2.axhline(0.5, color="gray", ls="--")
                            ax2.set_ylim(0, 1); ax2.set_xlabel("Segment"); ax2.set_ylabel("Score")
                            st.pyplot(fig2)
                    else:
                        st.info("👈 Klik 'Predict Video Ini' untuk melihat hasil prediksi")
            else:
                st.warning("Tidak ada video demo untuk kelas ini.")
        else:
            st.warning(
                "Video demo belum tersedia. Jalankan `preprocessing/extract_demo_videos.py` "
                "untuk mengekstrak fitur video demo."
            )

    # ===== FEATURE DEMO TAB (559 test items) =====
    with tab_feature:
        st.markdown("### Demo dengan 559 Video Test")
        st.markdown(
            "Pilih sample dari dataset (fitur S3D yang sudah diekstrak). "
            "Model memproses dan menampilkan hasil prediksi."
        )

        df, meta = load_metadata()
        test_items = [m for m in meta if m["split"] == "test"]

        normal_items = [m for m in test_items if m["label"] == 0]
        rusuh_items = [m for m in test_items if m["label"] == 1]

        ftype = st.radio("Pilih tipe:", ["Normal/Damai", "Rusuh"], horizontal=True, key="ftype")
        fpool = normal_items if ftype == "Normal/Damai" else rusuh_items

        selected_item = st.selectbox(
            "Pilih sample:",
            fpool,
            format_func=lambda x: Path(x["path"]).stem,
            key="feat_select",
        )

        if st.button("🔍 Predict", type="primary", width='stretch'):
            try:
                feat = np.load(selected_item["path"])
            except (FileNotFoundError, OSError):
                st.error(f"File fitur tidak ditemukan: {selected_item['path']}")
                st.stop()
            n_seg = min(16, feat.shape[0])
            feat_t = torch.FloatTensor(feat[:n_seg]).unsqueeze(0)

            with torch.no_grad():
                logits = model(feat_t)
            score = torch.sigmoid(logits).item()
            pred_label = "🔴 RUSUH" if score >= 0.5 else "🟢 NORMAL/DAMAI"
            confidence = max(score, 1 - score)
            true_label = "RUSUH" if selected_item["label"] == 1 else "NORMAL/DAMAI"
            correct = (score >= 0.5) == (selected_item["label"] == 1)

            st.markdown("---")

            col_g, col_r = st.columns([1, 1])
            with col_g:
                fig = go.Figure(go.Indicator(
                    mode="gauge+number+delta", value=score,
                    title={"text": "Anomaly Score", "font": {"size": 18}},
                    delta={"reference": 0.5, "increasing": {"color": "red"},
                           "decreasing": {"color": "green"}},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "red" if score >= 0.5 else "green"},
                        "steps": [{"range": [0, 0.5], "color": "lightgreen"},
                                  {"range": [0.5, 1], "color": "lightcoral"}],
                        "threshold": {"line": {"color": "black", "width": 4},
                                      "thickness": 0.75, "value": 0.5}
                    }
                ))
                fig.update_layout(height=250)
                st.plotly_chart(fig, width='stretch')

            with col_r:
                st.markdown("### Hasil")
                c1, c2 = st.columns(2)
                c1.metric("Prediction", pred_label)
                c2.metric("Confidence", f"{confidence:.2%}")
                c3, c4 = st.columns(2)
                c3.metric("True Label", true_label)
                c4.metric("Status", "✅ Correct" if correct else "❌ Wrong")
                st.markdown(f"- Sumber: {selected_item.get('source', 'unknown')}")
                st.markdown(f"- Segmen: {n_seg}")

            # Segment scores
            feat_all = feat
            n_all = min(16, feat_all.shape[0])
            seg_scores = []
            for i in range(n_all):
                f_t = torch.FloatTensor(feat_all[i:i+1]).unsqueeze(0)
                with torch.no_grad():
                    s = torch.sigmoid(model(f_t)).item()
                seg_scores.append(s)

            st.subheader("Segment-Level Scores")
            fig, ax = plt.subplots(figsize=(10, 3))
            colors = ["#e74c3c" if s >= 0.5 else "#2ecc71" for s in seg_scores]
            ax.bar(range(n_all), seg_scores, color=colors, alpha=0.85, edgecolor="white")
            ax.axhline(0.5, color="gray", ls="--", lw=2, label="Threshold = 0.5")
            ax.set_xlabel("Segment Index"); ax.set_ylabel("Anomaly Score")
            ax.set_title("Per-Segment Anomaly Scores", fontweight="bold")
            ax.legend(); ax.set_ylim(0, 1); ax.set_xticks(range(n_all))
            st.pyplot(fig)

    with tab_batch:
        st.markdown("### Test Set Batch Evaluation")
        st.markdown("Mengevaluasi model pada seluruh test set (559 video).")

        if st.button("▶️ Run Batch Evaluation", type="primary"):
            progress = st.progress(0)
            status = st.empty()

            test_items_batch = [m for m in meta if m["split"] == "test"]
            y_true, y_score_batch = [], []

            for i, item in enumerate(test_items_batch):
                feat = np.load(item["path"])
                n_seg = min(16, feat.shape[0])
                feat_t = torch.FloatTensor(feat[:n_seg]).unsqueeze(0)
                with torch.no_grad():
                    s = torch.sigmoid(model(feat_t)).item()
                y_true.append(item["label"])
                y_score_batch.append(s)
                progress.progress((i + 1) / len(test_items_batch))
                status.text(f"Processing {i+1}/{len(test_items_batch)}")

            y_true = np.array(y_true)
            y_score_batch = np.array(y_score_batch)
            y_pred = (y_score_batch >= 0.5).astype(int)

            auc = roc_auc_score(y_true, y_score_batch)
            cm = confusion_matrix(y_true, y_pred)
            acc = np.mean(y_true == y_pred)

            col_m1, col_m2, col_m3, col_m4 = st.columns(4)
            col_m1.metric("Accuracy", f"{acc:.2%}")
            col_m2.metric("AUC", f"{auc:.4f}")
            col_m3.metric("Normal Correct", f"{cm[0,0]}/{cm[0,0]+cm[0,1]}")
            col_m4.metric("Rusuh Correct", f"{cm[1,1]}/{cm[1,0]+cm[1,1]}")

            fig, axes = plt.subplots(1, 2, figsize=(12, 4.5))
            fpr, tpr, _ = roc_curve(y_true, y_score_batch)
            axes[0].plot(fpr, tpr, "b-", lw=2.5, label=f"AUC = {auc:.4f}")
            axes[0].plot([0, 1], [0, 1], "k--", alpha=0.5)
            axes[0].fill_between(fpr, tpr, alpha=0.1, color="blue")
            axes[0].set_xlabel("FPR"); axes[0].set_ylabel("TPR")
            axes[0].set_title("ROC Curve"); axes[0].legend()

            sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                        xticklabels=["Normal", "Rusuh"],
                        yticklabels=["Normal", "Rusuh"], ax=axes[1])
            axes[1].set_xlabel("Predicted"); axes[1].set_ylabel("Actual")
            axes[1].set_title("Confusion Matrix")

            plt.tight_layout()
            st.pyplot(fig)

            st.markdown("**Classification Report:**")
            report = classification_report(y_true, y_pred,
                                           target_names=["Normal/Damai", "Rusuh"],
                                           output_dict=True, zero_division=0)
            st.dataframe(pd.DataFrame(report).transpose().round(4))

    # ===== UPLOAD & DETECT TAB =====
    with tab_upload:
        st.markdown("### Upload Video + Deteksi Kerusuhan")
        st.markdown(
            "Upload video Anda sendiri. Sistem akan mengekstrak fitur "
            "menggunakan **S3D**, memprediksi dengan **AttentionMIL**, "
            "dan menampilkan bounding box **YOLO** pada setiap frame."
        )

        uploaded_file = st.file_uploader(
            "Pilih video (mp4, avi, mov)",
            type=["mp4", "avi", "mov"],
            key="upload_video",
        )

        if uploaded_file is not None:
            from tempfile import NamedTemporaryFile

            tfile = NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded_file.read())
            tfile.close()

            import cv2
            cap = cv2.VideoCapture(tfile.name)
            fps = cap.get(cv2.CAP_PROP_FPS)
            total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            duration = total_frames / fps if fps else 0
            cap.release()

            st.markdown(f"**File:** {uploaded_file.name} | **Durasi:** {duration:.1f}s | **Frame:** {total_frames}")

            if st.button("▶️ Process Video", type="primary", width='stretch'):
                with st.spinner("Memuat model..."):
                    model_atn = load_model()
                    s3d = load_s3d_extractor()
                    yolo = load_yolo()

                prog = st.progress(0)
                status = st.empty()
                status.text("Membaca frame video...")

                score, frames_out, vid_fps, err = predict_upload_video(tfile.name, model_atn, s3d, yolo)

                prog.progress(100)
                status.text("Selesai!")
                os.unlink(tfile.name)

                if err:
                    st.error(err)
                    st.stop()

                pred_label = "🔴 RUSUH" if score >= 0.5 else "🟢 NORMAL/DAMAI"
                confidence = max(score, 1 - score)

                st.markdown("---")
                col_a, col_b, col_c, col_d = st.columns(4)
                col_a.metric("Anomaly Score", f"{score:.4f}")
                col_b.metric("Prediksi", pred_label)
                col_c.metric("Confidence", f"{confidence:.2%}")
                col_d.metric("Frame Diproses", str(len(frames_out)))

                fig_g = go.Figure(go.Indicator(
                    mode="gauge+number", value=score,
                    title={"text": "Anomaly Score", "font": {"size": 18}},
                    gauge={
                        "axis": {"range": [0, 1]},
                        "bar": {"color": "red" if score >= 0.5 else "green"},
                        "steps": [{"range": [0, 0.5], "color": "lightgreen"},
                                  {"range": [0.5, 1], "color": "lightcoral"}],
                        "threshold": {"line": {"color": "black", "width": 4},
                                      "thickness": 0.75, "value": 0.5}
                    }
                ))
                fig_g.update_layout(height=250)
                st.plotly_chart(fig_g, width='stretch')

                out_path = tfile.name.replace(".mp4", "_out.mp4")
                h, w = frames_out[0].shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                out_writer = cv2.VideoWriter(out_path, fourcc, 10, (w, h))
                for frm in frames_out:
                    out_writer.write(frm)
                out_writer.release()

                with open(out_path, "rb") as f:
                    st.download_button(
                        "⬇️ Download Video Hasil (YOLO Bounding Box)",
                        data=f,
                        file_name=f"hasil_{uploaded_file.name}",
                        mime="video/mp4",
                    )
                os.unlink(out_path)
        else:
            st.info("Silakan upload video untuk memulai deteksi.")

        st.markdown("---")
        st.caption(
            "Pipeline: Frame Extraction → S3D Feature Extraction → AttentionMIL Prediction → "
            "YOLOv8 Person Detection + Bounding Box."
        )

    # ===== WEBCAM TAB =====
    with tab_webcam:
        st.markdown("### CCTV - Deteksi Orang (YOLOv8)")
        st.markdown(
            "Deteksi orang secara **real-time** menggunakan YOLOv8. "
            "Ambil foto dari webcam untuk memeriksa jumlah orang dan status keamanan."
        )

        cont_mode = st.checkbox("Mode CCTV Otomatis (refresh tiap 3 detik)", key="cctv_auto")
        cam_img = st.camera_input("Ambil foto dari webcam", key="webcam_cctv", disabled=cont_mode)

        if cam_img is not None:
            yolo = load_yolo()
            bytes_data = cam_img.getvalue()
            arr = np.frombuffer(bytes_data, np.uint8)
            img = cv2.imdecode(arr, cv2.IMREAD_COLOR)

            persons = detect_persons_yolo(img, yolo)
            img = draw_person_boxes(img, persons)
            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

            col_wc1, col_wc2 = st.columns(2)
            with col_wc1:
                st.image(cam_img, caption="Foto Asli", width='stretch')
            with col_wc2:
                st.image(img_rgb, caption=f"Deteksi ({len(persons)} orang)", width='stretch')

            n = len(persons)
            if n == 0:
                wc_status = "🟢 AMAN - Tidak ada orang"
            elif n <= 2:
                wc_status = "🟡 WASPADA - Beberapa orang"
            elif n <= 5:
                wc_status = "🟠 SIAGA - Keramaian"
            else:
                wc_status = "🔴 RUSUH - Kerumunan besar"

            st.markdown(f"**Status:** {wc_status}")
            st.metric("Jumlah Orang Terdeteksi", n)

            if cont_mode:
                import time
                time.sleep(3)
                st.rerun()
        else:
            st.info("Klik 'Ambil foto' untuk mengaktifkan webcam.")


elif page == "Evaluasi & Interpretasi":
    # ---- EVALUATION PAGE ----
    st.title("📈 Evaluasi & Interpretasi Model")

    tab_eval, tab_interp, tab_about = st.tabs([
        "Model Evaluation", "Model Interpretation", "About Model"
    ])

    with tab_eval:
        st.subheader("Performance Metrics")

        metrics = load_eval_metrics()
        if metrics:
            col1, col2, col3 = st.columns(3)
            col1.metric("Accuracy", f"{metrics['accuracy']:.2%}")
            col1.metric("AUC", f"{metrics['auc']:.4f}")
            col2.metric("Precision", f"{metrics['precision']:.4f}")
            col2.metric("Recall", f"{metrics['recall']:.4f}")
            col3.metric("F1 Score", f"{metrics['f1']:.4f}")
            col3.metric("MCC", f"{metrics['mcc']:.4f}")

            st.markdown(f"- **Test Set:** {metrics['test_samples']} videos")
            st.markdown(f"- **False Positives:** {metrics['false_positives']} (Normal → Rusuh)")
            st.markdown(f"- **False Negatives:** {metrics['false_negatives']} (Rusuh → Normal)")

            st.markdown("---")

            col_img1, col_img2 = st.columns(2)
            with col_img1:
                roc_path = EVAL_DIR / "roc_curve.png"
                if roc_path.exists():
                    st.image(str(roc_path), caption="ROC Curve (AUC = {:.4f})".format(
                        metrics['auc']))
            with col_img2:
                cm_path = EVAL_DIR / "confusion_matrix.png"
                if cm_path.exists():
                    st.image(str(cm_path), caption="Confusion Matrix")

            col_img3, col_img4 = st.columns(2)
            with col_img3:
                pr_path = EVAL_DIR / "pr_curve.png"
                if pr_path.exists():
                    st.image(str(pr_path), caption="Precision-Recall Curve")
            with col_img4:
                sd_path = EVAL_DIR / "score_distribution.png"
                if sd_path.exists():
                    st.image(str(sd_path), caption="Score Distribution by Class")

            st.subheader("Classification Report")
            df_report = pd.DataFrame({
                "Metric": ["Accuracy", "AUC", "F1", "Precision", "Recall", "MCC"],
                "Value": [
                    f"{metrics['accuracy']:.2%}",
                    f"{metrics['auc']:.4f}",
                    f"{metrics['f1']:.4f}",
                    f"{metrics['precision']:.4f}",
                    f"{metrics['recall']:.4f}",
                    f"{metrics['mcc']:.4f}",
                ],
            })
            st.dataframe(df_report, width='stretch')

    with tab_interp:
        st.subheader("Model Interpretation")

        st.markdown("""
        **Attention-based MIL** memberikan interpretasi dengan cara:
        1. Setiap segmen video mendapat attention weight
        2. Segmen dengan weight tinggi → kontribusi besar ke prediksi akhir
        3. Feature ablation mengukur dampak tiap segmen terhadap skor
        """)

        col_i1, col_i2 = st.columns(2)
        with col_i1:
            attn_path = INTERP_DIR / "attention_weights.png"
            if attn_path.exists():
                st.image(str(attn_path), caption="Attention Weights per Segment",
                         width='stretch')
        with col_i2:
            abl_path = INTERP_DIR / "feature_ablation.png"
            if abl_path.exists():
                st.image(str(abl_path), caption="Feature Ablation Impact",
                         width='stretch')

        col_i3, col_i4 = st.columns(2)
        with col_i3:
            evo_path = INTERP_DIR / "per_video_evolution.png"
            if evo_path.exists():
                st.image(str(evo_path), caption="Score Evolution per Video",
                         width='stretch')
        with col_i4:
            conv_path = INTERP_DIR / "score_convergence.png"
            if conv_path.exists():
                st.image(str(conv_path), caption="Score Convergence by #Segments",
                         width='stretch')

        st.subheader("Key Insights")
        st.markdown("""
        - **Attention weights** menunjukkan model fokus ke segmen dengan gerakan abnormal
        - **Ablation analysis** mengkonfirmasi setiap segmen berkontribusi; segmen akhir lebih penting
        - **Score convergence** stabil setelah 8-10 segmen
        - **Score evolution** menunjukkan model butuh ~4-6 segmen untuk keputusan akurat
        """)

    with tab_about:
        st.subheader("Tentang Model")

        st.markdown("""
        ### AttentionMIL Model Architecture

        ```
        Input: 16 segments x 1024-d S3D features
               │
               ▼
        Attention Network
        ┌─────────────────────────┐
        │ Linear(1024 → 256)      │
        │ Tanh                     │
        │ Linear(256 → 1)          │
        └─────────────────────────┘
               │
               ▼
        Softmax → Attention Weights
               │
               ▼
        Weighted Bag Representation
               │
               ▼
        Classifier MLP
        ┌─────────────────────────┐
        │ Linear(1024 → 256)      │
        │ ReLU + Dropout(0.3)     │
        │ Linear(256 → 128)       │
        │ ReLU + Dropout(0.3)     │
        │ Linear(128 → 1)         │
        └─────────────────────────┘
               │
               ▼
        Sigmoid → Anomaly Score (0-1)
        ```

        **Training Details:**
        - Optimizer: Adam
        - Loss: Binary Cross-Entropy
        - Epochs: 50 (with early stopping)
        - Batch size: 32
        - Learning rate: 0.001
        - Data split: 80/10/10

        **Feature Extractor:** S3D (Separable 3D CNN) - pretrained on Kinetics-400
        """)

elif page == "Dokumentasi":
    st.title(" Dokumentasi Proyek")
    st.markdown("---")

    tab_dset, tab_metodologi, tab_usage, tab_ref = st.tabs([
        " Dataset", " Metodologi", " Cara Penggunaan", " Referensi"
    ])

    with tab_dset:
        st.header("Dataset")
        st.markdown("""
        Proyek ini menggunakan dataset video dari berbagai sumber untuk mendeteksi
        kerusuhan (perkelahian, tawuran, kerusuhan massa) vs aktivitas normal/damai.

        **Total Video:** 5.552
        **Kelas:** 3 kelas awal (demo_rusuh=kerusuhan, demo_damai=normal, normal=ucf)
        **Final:** 2 kelas biner — Rusuh (1) vs Normal/Damai (0)
        **Split:** Train 4.440 (80%) | Val 553 (10%) | Test 559 (10%)
        """)

        st.subheader("Sumber Data")
        col_s1, col_s2 = st.columns(2)
        with col_s1:
            st.markdown("""
            **1. UCF Crime (Kaggle)**
            - 1.900 video abnormal (perkelahian, perampokan, dll.)
            - 950 video normal (walking, driving, dll.)
            - Resolusi 320x240, durasi bervariasi

            **2. SCVD (Surveillance Camera Violence Dataset)**
            - 1.659 video kekerasan dari CCTV
            - Anotasi frame-level
            """)
        with col_s2:
            st.markdown("""
            **3. MSV-PG (Moderating Severe Violence - Playground)**
            - Dataset kekerasan dari playground CCTV
            - 600+ video pertengkaran

            **4. YouTube & Instagram (Scraped)**
            - Video tawuran Indonesia (CCTV, amatir)
            - Video normal/damai aktivitas sehari-hari
            - 500+ video dari berbagai kanal

            **5. Real Life Nonviolence**
            - 900+ video interaksi non-kekerasan
            - Sumber: dataset akademik
            """)

        st.subheader("Distribusi Fitur")
        st.markdown("""
        - **Feature Extractor:** S3D (Separable 3D CNN) pretrained on Kinetics-400
        - **Dimensi Fitur:** 1024-d per segmen video
        - **Segmentasi:** 16 frame per segmen (4 FPS x 4 detik)
        - **Stride:** 8 frame antar segmen (overlapping 50%)
        """)

        st.subheader("Preprocessing Pipeline")
        st.markdown("""
        1. **Frame Extraction:** Video dibaca dengan OpenCV, di-resize ke 224x224
        2. **Resampling:** Frame disampling ke 4 FPS
        3. **Segmentasi:** 16 frame berurutan = 1 segmen (~ 4 detik video)
        4. **Feature Extraction:** Setiap segmen -> S3D -> 1024-d feature vector
        5. **Normalization:** Mean=[0.485,0.456,0.406], Std=[0.229,0.224,0.225]
        6. **Label Mapping:** 3 kelas -> biner (Rusuh/Normal)
        7. **Split:** Stratified 80/10/10
        """)

    with tab_metodologi:
        st.header("Metodologi")

        st.subheader("Pipeline End-to-End")
        st.markdown("""
        **Langkah 1: Frame Extraction**
        - Video dibaca frame per frame menggunakan OpenCV
        - Disampling ke 4 FPS (frame per detik)
        - Frame di-resize ke 224x224 piksel

        **Langkah 2: Segmentasi Temporal**
        - 16 frame berurutan = 1 segmen (~4 detik video)
        - Stride 8 frame (overlap 50%) antar segmen
        - Video pendek (< 16 frame) dilewati

        **Langkah 3: Feature Extraction (S3D)**
        - Setiap segmen diproses oleh S3D (Separable 3D CNN)
        - Pretrained pada Kinetics-400 (video classification)
        - Output: 1 vektor 1024-d per segmen

        **Langkah 4: AttentionMIL Model**
        - 16 segmen x 1024-d sebagai input
        - Attention network memberi bobot ke setiap segmen
        - Classifier MLP memproses bag representation
        - Sigmoid output: anomaly score [0, 1]

        **Langkah 5: Evaluasi**
        - Threshold 0.5 -> >= 0.5 = RUSUH
        - Metrics: AUC, Accuracy, F1, Precision, Recall, MCC
        """)

        st.subheader("Model: AttentionMIL")
        st.markdown("""
        Attention-based Multiple Instance Learning (MIL) memperlakukan setiap
        video sebagai **bag** (kumpulan segmen/instance). Model belajar memberikan
        **attention weight** ke setiap segmen, sehingga segmen yang paling relevan
        (mengandung kerusuhan) mendapat bobot tertinggi.

        **Parameter:** 558.082
        **Input:** 16 segments x 1024-d S3D features
        **Output:** Anomaly score [0, 1] (>= 0.5 = Rusuh)
        """)

        st.subheader("Training Details")
        col_tr1, col_tr2 = st.columns(2)
        with col_tr1:
            st.markdown("""
            - Optimizer: Adam (lr=0.001)
            - Loss: Binary Cross-Entropy
            - Batch Size: 32
            - Epochs: 50 (early stopping)
            - Data Split: 80/10/10
            - Device: CPU/GPU
            """)
        with col_tr2:
            st.markdown("""
            - Weight Decay: 1e-4
            - Early Stopping Patience: 10
            - Best Val AUC: 0.9563
            - Best Val Accuracy: 89.09%
            - Augmentasi: Weather augmentation
            """)

        st.subheader("Model Comparison")
        st.markdown("""
        | Model | AUC | Accuracy | F1 | Precision | Recall |
        |-------|-----|----------|----|-----------|--------|
        | XGBoost (baseline) | 0.9440 | 87.30% | 0.8426 | 0.8597 | 0.8261 |
        | AttentionMIL (final) | **0.9563** | **89.09%** | **0.8683** | **0.8627** | **0.8739** |

        **Kesimpulan:** AttentionMIL unggul di semua metrik karena mampu menangkap
        konteks temporal dan fokus ke segmen kerusuhan via attention mechanism.
        """)

        st.subheader("Interpretasi Model")
        st.markdown("""
        1. **Attention Weights** — Segmen dengan weight tinggi -> kontribusi besar
        2. **Feature Ablation** — Mengukur dampak tiap segmen terhadap skor akhir
        3. **Score Evolution** — Skor berubah seiring bertambahnya segmen
        4. **SHAP Analysis (XGBoost)** — Feature importance untuk model baseline

        **Insight:** Model fokus ke segmen dengan gerakan cepat/abnormal (pukulan,
        kejar-mengejar, lemparan). Segmen akhir video cenderung lebih penting.
        """)

    with tab_usage:
        st.header("Cara Penggunaan Aplikasi")
        st.markdown("Aplikasi memiliki 5 halaman yang dapat diakses via sidebar.")

        with st.expander("1. Beranda — Halaman Utama"):
            st.markdown("""
            **Fungsi:** Ringkasan proyek, metrik model, visualisasi utama.

            **Cara pakai:**
            1. Buka aplikasi -> langsung ke Beranda
            2. Lihat metrik (Accuracy, AUC, F1, MCC)
            3. Lihat dataset overview (total video, kelas, split)
            4. Scroll untuk ROC Curve, Confusion Matrix, Attention Weights
            """)

        with st.expander("2. Exploratory Data Analysis"):
            st.markdown("""
            **Fungsi:** Visualisasi interaktif dataset.

            **Tabs:**
            - **Label Distribution** — Bar chart & pie chart distribusi kelas
            - **Source Analysis** — Distribusi per sumber data
            - **Split Analysis** — Distribusi train/val/test
            - **PCA Visualization** — 2D PCA (2000 sample, ~30 detik loading)
            - **t-SNE Visualization** — 2D t-SNE (1000 sample, ~30 detik)
            """)

        with st.expander("3. Demo Model"):
            st.markdown("""
            **Fungsi:** Prediksi interaktif dengan video asli atau fitur dataset.

            **Tabs:**
            - **Video Demo** — Pilih video Rusuh -> Predict -> lihat gauge chart + segment scores. Video diputar langsung dengan bounding box YOLO.
            - **Feature Demo** — Pilih sample Normal/Rusuh -> Predict -> lihat hasil + segment scores
            - **Batch Test Set** — Run Batch Evaluation -> evaluasi 559 test video -> ROC, CM, report
            - **Upload Video** — Upload video sendiri -> otomatis ekstrak fitur S3D -> prediksi AttentionMIL -> bounding box YOLO
            - **CCTV Live** — Webcam langsung dengan deteksi YOLO + status keamanan real-time
            """)

        with st.expander("4. Evaluasi & Interpretasi"):
            st.markdown("""
            **Fungsi:** Metrik evaluasi detail dan interpretasi model.

            **Tabs:**
            - **Model Evaluation** — Semua metrik, ROC, CM, PR curve, score distribution
            - **Model Interpretation** — Attention weights, ablation, score evolution
            - **About Model** — Arsitektur lengkap, training details
            """)

        with st.expander("5. Dokumentasi (halaman ini)"):
            st.markdown("""
            **Fungsi:** Dokumentasi lengkap dataset, metodologi, cara pakai, referensi.

            **Tabs:**
            - Dataset — Penjelasan dataset dan preprocessing
            - Metodologi — Pipeline dan arsitektur
            - Cara Penggunaan — Panduan per halaman
            - Referensi — Daftar pustaka
            """)

        st.subheader("Tips")
        st.markdown("""
        - PCA & t-SNE butuh waktu loading (~30 detik)
        - Batch Test Set proses 559 video (~2-3 menit)
        - Video Demo hanya untuk format H.264 (didukung browser)
        - Upload Video: proses frame + S3D + YOLO butuh ~10-30 detik tergantung durasi
        - CCTV Live: centang "Mode CCTV Otomatis" untuk monitoring berkelanjutan
        - Gunakan sidebar untuk navigasi
        """)

    with tab_ref:
        st.header("Referensi")

        st.subheader("Dataset")
        st.markdown("""
        - UCF Crime: https://www.crcv.ucf.edu/projects/real-world/
        - SCVD: Surveillance Camera Violence Dataset
        - MSV-PG: Moderating Severe Violence - Playground
        - Real Life Nonviolence Dataset
        - YouTube & Instagram: Scraped untuk riset
        """)

        st.subheader("Papers")
        st.markdown("""
        - Ilse, M., Tomczak, J.M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning. ICML.
        - Xie, S., Sun, C., Huang, J., Tu, Z., & Murphy, K. (2018). Rethinking Spatiotemporal Feature Learning. ECCV.
        - Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
        - Lundberg, S.M., & Lee, S.I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
        """)

        st.subheader("Tools")
        st.markdown("""
        - Python 3.14 | PyTorch | Streamlit | Scikit-learn | XGBoost | SHAP
        - OpenCV | Pandas | NumPy | Matplotlib | Seaborn | Plotly
        - torchvision (S3D pretrained on Kinetics-400)
        """)

        st.subheader("Mata Kuliah")
        st.markdown("""
        - UAS Machine Learning — Teknik Informatika
        - Universitas Dian Nuswantoro — Genap 2025/2026
        """)
