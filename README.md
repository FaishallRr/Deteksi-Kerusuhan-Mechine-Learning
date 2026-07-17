# Deteksi Kerusuhan Menggunakan AttentionMIL

**UAS Machine Learning** | Universitas Dian Nuswantoro | Teknik Informatika | 2026

---

## Deskripsi Proyek

Sistem deteksi kerusuhan (perkelahian, tawuran, kerusuhan massa) dari video CCTV/video amatir menggunakan **Attention-based Multiple Instance Learning (AttentionMIL)** dengan ekstraksi fitur **S3D** (Separable 3D CNN, pretrained pada Kinetics-400). Model mengidentifikasi segmen video yang mengandung aktivitas kerusuhan dan memberikan skor anomali secara real-time. Dideploy sebagai aplikasi interaktif Streamlit dengan fitur YOLOv8 person detection, pipeline S3D+AttentionMIL, dan CCTV live webcam.

**Link Aplikasi:** https://deteksi-kerusuhan-mechine-learning.gevsrhre9uxyornmbwzy8z.streamlit.app/

---

## Model

| Komponen | Detail |
|----------|--------|
| **Feature Extractor** | S3D (pretrained Kinetics-400) -> 1024-d |
| **Architecture** | AttentionMIL (attention network + 2-layer MLP) |
| **Parameters** | 558,082 |
| **Input** | 16 segments x 1024 features |
| **Output** | Anomaly score [0, 1] (>= 0.5 = Rusuh) |

### Performance

| Metrik | XGBoost | MILRanking | AttentionMIL |
|--------|---------|------------|--------------|
| AUC | 0.9440 | 0.9124 | **0.9563** |
| Accuracy | 87.30% | 85.30% | **89.09%** |
| F1 Score | 0.8426 | - | **0.8683** |
| Precision | 0.8597 | - | **0.8627** |
| Recall | 0.8261 | - | **0.8739** |
| MCC | - | - | **0.7752** |

Confusion Matrix: TN=297, FP=32, FN=29, TP=201 (test set 559 video)

---

## Struktur Repository

```
├── app/                    # Streamlit application
│   └── app.py              # Main dashboard (all tabs)
├── core/                   # Model implementations
│   ├── mil_attention.py    # AttentionMILModel
│   ├── mil_ranking.py      # MILRankingModel
│   └── __init__.py
├── preprocessing/          # Feature extraction pipeline
│   └── feature_extractor.py
├── models/                 # Saved model weights
│   └── mil_final.pt
├── features/               # Pre-extracted S3D features
│   ├── final_dataset/      # metadata.json (5552 entries)
│   └── demo_videos/        # 14 demo .npy + metadata
├── notebooks/              # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_interpretation.ipynb
├── reports/                # Generated reports & figures
│   ├── evaluation/         # ROC, CM, PR, score distribution
│   ├── interpretation/     # attention weights, ablation, convergence
│   ├── model_comparison/   # XGBoost vs AttentionMIL comparison
│   ├── screenshots/        # Demo screenshots
│   ├── generate_report.py  # DOCX generator
│   └── yt_script.py        # YouTube presentation script
├── docs/                   # Documentation
│   ├── UAS ML.pdf          # Soal UAS
│   ├── problem_statement.md
│   ├── naskah_presentasi.md
│   └── prompt_laporan.md   # Prompt untuk generate laporan
├── test_videos/            # Demo video files
├── requirements.txt
└── README.md
```

---

## Fitur Aplikasi (Streamlit - 5 Halaman)

### 1. Beranda
- Metrik model (AUC, Accuracy, F1, MCC)
- Dataset overview
- ROC Curve + Confusion Matrix + Attention Weights

### 2. Exploratory Data Analysis
- Label Distribution, Source Analysis, Split Analysis
- PCA (2D, 2000 sample) dan t-SNE (2D, 1000 sample)
- Semua grafik interaktif (Plotly)

### 3. Demo Model (5 Tabs)
| Tab | Fitur |
|-----|-------|
| **Video Demo** | 14 video demo + bounding box YOLOv8 + prediksi AttentionMIL + gauge chart |
| **Feature Demo** | Pilih sample Normal/Rusuh dari test set -> prediksi + segment scores |
| **Batch Test Set** | Evaluasi 559 video test -> ROC, CM, classification report |
| **Upload Video** | Upload sendiri -> S3D feature extraction -> AttentionMIL prediction -> YOLO bounding box -> download hasil |
| **CCTV Live** | Webcam langsung dengan deteksi YOLOv8 + status keamanan (Aman/Waspada/Siaga/Rusuh) + mode auto-refresh |

### 4. Evaluasi & Interpretasi
- Metrics, ROC Curve, Confusion Matrix, PR Curve, Score Distribution
- Attention Weights, Feature Ablation, Score Evolution, Score Convergence
- Arsitektur model lengkap + training details

### 5. Dokumentasi
- Dataset, Metodologi, Cara Penggunaan, Referensi

---

## Dataset

| Sumber | Tipe | Jumlah |
|--------|------|--------|
| YouTube API | demo_rusuh / demo_damai / normal | ~2,500 |
| Kaggle (RWF-2000) | fight / non-fight | 2,000 |
| SCVD | violence / non-violence | 800 |
| MSV-PG (HuggingFace) | violence / non-violence | 252 |
| **Total** | 3 kelas -> binary | **5,552 video** |

Split: Train 4,440 (80%) | Val 553 (10%) | Test 559 (10%)
Distribusi: Normal 3,266 | Rusuh 2,286

## Preprocessing Pipeline
1. Frame extraction dengan OpenCV, resize ke 224x224
2. Downsampling ke 4 FPS
3. Segmentasi 16 frame/segmen, stride 8 (overlap 50%)
4. S3D feature extraction -> 1024-d vector per segmen
5. Normalisasi (mean, std ImageNet)
6. Simpan sebagai .npy

## Hyperparameter Tuning
- Grid search 24 konfigurasi (hidden_units: 128/256/512, dropout: 0.2/0.3/0.5, lr: 0.001/0.0005/0.0001, weight_decay: 1e-4/1e-5)
- Model terbaik: hidden_units=256, dropout=0.3, lr=0.001, weight_decay=1e-4

---

## Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit App
```bash
streamlit run app/app.py
```

### 3. Generate Report (DOCX)
```bash
python reports/generate_report.py
```

### 4. Jupyter Notebooks
```bash
jupyter notebook notebooks/01_eda.ipynb
```

---

## Interpretasi Model

1. **Attention Weights** - Setiap segmen mendapat bobot. Model fokus ke segmen dengan gerakan abnormal/kekerasan
2. **Feature Ablation** - Menghapus 1 segmen secara bergantian. Segmen akhir lebih penting
3. **Score Convergence** - Prediksi stabil setelah 8-10 segmen
4. **Score Evolution** - Video normal stabil rendah, video rusuh meningkat bertahap
5. **SHAP Analysis** - Untuk model XGBoost, mengidentifikasi fitur S3D paling diskriminatif

---

## Capaian Pembelajaran (CPL)

- **CPL8:** Mengimplementasikan metode computing untuk menyelesaikan masalah (penerapan MIL untuk deteksi kerusuhan)
- **CPL10:** Mengembangkan sistem cerdas dengan pembelajaran mesin (pengembangan aplikasi Streamlit)
- **Sub-CPMK 8.1.2:** Klasifikasi menggunakan MILRanking dan AttentionMIL
- **Sub-CPMK 8.1.3:** Ensemble learning menggunakan XGBoost

---

## Anggota

| Nama | NIM |
|------|-----|
| Faishal Rasyid Rusianto | A11.2024.15869 |

---

## Referensi

1. Ilse, M., Tomczak, J. M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning. ICML.
2. Xie, S., Sun, C., Huang, J., Tu, Z., & Murphy, K. (2018). Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification. ECCV.
3. Carreira, J., & Zisserman, A. (2017). Quo Vadis, Action Recognition? A New Model and the Kinetics Dataset. CVPR.
4. Tran, D., Bourdev, L., Fergus, R., Torresani, L., & Paluri, M. (2015). Learning Spatiotemporal Features with 3D Convolutional Networks. ICCV.
5. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. ACM SIGKDD.
6. Lundberg, S. M., & Lee, S. I. (2017). A Unified Approach to Interpreting Model Predictions. NeurIPS.
7. Wang, L., et al. (2016). Temporal Segment Networks: Towards Good Practices for Deep Action Recognition. ECCV.
