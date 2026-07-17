# Deteksi Kerusuhan Menggunakan AttentionMIL 🚨

**UAS Machine Learning** | Universitas Dian Nuswantoro | Teknik Informatika

---

## 📋 Deskripsi Proyek

Sistem deteksi kerusuhan dari video CCTV/video amatir menggunakan **Attention-based Multiple Instance Learning (AttentionMIL)** dengan ekstraksi fitur **S3D**. Model mampu mengidentifikasi segmen video yang mengandung aktivitas kerusuhan secara real-time.

**Masalah:** Deteksi dini kerusuhan dari rekaman video untuk membantu penegak hukum merespons lebih cepat.

## 🧠 Model

| Komponen | Detail |
|----------|--------|
| **Feature Extractor** | S3D (pretrained Kinetics-400) → 1024-d |
| **Architecture** | AttentionMIL (attention + 2-layer MLP) |
| **Parameters** | 558,082 |
| **Input** | 16 segments × 1024 features |
| **Output** | Anomaly score [0, 1] (≥0.5 = Rusuh) |

### 📊 Performance

| Metrik | Nilai |
|--------|-------|
| AUC | 0.9563 |
| Accuracy | 89.09% |
| F1 Score | 0.8683 |
| Precision | 0.8627 |
| Recall | 0.8739 |
| MCC | 0.7752 |

## 📁 Struktur Repository

```
├── app/                    # Streamlit application
│   └── app.py
├── core/                   # Model implementation
│   └── mil_attention.py
├── data/                   # Video dataset (raw/processed)
├── features/               # Pre-extracted S3D features
│   └── final_dataset/
├── models/                 # Saved model weights
│   └── mil_final.pt
├── notebooks/              # Jupyter notebooks
│   ├── 01_eda.ipynb
│   ├── 02_modeling.ipynb
│   └── 03_interpretation.ipynb
├── preprocessing/          # Feature extraction pipeline
├── training/               # Model training scripts
├── evaluation/             # Evaluation & interpretation scripts
├── reports/                # Generated reports & figures
│   ├── evaluation/
│   ├── interpretation/
│   └── eda/
├── requirements.txt
└── README.md
```

## 🚀 Cara Menjalankan

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. Run Streamlit App
```bash
streamlit run app/app.py
```

### 3. Run Evaluation
```bash
python evaluation/fast_eval.py       # Performance metrics + plots
python evaluation/interpret_model.py  # Attention weights + ablation
```

### 4. Jupyter Notebooks
```bash
jupyter notebook notebooks/01_eda.ipynb
```

## 📊 Dataset

| Sumber | Kelas | Jumlah |
|--------|-------|--------|
| YouTube API | demo_rusuh / demo_damai / normal | ~2,500 |
| Kaggle (RWF-2000) | fight / non-fight | 2,000 |
| SCVD | violence / non-violence | 800 |
| MSV-PG | violence / non-violence | 252 |
| **Total** | 3 kelas (binary) | **5,552 video** |

## 🔍 Interpretasi Model

Model AttentionMIL memberikan interpretabilitas melalui:

1. **Attention Weights** — Menunjukkan segmen video mana yang paling berkontribusi pada keputusan
2. **Feature Ablation** — Mengukur dampak penghapusan setiap segmen terhadap skor akhir
3. **Score Evolution** — Melihat bagaimana skor berubah seiring bertambahnya segmen yang diproses

## 🎯 Tujuan Pembelajaran (CPL)

- **CPL8:** Mengimplementasikan metode computing untuk menyelesaikan masalah
- **CPL10:** Mengembangkan sistem cerdas dengan pembelajaran mesin

## 👥 Anggota Kelompok

| Nama | NIM |
|------|-----|
| Faishal Rasyid Rusianto | A11.2024.15869 |

## 📝 Referensi

- Ilse, M., Tomczak, J.M., & Welling, M. (2018). *Attention-based Deep Multiple Instance Learning.* ICML.
- Xie, S., et al. (2018). *Rethinking Spatiotemporal Feature Learning: Speed-Accuracy Trade-offs in Video Classification.* ECCV.
