# NASKAH PRESENTASI YOUTUBE
## Deteksi Kerusuhan Menggunakan AttentionMIL
### UAS Machine Learning - Universitas Dian Nuswantoro
### Faishal Rasyid Rusianto - A11.2024.15869

**Durasi:** ~10 menit

---

## SLIDE 1 — PEMBUKAAN (0:00 - 0:30)

**Naskah:**

> Assalamualaikum Warahmatullahi Wabarakatuh.
>
> Perkenalkan, nama saya Faishal Rasyid Rusianto, NIM A11.2024.15869, dari Program Studi Teknik Informatika, Universitas Dian Nuswantoro.
>
> Pada kesempatan ini, saya akan mempresentasikan project UAS Machine Learning saya yang berjudul **"Deteksi Kerusuhan Menggunakan Attention-based Multiple Instance Learning"**.

---

## SLIDE 2 — LATAR BELAKANG & RUMUSAN MASALAH (0:30 - 1:30)

**Naskah:**

> Indonesia sering menghadapi tantangan kerusuhan dan demonstrasi yang berujung anarkis. Pemantauan CCTV secara manual tidak efisien karena operator manusia mudah lelah dan jumlah CCTV terus bertambah.
>
> **Rumusan masalah:**
> 1. Bagaimana mendeteksi kerusuhan dari video secara otomatis?
> 2. Bagaimana mengimplementasikan AttentionMIL untuk meningkatkan akurasi?
> 3. Bagaimana performa model dalam membedakan video rusuh dan non-rusuh?
>
> **Tujuan:** Membangun sistem deteksi kerusuhan otomatis dengan AttentionMIL, membandingkan dengan XGBoost, dan menyediakan aplikasi Streamlit interaktif.

---

## SLIDE 3 — LANDASAN TEORI & PERBANDINGAN ALGORITMA (1:30 - 3:00)

**Naskah:**

> Project ini menggunakan beberapa algoritma sesuai Sub-CPMK 8.1.2 dan 8.1.3:
>
> **1. XGBoost (Ensemble Learning)**
> Algoritma gradient boosting yang menggabungkan banyak decision trees. Digunakan sebagai baseline karena interpretabilitasnya melalui SHAP.
>
> **2. Multiple Instance Learning (MIL)**
> Pendekatan di mana setiap video adalah 'bag' dan segmen video adalah 'instance'. Tidak perlu labeling frame-by-frame.
>
> **3. Attention Mechanism**
> Memberi bobot berbeda pada setiap segmen. Segmen dengan gerakan abnormal mendapat bobot tinggi.
>
> **Perbandingan karakteristik algoritma:**
> - Kompleksitas: XGBoost (rendah) < MILRanking (sedang) < AttentionMIL (tinggi)
> - Akurasi: MILRanking (85.3%) < XGBoost (87.3%) < AttentionMIL (89.09%)
> - Interpretabilitas: XGBoost+SHAP (tinggi) > AttentionMIL (sedang) > MILRanking (rendah)

---

## SLIDE 4 — DATASET & FEATURE EXTRACTION (3:00 - 4:00)

**Naskah:**

> Dataset berjumlah **5.552 video** dari 4 sumber:
> 1. YouTube API — Video demo damai/rusuh dari Indonesia
> 2. Kaggle RWF-2000 — 2.000 video pertengkaran internasional
> 3. SCVD — Smart City Violence Dataset
> 4. MSV-PG — Multi-Source Violence Dataset (HuggingFace)
>
> **Proses Feature Extraction:**
> - Setiap video dipotong menjadi segmen 16 frame
> - Setiap segmen di-resize ke 640x640
> - Diekstrak menggunakan **S3D (Separable 3D CNN)** pre-trained pada Kinetics-400
> - Output: vektor fitur 1024 dimensi per segmen
> - Total: setiap video = 16 segmen × 1024 fitur
>
> Dataset dibagi 80% train, 10% val, 10% test secara stratified.

---

## SLIDE 5 — EXPLORATORY DATA ANALYSIS (4:00 - 4:45)

**Naskah:**

> EDA dilakukan untuk memahami karakteristik dataset:
>
> **Label Distribution:** Dataset memiliki 3 kelas (demo_rusuh, demo_damai, normal) yang dipetakan ke binary: 41% rusuh, 59% non-rusuh.
>
> **PCA & t-SNE Visualization:** Fitur S3D mampu memisahkan kelas rusuh dan non-rusuh, meskipun ada overlap di beberapa region.
>
> **Feature Analysis:** Distribusi fitur menunjukkan variance yang cukup untuk diskriminasi.
>
> **Source Distribution:** Data multi-source meningkatkan generalisasi model.
>
> Visualisasi ini bisa dilihat interaktif di aplikasi Streamlit pada halaman EDA.

---

## SLIDE 6 — PEMODELAN & HYPERPARAMETER TUNING (4:45 - 6:00)

**Naskah:**

> Tiga model dilatih dan dibandingkan:
>
> **1. XGBoost (Baseline)**
> - Input: Mean pooling fitur per video (1024-d)
> - Tuning: Grid search dengan 3-fold CV
> - Parameter: max_depth [4,8], learning_rate [0.05, 0.1]
> - Best: max_depth=4, learning_rate=0.1
>
> **2. MILRankingModel (Frame-level)**
> - Setiap segmen diproses independen oleh MLP
> - Max pooling untuk agregasi
> - Parameter: 263.170
>
> **3. AttentionMILModel (Video-level) — Model Utama**
> - Attention Network: Linear(1024→256) + Tanh + Linear(256→1) + Softmax
> - Classifier: 2-layer MLP (1024→256→128→1) + Dropout 0.3
> - Total parameter: 558.082
> - Tuning: 24 konfigurasi (hidden units, dropout, learning rate, weight decay)
> - Loss: Binary Cross-Entropy, Optimizer: Adam
>

---

## SLIDE 7 — HASIL EVALUASI & CONFUSION MATRIX (6:00 - 8:00)

**Naskah:**

> Hasil evaluasi pada test set (559 video):
>
> **Tabel Perbandingan Model:**
>
> | Model | AUC | F1 | Precision | Recall | Accuracy |
> |-------|-----|----|-----------|--------|---------|
> | XGBoost | 0.9440 | 0.8426 | 0.8597 | 0.8261 | 87.30% |
> | MILRanking | 0.9124 | 0.8315 | 0.8241 | 0.8390 | 85.30% |
> | **AttentionMIL** | **0.9563** | **0.8683** | **0.8627** | **0.8739** | **89.09%** |
>
> **AttentionMIL sebagai Model Terbaik** — unggul di semua metrik. Alasan:
> - Attention mechanism memungkinkan model fokus pada segmen relevan
> - Segment-level information lebih informatif daripada mean pooling
>
> **Confusion Matrix AttentionMIL:**
>
> ```
>               Pred Normal     Pred Rusuh
> Actual Normal    297  (TN)      32  (FP)
> Actual Rusuh      29  (FN)     201  (TP)
> ```
>
> Analisis:
> - **True Negatives (TN) = 297:** Video normal benar terdeteksi
> - **False Positives (FP) = 32:** Video normal salah dikira rusuh (8.9% dari normal)
> - **True Positives (TP) = 201:** Video rusuh benar terdeteksi
> - **False Negatives (FN) = 29:** Video rusuh terlewat (12.6% dari rusuh)
>
> **Score Distribution:** Pemisahan bersih — normal di skor 0-0.2, rusuh di 0.7-1.0

---

## SLIDE 8 — SHAP ANALYSIS & INTERPRETASI (8:00 - 9:00)

**Naskah:**

> **SHAP Analysis (XGBoost):**
> Menggunakan SHAP TreeExplainer untuk menginterpretasi model XGBoost:
> - Fitur S3D tertentu menjadi indikator kuat kerusuhan
> - SHAP summary plot menunjukkan feature importance dan arah pengaruh
> - Plot SHAP bar mengidentifikasi 15 fitur paling penting
>
> **Interpretasi AttentionMIL:**
> Model memberikan interpretabilitas天然的 melalui:
>
> 1. **Attention Weights:** Setiap segmen mendapat bobot. Segmen dengan gerakan abnormal mendapat bobot tinggi pada video rusuh. Pada video normal, bobot lebih merata.
>
> 2. **Feature Ablation:** Menghilangkan segmen tertentu mengubah skor prediksi. Segmen akhir cenderung lebih penting.
>
> 3. **Score Convergence:** Prediksi stabil setelah 8-10 segmen — bisa prediksi cepat bahkan sebelum video selesai.
>
> 4. **Score Evolution:** Video normal skor rendah stabil. Video rusuh skor meningkat bertahap seiring segmen menunjukkan aktivitas mencurigakan.

---

## SLIDE 9 — STREAMLIT DASHBOARD TOUR (9:00 - 9:45)

**Naskah:**

> Aplikasi Streamlit memiliki 4 halaman:
>
> **1. Beranda:** Overview project, metrik model, dataset info
> **2. EDA Interaktif:** Distribusi label per split, PCA/t-SNE interaktif, source analysis
> **3. Demo Model:** Pilih sample video → model prediksi → gauge interaktif → segment-level scores
> **4. Evaluasi:** ROC curve, Confusion Matrix, classification report, attention weights
>
> Aplikasi ini bisa diakses publik setelah di-deploy ke Streamlit Cloud.

---

## SLIDE 10 — KESIMPULAN & SARAN (9:45 - 10:15)

**Naskah:**

> **Kesimpulan:**
> 1. AttentionMIL berhasil diimplementasikan dengan AUC 0.9563 dan akurasi 89.09%
> 2. Model mampu membedakan video rusuh dan non-rusuh secara efektif
> 3. Attention mechanism memberikan peningkatan signifikan vs XGBoost (AUC 0.9440)
> 4. Dataset multi-source 5.552 video memberikan generalisasi baik
> 5. Aplikasi Streamlit untuk demo interaktif
>
> **Saran:**
> - Integrasi deteksi senjata tajam sebagai fitur tambahan
> - Deployment real-time pada aliran CCTV
> - Model lebih ringan untuk edge devices

---

## SLIDE 11 — PENUTUP (10:15 - 10:30)

**Naskah:**

> Sekian presentasi dari saya. Terima kasih atas perhatiannya.
>
> Wassalamualaikum Warahmatullahi Wabarakatuh.
>
> Jika ada pertanyaan, silakan disampaikan.

---

## 🎥 Tips Rekam

1. **Tools:** OBS Studio (gratis) / Windows + G (Xbox Game Bar)
2. **Durasi:** 10-12 menit
3. **Slide:** PowerPoint/Canva — export gambar
4. **Screenshot yang ditampilkan:**
   - `reports/evaluation/roc_curve.png`
   - `reports/evaluation/confusion_matrix.png`
   - `reports/evaluation/score_distribution.png`
   - `reports/interpretation/attention_weights.png`
   - `reports/model_comparison/shap_summary.png`
   - `reports/model_comparison/shap_feature_importance.png`
   - Aplikasi Streamlit saat running (http://localhost:8501)
5. **Upload:** YouTube → unlisted → copy link → taruh di README.md
