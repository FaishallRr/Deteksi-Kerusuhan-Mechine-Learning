# NASKAH PRESENTASI YOUTUBE (DETAIL)
## Deteksi Kerusuhan Menggunakan AttentionMIL
### UAS Machine Learning - Universitas Dian Nuswantoro
### Faishal Rasyid Rusianto - A11.2024.15869

**Durasi:** ~15 menit (11 slide)

---

## SLIDE 1 — PEMBUKAAN (0:00 - 0:30)

**Naskah:**

> Assalamualaikum Warahmatullahi Wabarakatuh.
>
> Perkenalkan, nama saya Faishal Rasyid Rusianto, NIM A11.2024.15869, dari Program Studi Teknik Informatika, Universitas Dian Nuswantoro.
>
> Pada kesempatan ini, saya akan mempresentasikan project UAS Machine Learning saya yang berjudul **"Deteksi Kerusuhan Menggunakan Attention-based Multiple Instance Learning"**.
>
> Project ini merupakan solusi end-to-end berbasis machine learning untuk mendeteksi kerusuhan dari video secara otomatis, mencakup seluruh pipeline mulai dari akuisisi data, preprocessing, pemodelan, evaluasi, hingga deployment ke aplikasi web interaktif.

---

## SLIDE 2 — LATAR BELAKANG & RUMUSAN MASALAH (0:30 - 1:30)

**Naskah:**

> **Latar Belakang:**
> Indonesia sebagai negara demokrasi sering menghadapi demonstrasi dan kerusuhan massa. Pemantauan CCTV secara manual memiliki keterbatasan: operator manusia mudah lelah, jumlah CCTV terus bertambah, dan respons terhadap insiden sering terlambat.
>
> **Rumusan Masalah, sesuai UAS Soal 1 (Problem Definition):**
> 1. Bagaimana mendeteksi kerusuhan dari video secara otomatis menggunakan machine learning?
> 2. Algoritma apa yang paling tepat — dengan mempertimbangkan kompleksitas, akurasi, dan interpretabilitas (Sesuai Sub-CPMK 8.1.2 dan 8.1.3)?
> 3. Bagaimana performa model (AUC, F1, Accuracy) dalam membedakan video rusuh dan non-rusuh?
> 4. Bagaimana menyajikan solusi dalam aplikasi Streamlit yang interaktif?
>
> **Tujuan bisnis:** Membantu penegak hukum merespons kerusuhan lebih cepat dengan sistem deteksi otomatis.
>
> **Metrik kesuksesan:** AUC > 0.90, Accuracy > 85%, F1 Score > 0.85.

---

## SLIDE 3 — DATASET & SUMBER DATA (1:30 - 2:30)

**Naskah:**

> Sesuai UAS Soal 1 poin 2, dataset dikumpulkan dari berbagai sumber untuk memastikan generalisasi model yang baik.
>
> **Total:** 5.552 video (3 kelas → dipetakan ke 2 kelas biner)
>
> **Sumber Data:**
> 1. **UCF Crime Dataset (Kaggle):** 2.850 video abnormal (perkelahian, perampokan, penembakan) dan 950 video normal. Sumber publik, lisensi akademik.
> 2. **SCVD (Surveillance Camera Violence Dataset):** 1.659 video kekerasan dari CCTV internasional.
> 3. **MSV-PG (Moderating Severe Violence - Playground):** 600+ video pertengkaran dari playground, dari HuggingFace Datasets.
> 4. **YouTube & Instagram:** 500+ video tawuran Indonesia dan aktivitas normal, dikumpulkan via YouTube API dan scraping untuk riset.
> 5. **Real Life Nonviolence:** 900+ video interaksi non-kekerasan dari dataset akademik.
>
> **Distribusi kelas (setelah mapping biner):**
> - Rusuh: 2.286 video (41%)
> - Normal/Damai: 3.266 video (59%)
>
> Dataset ini memenuhi kriteria UAS: multi-source, relevan dengan masalah nyata, dan cukup besar untuk deep learning.

---

## SLIDE 4 — PREPROCESSING PIPELINE DETAIL (2:30 - 5:00)

**Naskah:**

> Ini adalah bagian kritis dari project. Sesuai UAS Soal 2, setiap langkah preprocessing harus didokumentasikan dan dijustifikasi.
>
> **Langkah 1: Frame Extraction (Mengapa 4 FPS?)**
> Video dibaca menggunakan OpenCV. Kami melakukan resampling ke 4 FPS (frame per detik). Mengapa 4 FPS?
> - **Terlalu tinggi (30 FPS):**  menghasilkan ribuan frame per video → komputasi mahal, redundant (frame berturut-turut hampir identik)
> - **Terlalu rendah (1 FPS):**  kehilangan informasi temporal untuk aktivitas kerusuhan yang terjadi cepat (pukulan, lemparan)
> - **4 FPS:** keseimbangan optimal — cukup untuk menangkap gerakan abnormal tanpa redundansi
>
> Frame di-resize ke 224x224 piksel, sesuai dengan input yang diharapkan oleh S3D (Separable 3D CNN).
>
> **Langkah 2: Segmentasi Temporal (Mengapa 16 frame, stride 8?)**
> 16 frame berurutan = 1 segmen (setara ~4 detik video pada 4 FPS). Mengapa 16?
> - S3D dirancang untuk memproses 16 frame sekaligus (temporal window)
> - 4 detik cukup untuk menangkap satu kejadian kerusuhan (pukulan, lemparan)
>
> Stride 8 antar segmen (50% overlap). Mengapa overlap?
> - Agar momen kritis tidak terpotong di antara dua segmen
> - Setiap momen muncul di 2 segmen → robust terhadap segmentasi
>
> **Langkah 3: Feature Extraction dengan S3D (Mengapa S3D?)**
> Setiap segmen diproses oleh S3D (Separable 3D CNN). Mengapa S3D bukan model lain?
> - S3D adalah varian efisien dari I3D (Inflated 3D ConvNet)
> - Separable convolution → lebih cepat dari I3D dengan akurasi setara
> - Pretrained pada Kinetics-400 (dataset 400 kelas aksi manusia) → transfer learning: model sudah tahu cara mengenali gerakan manusia
> - Output: vektor 1024 dimensi per segmen → representasi kaya yang menangkap pola gerakan
>
> **Langkah 4: Augmentasi Data (Weather Augmentation)**
> Untuk meningkatkan generalisasi, kami menerapkan weather augmentation secara acak:
> - Perubahan brightness/kontras (simulasi kondisi pencahayaan berbeda)
> - Gaussian noise (simulasi CCTV kualitas rendah)
> - Gaussian blur (simulasi gerakan kamera/fokus tidak sempurna)
>
> **Output akhir:** Setiap video menjadi tensor [N_segmen, 1024]. Untuk model, kita ambil 16 segmen pertama. Video pendek (< 16 segmen) di-padding dengan repeat.

---

## SLIDE 5 — EXPLORATORY DATA ANALYSIS (5:00 - 6:15)

**Naskah:**

> Sesuai UAS Soal 2, EDA dilakukan untuk memahami dataset sebelum modeling. 5 insight penting:
>
> **Insight 1: Distribusi Label — Tidak Sepenuhnya Balance**
> Rusuh 41%, Normal 59%. Ini imbalance moderat. Kami menggunakan AUC sebagai metrik utama karena robust terhadap imbalance.
>
> **Insight 2: Multi-Source = Multi-Domain**
> Distribusi per source menunjukkan variasi: UCF Crime dominan disusul Real Life Nonviolence dan SCVD. Setiap source punya karakteristik visual berbeda (CCTV hitam-putih, YouTube colorful, amatir vs profesional). Ini bagus untuk generalisasi.
>
> **Insight 3: PCA — Separabilitas Terbatas di 2D**
> [TAMPILKAN GAMBAR PCA]
> PCA 2 komponen hanya menjelaskan ~20% variance — fitur 1024-d tidak bisa direduksi ke 2D tanpa kehilangan informasi. Namun overlap tidak terlalu parah.
>
> **Insight 4: t-SNE — Cluster Lebih Terpisah**
> [TAMPILKAN GAMBAR t-SNE]
> t-SNE menunjukkan cluster lebih terpisah antara rusuh dan normal. Konfirmasi bahwa fitur S3D memang membawa sinyal diskriminatif.
>
> **Insight 5: Distribusi Jumlah Segmen**
> Rata-rata video punya ~8-12 segmen (setara 32-48 detik video). Model AttentionMIL menggunakan 16 segmen pertama.
>
> Semua visualisasi ini bisa dilihat interaktif di aplikasi Streamlit halaman EDA.

---

## SLIDE 6 — TRAIN/VAL/TEST SPLIT + MODEL ARCHITECTURE (6:15 - 8:00)

**Naskah:**

> **Train/Val/Test Split: Mengapa 80/10/10?**
>
> Ini adalah rasio standar dalam deep learning. Alasannya:
> - **80% Train:** Data cukup besar (4.440 video) untuk melatih 558 ribu parameter model
> - **10% Val (553 video):** Untuk early stopping dan hyperparameter tuning. Cukup untuk melihat trend without overfitting ke test set
> - **10% Test (559 video):** Evaluasi final yang tidak pernah dilihat model selama training
>
> Split dilakukan secara stratified — menjaga proporsi kelas yang sama di setiap split. Ini penting agar evaluasi tidak bias.
>
> **Kita membandingkan 3 model, sesuai UAS Soal 3:**
> **1. XGBoost (Ensemble Learning — Sub-CPMK 8.1.3)**
> - Input: mean pooling fitur per video (16 segmen → rata-rata → 1024-d)
> - Tuning: GridSearchCV (3-fold)
> - Best params: max_depth=4, learning_rate=0.1, n_estimators=100
> - Dipilih sebagai baseline karena interpretabilitas via SHAP
>
> **2. AttentionMIL — Model Utama (Sub-CPMK 10.1.2: Kembangkan metode sendiri)**
> [TAMPILKAN DIAGRAM ARSITEKTUR]
>
> Arsitektur detail:
> ```
> Input: 16 segmen × 1024-d
>        │
>        ▼
> Attention Network:
> ┌ Linear(1024 → 256) ─── Tanh ─── Linear(256 → 1) ─┬─┐
> │   (memproyeksi tiap segmen ke 256-d)              │  │
> │   (aktivasi Tanh untuk non-linearitas)            │  │
> │   (skor attention per segmen)                     │  │
> └──────────────────────────────────────────────────────┘
>        │
>        ▼
> Softmax → Attention Weights (bobot per segmen, total = 1)
>        │
>        ▼
> Weighted Sum → Bag Representation (1024-d)
>        │
>        ▼
> Classifier MLP:
> ┌ Linear(1024 → 256) ─ ReLU ─ Dropout(0.3) ─┐
> │ Linear(256 → 128)  ─ ReLU ─ Dropout(0.3)  │
> │ Linear(128 → 1)                            │
> └────────────────────────────────────────────┘
>        │
>        ▼
> Sigmoid → Anomaly Score [0, 1]
> ```
>
> Mengapa AttentionMIL?
> - **Multiple Instance Learning:** Tidak perlu label per-frame. Cukup label per video (apakah ada kerusuhan di video ini?)
> - **Attention Mechanism:** Model belajar fokus ke segmen yang mengandung kerusuhan. Segmen dengan gerakan abnormal mendapat bobot tinggi, segmen normal diabaikan.
> - **Dropout 0.3:** Mencegah overfitting pada 558 ribu parameter
> - **Training:** Adam optimizer, LR=0.001, 50 epoch dengan early stopping

---

## SLIDE 7 — HASIL EVALUASI: METRIK & GRAFIK (8:00 - 10:30)

**Naskah:**

> Sesuai UAS Soal 3, kami melakukan evaluasi komprehensif pada test set 559 video.
>
> **Tabel Perbandingan Model:**
>
> | Model | AUC | Accuracy | F1 | Precision | Recall |
> |-------|-----|----------|----|-----------|--------|
> | XGBoost (baseline) | 0.9440 | 87.30% | 0.8426 | 0.8597 | 0.8261 |
> | AttentionMIL (final) | **0.9563** | **89.09%** | **0.8683** | **0.8627** | **0.8739** |
>
> AttentionMIL unggul di SEMUA metrik. Mari kita lihat detailnya.
>
> **1. ROC Curve & AUC**
> [TAMPILKAN GAMBAR ROC CURVE]
> Kurva ROC memplot TPR (True Positive Rate = recall) vs FPR (False Positive Rate) pada berbagai threshold.
> - Garis biru adalah model kita. Semakin mendekati pojok kiri atas, semakin baik.
> - Garis hitam putus-putus adalah random classifier (AUC = 0.5).
> - AUC = **0.9563** artinya: jika kita ambil random pair (1 video rusuh, 1 video normal), model punya 95.63% kemungkinan memberi skor lebih tinggi ke video rusuh. Ini sangat baik.
>
> **2. Confusion Matrix**
> [TAMPILKAN GAMBAR CONFUSION MATRIX]
> - **TN = 297:** Model benar mengatakan NORMAL untuk 297 video normal
> - **FP = 32:** Model salah mengatakan RUSUH untuk 32 video normal (false alarm 8.9%)
> - **FN = 29:** Model salah mengatakan NORMAL untuk 29 video rusuh (missed detection 12.6%)
> - **TP = 201:** Model benar mengatakan RUSUH untuk 201 video rusuh
>
> Interpretasi bisnis: False positive (8.9%) = alarm palsu yang bisa diabaikan (lebih aman waspada). False negative (12.6%) = kerusuhan terlewat — perlu improvement.
>
> **3. Precision-Recall Curve**
> [TAMPILKAN GAMBAR PR CURVE]
> Berbeda dengan ROC, PR Curve lebih sensitif terhadap class imbalance. Kurva yang tetap tinggi di sebelah kiri atas menunjukkan precision tetap baik bahkan saat recall tinggi.
>
> **4. Score Distribution**
> [TAMPILKAN GAMBAR SCORE DISTRIBUTION]
> Grafik ini menunjukkan distribusi skor prediksi per kelas:
> - **Normal/Damai (hijau):** Sebagian besar terkonsentrasi di skor 0.0 - 0.2 (yakin bahwa ini normal)
> - **Rusuh (merah):** Sebagian besar di skor 0.7 - 1.0 (yakin bahwa ini rusuh)
> - Ada sedikit overlap di sekitar 0.3 - 0.7 (zona tidak yakin)
>
> Pemisahan yang bersih ini menandakan model bekerja dengan baik.

---

## SLIDE 8 — INTERPRETASI MODEL DETAIL (10:30 - 12:30)

**Naskah:**

> Sesuai UAS Soal 3 poin 4, kami melakukan interpretasi model mendalam.
>
> **Attention Weights — Model "Melihat" ke Mana?**
> [TAMPILKAN GAMBAR ATTENTION WEIGHTS]
>
> Grafik ini menunjukkan bobot attention untuk setiap segmen video:
> - **Sumbu X:** Nomor segmen (1 sampai 16)
> - **Sumbu Y:** Bobot attention (semakin tinggi, semakin penting)
> - **Bar merah:** Video RUSUH
> - **Bar hijau:** Video NORMAL
>
> **Interpretasi:** Pada video rusuh, bobot tidak merata — beberapa segmen mendapat bobot tinggi (yang mengandung pukulan, kejar-mengejar). Pada video normal, bobot lebih merata karena semua segmen memang normal.
>
> **Feature Ablation — Apa yang Terjadi Jika Satu Segmen Dihilangkan?**
> [TAMPILKAN GAMBAR FEATURE ABLATION]
>
> Kami menghilangkan satu per satu segmen dan mengukur perubahan skor:
> - Jika menghilangkan segmen i menyebabkan skor turun drastis → segmen itu penting untuk deteksi rusuh
> - Segmen akhir cenderung lebih penting — kerusuhan biasanya meningkat menjelang akhir
>
> **Score Convergence — Berapa Banyak Segmen yang Dibutuhkan?**
> [TAMPILKAN GAMBAR SCORE CONVERGENCE]
>
> Grafik ini menunjukkan skor prediksi saat kita menggunakan N segmen pertama:
> - Pada 2-4 segmen: Skor masih fluktuatif
> - Pada 8-10 segmen: Skor mulai stabil
> - Setelah 12 segmen: Skor mendekati final
>
> **Implikasi:** Model bisa memberikan prediksi akurat hanya dengan ~8 segmen (~32 detik video) — ini penting untuk real-time detection!
>
> **SHAP Analysis untuk XGBoost (Model Comparison)**
> [TAMPILKAN GAMBAR SHAP SUMMARY + SHAP FEATURE IMPORTANCE]
>
> SHAP (SHapley Additive exPlanations) menjelaskan prediksi XGBoost:
> - **Summary plot (kiri):** Setiap titik = 1 sample. Warna merah = fitur bernilai tinggi, biru = rendah. Posisi kanan/kiri = kontribusi ke prediksi rusuh/normal.
> - **Bar plot (kanan):** 15 fitur S3D paling penting. Beberapa fitur S3D secara konsisten menjadi indikator kerusuhan (mendeteksi gerakan cepat, perubahan tekstur mendadak).
>
> **Key Insights Interpretasi:**
> 1. Model fokus ke segmen dengan gerakan abnormal
> 2. Tidak perlu menonton seluruh video — 8 segmen cukup
> 3. SHAP mengkonfirmasi fitur S3D memang relevan untuk deteksi kerusuhan

---

## SLIDE 9 — STREAMLIT DASHBOARD TOUR (12:30 - 14:00)

**Naskah:**

> Sesuai UAS Soal 4, saya mengembangkan aplikasi Streamlit dengan 5 halaman:
>
> **Halaman 1: Beranda**
> [TAMPILKAN SCREENSHOT BERANDA]
> Overview project: metrik utama (Accuracy 89.09%, AUC 0.9563, F1 0.8683, MCC 0.7752), dataset info, model architecture, dan tujuan proyek.
>
> **Halaman 2: Exploratory Data Analysis**
> [TAMPILKAN SCREENSHOT EDA]
> Lima tab interaktif: distribusi label (bar + pie chart), analisis sumber data (horizontal bar + cross-tab), distribusi split train/val/test, PCA visualization (2.000 sample acak dengan Plotly), dan t-SNE visualization (1.000 sample, perplexity 30).
>
> **Halaman 3: Demo Model**
> [TAMPILKAN SCREENSHOT DEMO]
> Tiga tab:
> - **Video Demo (Asli):** Pilih video rusuh dari 4 video demo → video diputar langsung menggunakan st.video() → klik "Predict Video Ini" → muncul gauge chart anomaly score, prediksi (RUSUH/NORMAL), confidence, true label, dan segment-level bar chart.
> - **Feature Demo:** Pilih sample dari 559 test items (filter Normal/Rusuh) → prediksi cepat → lihat hasil + segment scores
> - **Batch Test Set:** Klik "Run Batch Evaluation" → proses 559 video → tampilkan accuracy, AUC, confusion matrix, ROC curve, classification report
>
> **Halaman 4: Evaluasi & Interpretasi**
> [TAMPILKAN SCREENSHOT EVALUASI]
> Tiga tab:
> - **Model Evaluation:** Semua metrik + ROC curve, CM, PR curve, score distribution
> - **Model Interpretation:** Attention weights, feature ablation, score evolution, score convergence
> - **About Model:** Arsitektur lengkap + training details
>
> **Halaman 5: Dokumentasi**
> [TAMPILKAN DOKUMENTASI]
> Empat tab: Dataset (sumber, preprocessing), Metodologi (pipeline end-to-end, arsitektur, training), Cara Penggunaan (panduan per halaman), Referensi (papers & tools)
>
> **Deployment:** Aplikasi akan di-deploy ke Streamlit Community Cloud agar bisa diakses publik.
>
> [DEMO LANGSUNG: BUKA localhost:8501, TUNJUKKAN SETIAP HALAMAN SECARA SINGKAT, FOKUS KE HALAMAN DEMO — PREDICT 1 VIDEO RUSUH + BATCH TEST SET]

---

## SLIDE 10 — KESIMPULAN & SARAN (14:00 - 15:00)

**Naskah:**

> **Kesimpulan — Menjawab Rumusan Masalah:**
>
> 1. **Masalah 1: Deteksi otomatis** — AttentionMIL berhasil mendeteksi kerusuhan dari video dengan AUC 0.9563 dan akurasi 89.09%.
>
> 2. **Masalah 2: Perbandingan algoritma (Sub-CPMK 8.1.2 & 8.1.3):**
>    - XGBoost: kompleksitas rendah, interpretabilitas tinggi (SHAP), akurasi 87.30%
>    - AttentionMIL: kompleksitas tinggi, interpretabilitas sedang (attention), akurasi 89.09% — TERBAIK
>    - Perbandingan menunjukkan attention mechanism memberikan peningkatan signifikan
>
> 3. **Masalah 3: Performa model** — AUC 0.9563 melampaui target (AUC > 0.90). Model bisa membedakan rusuh vs normal dengan sangat baik.
>
> 4. **Masalah 4: Aplikasi** — Streamlit app dengan 5 halaman interaktif siap digunakan.
>
> **Saran untuk Pengembangan ke Depan:**
> - Integrasi deteksi senjata tajam (multimodal: visual + audio)
> - Deployment real-time pada aliran CCTV (menggunakan frame sampling + sliding window)
> - Model distillation untuk perangkat edge (Raspberry Pi, smartphone)
> - Dataset diperluas dengan lebih banyak variasi dari Indonesia
>
> **Keterkaitan dengan CPL:**
> - **CPL 8 (Computing):** Implementasi AttentionMIL sebagai algoritma kompleks untuk masalah nyata
> - **CPL 10 (Sistem Cerdas):** Solusi berbasis sistem cerdas untuk deteksi kerusuhan, dari rancang hingga evaluasi

---

## SLIDE 11 — PENUTUP (15:00 - 15:30)

**Naskah:**

> **Link dan Referensi:**
> - GitHub Repository: https://github.com/FaishallRr/Deteksi-Kerusuhan-Mechine-Learning
> - Aplikasi Streamlit: [URL setelah deploy]
> - Laporan DOCX: Ada di folder `reports/`
>
> **Daftar Pustaka Utama:**
> 1. Ilse, M., Tomczak, J.M., & Welling, M. (2018). Attention-based Deep Multiple Instance Learning. ICML.
> 2. Xie, S., et al. (2018). Rethinking Spatiotemporal Feature Learning: S3D. ECCV.
> 3. Chen, T., & Guestrin, C. (2016). XGBoost: A Scalable Tree Boosting System. KDD.
> 4. Lundberg, S.M., & Lee, S.I. (2017). SHAP: A Unified Approach to Interpreting Model Predictions. NeurIPS.
>
> Sekian presentasi dari saya. Terima kasih atas perhatiannya.
>
> Wassalamualaikum Warahmatullahi Wabarakatuh.
>
> Jika ada pertanyaan, silakan disampaikan di kolom komentar.

---

## LAMPIRAN: TIPS REKAM

**Tools yang digunakan:**
1. **OBS Studio** (gratis, open-source) — untuk screen recording + webcam
2. **Slide Presentasi:** PowerPoint / Canva — export PNG untuk ditampilkan
3. **Aplikasi Streamlit:** Jalankan `streamlit run app/app.py` sebelum rekam

**Screenshot/Visual yang harus ditampilkan di video (urut):**
1. `reports/evaluation/roc_curve.png` — Slide 7
2. `reports/evaluation/confusion_matrix.png` — Slide 7
3. `reports/evaluation/pr_curve.png` — Slide 7
4. `reports/evaluation/score_distribution.png` — Slide 7
5. `reports/interpretation/attention_weights.png` — Slide 8
6. `reports/interpretation/feature_ablation.png` — Slide 8
7. `reports/interpretation/score_convergence.png` — Slide 8
8. `reports/interpretation/per_video_evolution.png` — Slide 8
9. `reports/model_comparison/shap_summary.png` — Slide 8
10. `reports/model_comparison/shap_feature_importance.png` — Slide 8
11. Screenshot Streamlit: `reports/screenshots/*.png` — Slide 9
12. **Demo langsung:** Buka http://localhost:8501, tunjukkin halaman Demo → predict video → Batch Test Set

**Pro Tips:**
- Gunakan Windows + G (Xbox Game Bar) untuk quick recording jika OBS terlalu berat
- Rekam dalam resolusi 1080p untuk kualitas slide terbaca
- Upload ke YouTube sebagai "Unlisted" → copy link → masukkan ke README.md dan laporan
- Durasi ideal: 12-15 menit (jangan lebih dari 15 menit agar tidak membosankan)
