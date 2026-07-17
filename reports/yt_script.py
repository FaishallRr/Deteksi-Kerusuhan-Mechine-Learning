"""
Naskah Presentasi YouTube -- Deteksi Kerusuhan Menggunakan AttentionMIL
UAS Machine Learning | Universitas Dian Nuswantoro
Faishal Rasyid Rusianto - A11.2024.15869
Durasi: ~15 menit (11 slide)
"""

import json
from pathlib import Path

BASE = Path(__file__).resolve().parent.parent

meta_path = BASE / "features/final_dataset/metadata.json"
metrics_path = BASE / "reports/evaluation/metrics.json"

meta = json.load(open(meta_path)) if meta_path.exists() else []
metrics = json.load(open(metrics_path)) if metrics_path.exists() else {}

total = len(meta)
rusuh = sum(1 for m in meta if m["label"] == 1)
normal = total - rusuh

AUC = metrics.get("auc", 0.9563)
ACC = metrics.get("accuracy", 0.8909)
F1 = metrics.get("f1", 0.8683)
PREC = metrics.get("precision", 0.8627)
REC = metrics.get("recall", 0.8739)
FP = metrics.get("false_positives", 32)
FN = metrics.get("false_negatives", 29)
CM = metrics.get("cm", [[297, 32], [29, 201]])
TN, TP = CM[0][0], CM[1][1]

FP_PCT = FP / (TN + FP) * 100
FN_PCT = FN / (TP + FN) * 100
RUSUH_PCT = rusuh / total * 100
NORMAL_PCT = normal / total * 100

SPLITS = {}
for m in meta:
    s = m.get("split", "?")
    SPLITS[s] = SPLITS.get(s, 0) + 1

print("=" * 70)
print("  NASKAH PRESENTASI YOUTUBE")
print("  Deteksi Kerusuhan Menggunakan AttentionMIL")
print("  UAS Machine Learning - Universitas Dian Nuswantoro")
print("  Faishal Rasyid Rusianto - A11.2024.15869")
print(f"  Dataset: {total} video | AUC: {AUC:.4f} | Akurasi: {ACC:.2%}")
print("=" * 70)

SLIDES = f"""

--------------------------------------------------------------------------------
 SLIDE 1 - PEMBUKAAN (0:00 - 0:30)
--------------------------------------------------------------------------------

Assalamualaikum Warahmatullahi Wabarakatuh.

Perkenalkan, nama saya Faishal Rasyid Rusianto, NIM A11.2024.15869,
dari Program Studi Teknik Informatika, Universitas Dian Nuswantoro.

Pada kesempatan ini, saya akan mempresentasikan project UAS Machine Learning
saya yang berjudul "Deteksi Kerusuhan Menggunakan Attention-based Multiple
Instance Learning".

Project ini merupakan solusi end-to-end berbasis machine learning untuk
mendeteksi kerusuhan dari video secara otomatis, mencakup seluruh pipeline
mulai dari akuisisi data, preprocessing, pemodelan, evaluasi, hingga
deployment ke aplikasi web interaktif.

------------------------------------------------------------------------
 SLIDE 2 -- LATAR BELAKANG & RUMUSAN MASALAH (0:30 - 1:30)
------------------------------------------------------------------------

Latar Belakang:
Indonesia sebagai negara demokrasi sering menghadapi demonstrasi dan
kerusuhan massa. Pemantauan CCTV secara manual memiliki keterbatasan:
operator manusia mudah lelah, jumlah CCTV terus bertambah, dan respons
terhadap insiden sering terlambat.

Rumusan Masalah:
1. Bagaimana mendeteksi kerusuhan dari video secara otomatis?
2. Algoritma apa yang paling tepat?
3. Bagaimana performa model?
4. Bagaimana menyajikan solusi dalam aplikasi Streamlit?

Tujuan bisnis: Membantu penegak hukum merespons kerusuhan lebih cepat.
Metrik kesuksesan: AUC > 0.90, Accuracy > 85%, F1 Score > 0.85.

------------------------------------------------------------------------
 SLIDE 3 -- DATASET & SUMBER DATA (1:30 - 2:30)
------------------------------------------------------------------------

Total: {total:,} video
Rusuh: {rusuh:,} video ({RUSUH_PCT:.0f}%)
Normal/Damai: {normal:,} video ({NORMAL_PCT:.0f}%)

Sumber Data:
1. UCF Crime Dataset (Kaggle): ~3.800 video
2. SCVD (Surveillance Camera Violence Dataset): ~1.600 video
3. MSV-PG (HuggingFace): 252 video
4. YouTube & Instagram: ~500 video tawuran Indonesia
5. Real Life Nonviolence: ~900 video

Dataset multi-source memastikan generalisasi model yang baik.

------------------------------------------------------------------------
 SLIDE 4 -- PREPROCESSING PIPELINE (2:30 - 5:00)
------------------------------------------------------------------------

Langkah 1 -- Frame Extraction (4 FPS):
- 30 FPS: mahal, redundant
- 1 FPS: kehilangan informasi temporal
- 4 FPS: keseimbangan optimal

Langkah 2 -- Segmentasi Temporal (16 frame, stride 8):
- 16 frame = ~4 detik video
- Stride 8 (50% overlap) -- momen kritis tidak terpotong

Langkah 3 -- Feature Extraction dengan S3D:
- Pretrained pada Kinetics-400 (transfer learning)
- Output: vektor 1024-d per segmen

Langkah 4 -- Augmentasi:
- Weather augmentation: brightness, noise, blur

------------------------------------------------------------------------
 SLIDE 5 -- EXPLORATORY DATA ANALYSIS (5:00 - 6:15)
------------------------------------------------------------------------

5 Insight penting dari EDA:

1. Distribusi Label: Rusuh {RUSUH_PCT:.0f}%, Normal {NORMAL_PCT:.0f}%
   Imbalance moderat -> AUC sebagai metrik utama

2. Multi-Source = Multi-Domain:
   Setiap sumber punya karakteristik visual berbeda

3. PCA: Separabilitas terbatas di 2D (~20% variance)

4. t-SNE: Cluster lebih terpisah

5. Rata-rata video punya 8-12 segmen (32-48 detik)

Visualisasi interaktif di halaman EDA Streamlit.

------------------------------------------------------------------------
 SLIDE 6 -- SPLIT & MODEL ARCHITECTURE (6:15 - 8:00)
------------------------------------------------------------------------

Split: 80/10/10
- Train: {SPLITS.get('train', 4440):,} video
- Val: {SPLITS.get('val', 553):,} video
- Test: {SPLITS.get('test', 559):,} video
Split dilakukan stratified.

Perbandingan 3 model:

1. XGBoost (Baseline - Ensemble Learning)
   - Mean pooling fitur -> 1024-d per video
   - GridSearchCV: max_depth=4, lr=0.1

2. AttentionMIL (Model Utama)
   Arsitektur:
   - Input: 16 segmen x 1024-d
   - Attention Network: Linear(1024->256) + Tanh + Linear(256->1) + Softmax
   - Weighted Sum -> Bag Representation (1024-d)
   - MLP Classifier: 1024->256->128->1 + ReLU + Dropout(0.3)
   - Sigmoid -> Anomaly Score [0,1]
   - Total parameter: 558.082

Mengapa AttentionMIL?
- Multiple Instance Learning: tidak perlu label per-frame
- Attention: model fokus ke segmen kerusuhan
- Dropout mencegah overfitting

------------------------------------------------------------------------
 SLIDE 7 -- HASIL EVALUASI (8:00 - 10:30)
------------------------------------------------------------------------

Evaluasi pada {SPLITS.get('test', 559)} video test:

Tabel Perbandingan:
| Model        | AUC    | Accuracy | F1     |
| XGBoost      | 0.9440 | 87.30%   | 0.8426 |
| AttentionMIL | {AUC:.4f} | {ACC:.2%}   | {F1:.4f} |

AttentionMIL unggul di SEMUA metrik.

ROC Curve:
- AUC = {AUC:.4f}
- 95.63% kemungkinan model memberi skor lebih tinggi ke video rusuh

Confusion Matrix:
- TN = {TN} (model benar bilang NORMAL)
- FP = {FP} (false alarm {FP_PCT:.1f}%)
- FN = {FN} (kerusuhan terlewat {FN_PCT:.1f}%)
- TP = {TP} (model benar deteksi RUSUH)

Score Distribution:
- Normal dominan di skor 0.0-0.2
- Rusuh dominan di skor 0.7-1.0
- Overlap minimal -> threshold 0.5 sudah tepat

------------------------------------------------------------------------
 SLIDE 8 -- INTERPRETASI MODEL (10:30 - 12:30)
------------------------------------------------------------------------

1. Attention Weights:
   - Video rusuh: bobot tidak merata -> fokus ke segmen abnormal
   - Video normal: bobot lebih merata

2. Feature Ablation:
   - Hapus 1 segmen -> ukur perubahan skor
   - Segmen akhir lebih penting

3. Score Convergence:
   - 2-4 segmen: fluktuatif
   - 8-10 segmen: mulai stabil
   - 12+ segmen: mendekati final
   - Prediksi akurat hanya dengan ~32 detik video!

4. SHAP Analysis (XGBoost):
   - Summary plot: feature contribution per sample
   - Bar plot: 15 fitur S3D paling penting

------------------------------------------------------------------------
 SLIDE 9 -- STREAMLIT DASHBOARD (12:30 - 14:00)
------------------------------------------------------------------------

Aplikasi Streamlit dengan 5 halaman:

1. Beranda: Metrik utama, info dataset, arsitektur model
2. EDA: Distribusi label, sumber data, PCA, t-SNE
3. Demo Model: Video Demo (4 video), Feature Demo (559 items), Batch Eval
4. Evaluasi & Interpretasi: ROC, CM, PR curve, attention, SHAP
5. Dokumentasi: Dataset, metodologi, cara pakai, referensi

Deployment:
https://deteksi-kerusuhan-mechine-learning-gevsrhre9uxyornmbwzy8z.streamlit.app/

------------------------------------------------------------------------
 SLIDE 10 -- KESIMPULAN & SARAN (14:00 - 15:00)
------------------------------------------------------------------------

Kesimpulan:
1. Deteksi otomatis berhasil: AUC {AUC:.4f}, Accuracy {ACC:.2%}
2. AttentionMIL unggul vs XGBoost di semua metrik
3. AUC {AUC:.4f} melampaui target > 0.90
4. Model bisa bedakan rusuh vs normal dengan sangat baik
5. Streamlit app siap digunakan

Saran pengembangan:
- Integrasi deteksi senjata tajam (multimodal)
- Deployment real-time pada aliran CCTV
- Model distillation untuk perangkat edge
- Dataset diperluas dengan variasi Indonesia

------------------------------------------------------------------------
 SLIDE 11 -- PENUTUP (15:00 - 15:30)
------------------------------------------------------------------------

Link Penting:
- GitHub: https://github.com/FaishallRr/Deteksi-Kerusuhan-Mechine-Learning
- Streamlit: https://deteksi-kerusuhan-mechine-learning-gevsrhre9uxyornmbwzy8z.streamlit.app/

Referensi:
1. Ilse et al. (2018). Attention-based Deep MIL. ICML.
2. Xie et al. (2018). S3D: Rethinking Spatiotemporal Feature Learning. ECCV.
3. Chen & Guestrin (2016). XGBoost. KDD.
4. Lundberg & Lee (2017). SHAP. NeurIPS.

Sekian presentasi dari saya. Terima kasih atas perhatiannya.
Wassalamualaikum Warahmatullahi Wabarakatuh.

------------------------------------------------------------------------
 TIPS REKAM
------------------------------------------------------------------------
1. Gunakan OBS Studio untuk screen recording
2. Siapkan slide (PowerPoint/Canva) - export PNG
3. Jalankan `streamlit run app/app.py` sebelum rekam
4. Tampilkan screenshot di folder reports/screenshots/
5. Rekam 1080p, upload ke YouTube (unlisted)
6. Durasi ideal: 12-15 menit
"""

print(SLIDES)
