# Problem Statement: Deteksi Kerusuhan Menggunakan Machine Learning

## 1. Latar Belakang Masalah

Indonesia sebagai negara demokrasi terbesar ketiga di dunia sering menghadapi
tantangan dalam menjaga keamanan dan ketertiban umum. Aksi demonstrasi yang
berujung kerusuhan, tawuran antar kelompok, dan tindak kekerasan massal
merupakan fenomena yang sering terjadi di berbagai wilayah. Data dari BPS
menunjukkan bahwa terjadi peningkatan jumlah demonstrasi di Indonesia setiap
tahunnya, di mana sebagian berlangsung damai namun sebagian lainnya berpotensi
menimbulkan kerusuhan.

Pemantauan keamanan melalui video CCTV saat ini masih sangat bergantung pada
pengawasan manual oleh operator manusia. Pendekatan ini memiliki beberapa
kelemahan signifikan: (1) operator manusia mudah lelah dan kehilangan fokus
setelah waktu pengawasan yang lama, (2) jumlah CCTV yang terus bertambah tidak
sebanding dengan jumlah operator yang tersedia, (3) respons terhadap insiden
menjadi lambat karena harus menunggu operator mendeteksi secara manual.

Oleh karena itu, diperlukan sistem otomatis yang dapat mendeteksi kerusuhan
dari video secara real-time untuk membantu aparat keamanan merespons dengan
lebih cepat dan tepat.

## 2. Tujuan Bisnis dan Analisis

Tujuan dari project ini adalah mengembangkan model machine learning yang mampu
mengklasifikasikan video apakah mengandung aktivitas kerusuhan atau tidak
secara otomatis. Model akan diintegrasikan ke dalam aplikasi Streamlit yang
dapat digunakan untuk demonstrasi dan pengujian secara interaktif.

Secara lebih spesifik, project ini bertujuan untuk:
- Mengekstrak fitur temporal dari video menggunakan S3D (Separable 3D CNN)
- Mengimplementasikan Attention-based Multiple Instance Learning (AttentionMIL)
  untuk klasifikasi video
- Membandingkan performa berbagai arsitektur MIL
- Menyediakan aplikasi demo yang dapat digunakan stakeholder non-teknis

## 3. Metrik Kesuksesan

Keberhasilan project ini diukur berdasarkan metrik berikut:

| Metrik | Target | Deskripsi |
|--------|--------|-----------|
| AUC | ≥ 0.90 | Kemampuan model membedakan kelas rusuh dan non-rusuh |
| F1 Score | ≥ 0.80 | Keseimbangan precision dan recall |
| Akurasi | ≥ 85% | Persentase prediksi benar |
| Waktu Inferensi | < 1 detik/video | Kecepatan prediksi untuk aplikasi real-time |

## 4. Dataset

Dataset yang digunakan dalam project ini berjumlah 5.552 video yang dikumpulkan
dari berbagai sumber:
- YouTube API: Video demo damai dan demo rusuh dari kanal publik
- Kaggle (RWF-2000): Dataset video kerusuhan internasional (2.000 video)
- SCVD (Smart City Violence Dataset): Dataset kekerasan perkotaan
- MSV-PG (HuggingFace): Multi-source Violence Dataset

Dataset dibagi menjadi 3 kelas: demo_rusuh, demo_damai, dan normal, yang
kemudian dipetakan menjadi binary classification (rusuh vs non-rusuh).
Pembagian data: 80% training, 10% validation, 10% testing.

## 5. Pendekatan Solusi

Solusi menggunakan pendekatan Multiple Instance Learning (MIL) di mana setiap
video dipandang sebagai "bag" yang berisi "instance" berupa segmen-segmen
video. Attention mechanism digunakan untuk mempelajari bobot kepentingan
setiap segmen secara otomatis. Feature extraction menggunakan S3D yang
di-pretrained pada Kinetics-400 menghasilkan vektor fitur 1024-dimensi untuk
setiap segmen 16 frame.
