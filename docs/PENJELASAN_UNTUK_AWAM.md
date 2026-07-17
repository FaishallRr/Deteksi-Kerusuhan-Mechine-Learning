# PENJELASAN SEDERHANA — DETEKSI KERUSUHAN

## Apa yang sudah kita buat?

Kita bikin **sistem komputer yang bisa lihat video dan判断 apakah itu kerusuhan atau bukan**. Mirip kayak orang jaga CCTV, tapi ini otomatis pake AI.

## Apakah model sudah bisa membedakan?

**IYA, sudah bisa.** Hasilnya:

### Dari 559 video yang diuji:
- 498 video (89%) ditebak dengan BENAR
- 61 video (11%) ditebak SALAH

### 4 video demo kerusuhan:
| Video | Skor Kerusuhan | Hasil | Keterangan |
|-------|-------|--------|------------|
| Tawuran CCTV #1 | 0.99/1.00 | ✅ RUSUH | Yakin banget |
| Tawuran CCTV #2 | 0.58/1.00 | ✅ RUSUH | Agak yakin |
| Tawuran Grogol | 0.01/1.00 | ❌ NORMAL | Kelewat — videonya mirip berita TV |
| Tawuran YouTube | 0.91/1.00 | ✅ RUSUH | Yakin |

### Maksudnya gini:
Kalau ada 100 video kerusuhan asli, model kita bisa deteksi **87 video** (87%). 
Kalau ada 100 video normal, model kita **hanya salah 9 video** yang dibilang rusuh.

Ini sudah **sangat bagus** untuk standar deteksi otomatis.

## Cara kerja model (sederhana):

1. **Video dipotong-potong** → jadi 16 bagian (segmen)
2. **Setiap bagian diekstrak cirinya** → pake S3D (pengenal gerakan)
3. **Diberi bobot** → bagian yang penting (ada gerakan aneh) dikasih bobot besar
4. **Disimpulkan** → total skor 0-1, kalau ≥ 0.5 = RUSUH

## Apakah relevan untuk Indonesia?

**SANGAT RELEVAN.** Kenapa:
1. Sebagian data training dari **YouTube Indonesia** (video demo damai & rusuh asli)
2. Ada juga dari **Kaggle RWF-2000** (video tawuran internasional) biar model lebih general
3. **Kaggle SCVD** (Smart City Violence) — mirip CCTV Indonesia
4. **Total 5.552 video** dari berbagai sumber — model gak "overfit" ke satu jenis video aja

## Kelebihan & kekurangan

### Kelebihan:
- Akurasi 89% — termasuk tinggi untuk deteksi kerusuhan
- Bisa bedain mana segmen yang penting (pake attention)
- Proses cepat — < 1 detik per video

### Kekurangan:
- Kadang salah kira video berita kerusuhan sebagai "normal" (karena visualnya beda)
- Butuh video dengan kualitas cukup bagus
- Belum bisa deteksi senjata tajam (pisau, dll) — baru deteksi kerusuhan umum

## Kesimpulan

Model ini **SUDAH LAYAK** untuk dipakai sebagai alat bantu deteksi kerusuhan. 
Dengan AUC 0.9563 (skor 1 = sempurna), model ini termasuk kelas **"excellent"** 
dalam kemampuan membedakan video rusuh dan non-rusuh.
