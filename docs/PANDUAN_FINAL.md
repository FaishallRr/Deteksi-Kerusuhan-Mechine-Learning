# PANDUAN FINAL: Setup Repository & Deployment
## UAS Machine Learning - Deteksi Kerusuhan

---

## 1. GITHUB REPOSITORY (WAJIB — PUBLIC)

```bash
# Buat repo baru di github.com, lalu:
git remote add origin https://github.com/faishalrasyid/deteksi-kerusuhan-ml.git
git add .
git commit -m "Final: Deteksi Kerusuhan dengan AttentionMIL"
git branch -M main
git push -u origin main
```

---

## 2. STREAMLIT DEPLOYMENT (WAJIB)

### Cara deploy ke Streamlit Community Cloud (GRATIS):

1. **Push ke GitHub** (langkah 1 di atas)
2. Buka https://streamlit.io/cloud
3. Login dengan GitHub
4. Klik "New app"
5. Pilih repository: `faishalrasyid/deteksi-kerusuhan-ml`
6. Branch: `main`
7. Main file path: `app/app.py`
8. Klik "Deploy"

### ⚠️ PENTING: Sebelum deploy, set konfigurasi di `.streamlit/config.toml` (biar cepat):
```toml
[server]
maxUploadSize = 200
headless = true

[browser]
gatherUsageStats = false
```

### Setelah deploy berhasil:
- Dapat URL seperti: `https://deteksi-kerusuhan-ml.streamlit.app`
- Taruh URL di README.md

---

## 3. YOUTUBE VIDEO

### Step rekam:
1. Buka slide presentasi (PowerPoint/Canva)
2. Buka Streamlit app di browser
3. Rekam pake **OBS Studio** (gratis) atau **Windows + G** (Xbox Game Bar)
4. Baca naskah dari `docs/naskah_presentasi.md`

### Step upload:
1. Upload ke YouTube (unlisted / public)
2. Copy link
3. Taruh di README.md:
```markdown
## 📹 Video Presentasi
[Link YouTube](https://youtu.be/PASTE_LINK_HERE)
```

---

## 4. LAPORAN

File sudah digenerate:
- **DOCX:** `reports/Laporan_UAS_Deteksi_Kerusuhan.docx`
- Bisa langsung di-edit di Word atau convert ke PDF

Cover yang perlu diedit manual:
- Tambah nama anggota kelompok jika ada
- Sesuaikan tanggal

---

## 5. PENGUMPULAN VIA KULINO

Yang dikumpulkan:
1. **Link GitHub repository** (public)
2. **Link Streamlit deployment**
3. **Laporan (DOCX atau PDF)**
4. **Link YouTube**

---

## 6. FINAL CHECKLIST

| Item | Status | Keterangan |
|------|--------|------------|
| Problem statement (min 200 kata) | ✅ | `docs/problem_statement.md` |
| GitHub public repository | ⏳ | Lo buat + push |
| Streamlit deployment | ⏳ | Lo deploy |
| YouTube video | ⏳ | Lo rekam + upload |
| Laporan PDF/DOCX | ✅ | `reports/Laporan_UAS_Deteksi_Kerusuhan.docx` |
| EDA Notebook | ✅ | `notebooks/01_eda.ipynb` |
| ROC curve + CM + PR plots | ✅ | `reports/evaluation/` |
| Screenshots app | ✅ | `reports/screenshots/` |
| Naskah presentasi | ✅ | `docs/naskah_presentasi.md` |
| README.md informatif | ✅ | Ada |
| requirements.txt | ✅ | Ada |
| Model .pt + .pkl | ✅ | `models/mil_final.pt` |
| Interpretasi model | ✅ | `reports/interpretation/` |

---

## 7. PERKIRAAN NILAI (kalo semua beres)

| Kriteria | Bobot | Target |
|----------|-------|--------|
| Kualitas & Kelengkapan Analisis | 25% | ✅ EDA notebook + Streamlit dashboard |
| Kualitas Pemodelan | 30% | ✅ AUC 0.9563, 2 models, tuning, interpretasi |
| Fungsionalitas Deployment | 20% | ⏳ Perlu deploy ke cloud |
| Dokumentasi & Presentasi | 15% | ⏳ Perlu upload YouTube + laporan |
| Originalitas & Kompleksitas | 10% | ✅ AttentionMIL, video classification |

**Estimasi nilai akhir: A (85-95)** kalo semua ⏳ diisi.
