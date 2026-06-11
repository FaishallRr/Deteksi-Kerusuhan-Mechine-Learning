# 📊 PROGRESS — Deteksi Kerusuhan & Anomali

## Fase 1: Setup ✅ (11 Juni 2026)
- [x] Struktur direktori proyek lengkap
- [x] `config.yaml` — semua parameter sistem
- [x] `requirements.txt` — dependencies lengkap
- [x] `.gitignore` + git init
- [x] Skeleton semua modul Python
- [x] `Dockerfile`

## Fase 2: Preprocessing Pipeline ✅ (11 Juni 2026)
- [x] `FrameExtractor` — ekstraksi frame dari video
- [x] `TemporalFeatureExtractor` — S3D feature extraction (pengganti X3D)
- [x] `generate_test_video.py` — generator video sintetis
- [x] `generate_dataset.py` — dataset sintetis (5 normal + 5 anomaly)
- [x] `PreprocessingPipeline` — pipeline end-to-end
- [x] Testing: 30 frame video → 1 segmen → 1024-d feature vector ✅

## Fase 3: MIL Ranking Training ✅ (11 Juni 2026)
- [x] `MILRankingModel` — classifier + ranking loss
- [x] `MILBagProcessor` — inference wrapper
- [x] `train_mil.py` — training script
- [x] Training dengan 10 video sintetis:
  - Normal: avg score 0.20
  - Anomaly: avg score 0.80
  - Test accuracy: 100%
- [x] Model saved: `models/mil_model.pt`

## Fase 4: YOLO Integration ✅ (11 Juni 2026)
- [x] `YOLODetector` — YOLOv11-nano wrapper
- [x] Face detection (retinaface)
- [x] License plate OCR (easyocr)
- [x] Score fusion (MIL 60% + Objek 40%)
- [x] 5-layer anti false positive:
  - Temporal smoothing ✅
  - Threshold 0.8 ✅
  - 10 frame confirmation ✅
  - Object validation ✅
  - Cooldown + rate limit ✅
- [x] Headless test pipeline ✅
- [x] YOLOv11-nano auto-download ✅

## Fase 5: Alert System ✅ (11 Juni 2026)
- [x] `TelegramAlert` — kirim ke Telegram grup
- [x] `WhatsAppAlert` — kirim ke WhatsApp
- [x] Graceful handling jika API key belum diisi
- [x] Mock mode untuk testing tanpa API key

## Fase 6: Deployment ⬜ (Siap)
- [x] `api_service.py` — FastAPI + /health + /detect endpoint
- [x] `app.py` — Streamlit dashboard interaktif
- [x] `Dockerfile` — container ready
- [ ] Test Streamlit dashboard (perlu sample video real)

## Fase 7: Dokumentasi ⬜
- [ ] docs/01-bab1-pendahuluan.md
- [ ] docs/02-bab2-landasan-teori.md
- [ ] docs/03-bab3-metodologi.md
- [ ] docs/04-bab4-implementasi.md
- [ ] docs/05-bab5-deployment.md
- [ ] docs/06-bab6-penutup.md

## 📈 Hasil Testing Pipeline
| Video | Avg Score | Alert Triggered | Status |
|---|---|---|---|
| test_normal.mp4 (30f) | 0.20 | ❌ | ✅ Normal |
| test_normal_long.mp4 (50f) | 0.20 | ❌ | ✅ Normal |
| test_anomaly.mp4 (30f) | 0.80 | ❌ (too short) | ✅ Terdeteksi anomaly |
| test_anomaly_long.mp4 (50f) | 0.80 | ❌ (threshold 0.8, butuh 10 consecutive) | ✅ Terdeteksi anomaly |
| normal_00-04.mp4 (5 vids) | 0.20 | ❌ | ✅ Normal |
| anomaly_00-04.mp4 (5 vids) | 0.80 | ❌ (need longer video for 10 frame confirmation) | ✅ Terdeteksi anomaly |
