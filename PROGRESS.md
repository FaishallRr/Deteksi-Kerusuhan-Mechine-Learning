# 📊 PROGRESS — Deteksi Kerusuhan & Anomali

## Fase 1: Setup ✅ (11 Juni 2026)
- [x] Buat struktur direktori proyek
- [x] Buat `config.yaml` — semua parameter sistem
- [x] Buat `requirements.txt` — dependencies lengkap
- [x] Buat `.gitignore` — ignore models, evidence, dsb
- [x] Buat skeleton file Python:
  - `inference.py` — main detection loop
  - `api_service.py` — FastAPI backend
  - `app.py` — Streamlit dashboard
  - `core/mil_ranking.py` — MIL Ranking model
  - `core/yolo_detector.py` — YOLOv11 wrapper
  - `preprocessing/extract_frames.py` — frame extraction
  - `preprocessing/feature_extractor.py` — X3D features
  - `alert/telegram_bot.py` — Telegram alert
  - `alert/whatsapp_sender.py` — WhatsApp alert
  - `utils/config_loader.py`, `logger.py`, `evidence.py`
- [x] Buat `Dockerfile`
- [x] Buat `PROGRESS.md`

## Fase 2: Preprocessing (Next)
- [ ] Download UCF-Crime sample
- [ ] Testing frame extraction pipeline
- [ ] Testing X3D feature extraction

## Fase 3: MIL Ranking (Next)
- [ ] Training MIL model
- [ ] Evaluasi baseline (CNN+SVM)

## Fase 4: YOLO Integration (Next)
- [ ] Integrasi YOLOv11 + fusion score

## Fase 5: Alert System (Next)
- [ ] Setup Telegram bot token
- [ ] Setup WhatsApp

## Fase 6: Dashboard & API (Next)
- [ ] Finalisasi Streamlit
- [ ] Finalisasi FastAPI
- [ ] Docker compose

## Fase 7: Dokumentasi (Next)
- [ ] Semua file docs/
