# 📊 PROGRESS — Deteksi Kerusuhan & Anomali

## Fase 1: Setup ✅
- [x] Struktur direktori, config.yaml, requirements, Dockerfile, git init

## Fase 2: Preprocessing Pipeline ✅
- [x] FrameExtractor, S3D FeatureExtractor, pipeline lengkap

## Fase 3: MIL Ranking Training (Synthetic) ✅
- [x] Training dengan 10 video sintetis → akurasi 100%

## Fase 4: YOLOv11 Integration ✅
- [x] YOLOv11-nano, face detection, LPR, score fusion, 5-layer anti false positive

## Fase 5: Alert System ✅
- [x] Telegram + WhatsApp (graceful mock mode)

## Fase 6: Deployment ✅
- [x] FastAPI, Streamlit dashboard, Dockerfile

## ✅ FASE 7: TRAINING DENGAN UCF-CRIME REAL ✅
- [x] Download UCF-Crime dataset (12GB) dari Kaggle
- [x] Preprocessing: 156 video → 3.458 S3D feature vectors (1024-d)
- [x] Training MIL Ranking (500 epoch, early stopping at epoch 61)

### Hasil Training dengan UCF-Crime
| Metrik | Hasil |
|---|---|
| **ROC-AUC** | **0.9984** |
| **Precision** | **0.9860** |
| **Recall** | **0.9832** |
| **F1-Score** | **0.9846** |
| **Accuracy** | **0.9788** |
| False Positive | 5 dari 161 normal |
| Missed Anomaly | 6 dari 358 anomaly |

### Confusion Matrix (Test Set)
```
                Predicted
                Normal  Anomaly
Actual Normal      156      5
       Anomaly       6    352
```

### Inference Test (Real UCF-Crime Videos)
| Video | Score | Correct |
|---|---|---|
| Fighting | 0.9846 | ✅ |
| Assault | 1.0000 | ✅ |
| Normal | 0.9446 | ❌ (1 FP) |
| Robbery | 0.9905 | ✅ |
| Shooting | 1.0000 | ✅ |

## 📁 Struktur Final
```
deteksi-kerusuhan/
├── config.yaml                    # All configs
├── inference.py                   # Main detection (YOLO + S3D + MIL)
├── api_service.py                 # FastAPI
├── app.py                         # Streamlit dashboard
├── preprocessing/                 # Frame extraction, S3D features, UCF preprocessing
├── core/                          # MIL ranking, YOLO detector, training
├── alert/                         # Telegram + WhatsApp
├── utils/                         # Config, logger, evidence
├── models/
│   ├── mil_model.pt              # Trained on synthetic
│   ├── mil_model_ucf.pt         # Trained on UCF-Crime (BEST)
│   ├── evaluation_results.json
│   └── training_history.png
├── features/ucf_crime/           # Precomputed features
├── ucf_crime_raw/                # Raw UCF-Crime dataset
├── sample_videos/                # Test videos
├── evidence/                     # Alert evidence
└── docs/                         # Documentation
```

## ⬜ NEXT
- [ ] Scraping YouTube Indonesia untuk fine-tuning konteks lokal
- [ ] Dokumentasi lengkap (docs/*.md untuk dosen)
- [ ] Deploy Streamlit dashboard
- [ ] Setup actual Telegram bot token
