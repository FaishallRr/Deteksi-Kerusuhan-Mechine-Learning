# 🚨 Sistem Deteksi Kerusuhan & Anomali — CCTV Real-time

Deteksi kerusuhan, kekerasan, dan anomali dari **CCTV Semarang live 24/7** menggunakan **YOLO11m ONNX + CUDA** dengan dashboard **Streamlit** real-time.

---

## Arsitektur Sistem

```
┌─────────────────┐     HLS (m3u8)      ┌──────────────────┐
│  CCTV Semarang  │ ──────────────────> │  Browser (hls.js)│
│  (54 kamera)    │                     │  + Canvas Overlay│
└─────────────────┘                     └────────┬─────────┘
                                                 │
                                          capture frame
                                          (JPEG, 640×360)
                                                 │
                                                 v
┌──────────────────┐     WebSocket        ┌──────────────────┐
│  ws_detect_server │ <───────────────── │  Streamlit App   │
│  :8765            │   JSON detection    │  :8501           │
│  YOLO11m ONNX     │                     │  Multi-camera    │
│  + Indo Weapon    │                     │  HUD overlay     │
│  + Sajam CNN      │                     │  Status dashboard│
└──────────────────┘                     └──────────────────┘
```

### Komponen Utama

| Komponen | Port | Deskripsi |
|----------|------|-----------|
| **Streamlit Dashboard** | `:8501` | Multi-camera grid, HUD overlay, canvas tracking, pilih CCTV via sidebar |
| **WebSocket Server** | `:8765` | YOLO11m inference, anomaly scoring, per-frame detection JSON |
| **hls.js (browser)** | — | Play HLS stream, capture frame → kirim ke WS → render bounding box overlay |

---

## Fitur

- **Live HLS CCTV** — 42 kamera Semarang aktif, play via hls.js
- **Real-time Detection** — YOLO11m ONNX FP16 ~45fps (CUDA)
- **Anomaly Scoring** — 5 komponen: weapon(30%) + crowd(20%) + speed(15%) + proximity(20%) + vehicle(15%)
- **Status HUD** — 🟢 NORMAL / 🟠 MENCURIGAKAN / 🔴 BAHAYA
- **Box Tracking** — Class-specific (mobil 0.15–0.40, motor 0.28–0.58), velocity EMA smoothing, zombie revival, duplicate cleanup
- **Multi-camera** — Pilih beberapa CCTV via sidebar, masing-masing di grid terpisah
- **Vehicle Classification** — YOLO heuristic refinement (bicycle→motorcycle, car/truck/bus size-based correction)
- **Indo Weapon Detection** — Model khusus senjata Indonesia (Celurit, Golok, Kapak, Pedang, Pisau, Pistol, Senapan)
- **Sajam CNN Verifier** — Binary classifier untuk verifikasi senjata tajam
- **File Mode** — Deteksi dari file video MP4 untuk simulasi

---

## Progress

### ✅ Selesai

| Area | Detail |
|------|--------|
| **YOLO11m ONNX FP16** | Pipeline ~22ms/frame (~45fps). Conf threshold 0.15 untuk deteksi jarak jauh |
| **imgsz=800** | Auto-fallback ke 640 jika ONNX reject shape. Objek kecil 1.56x lebih besar |
| **Enhanced Anomaly** | Person velocity tracking, crowd density, weapon proximity, vehicle anomaly (5 komponen scoring) |
| **JS Tracking** | Class-specific smoothing: mobil alpha 0.15–0.40, motor alpha 0.28–0.58. Velocity EMA, zombie revival (100px), duplicate cleanup (60px) |
| **Motorcycle Tracking** | Stabil — velocity EMA 50/50, predict 8%, alpha 0.28–0.58, size smoothing 0.15 |
| **Vehicle Classifier** | YOLO heuristic refinement: threshold realistic (motor→car only if >12000px), bicycle→motorcycle, car→truck if >20000px |
| **CCTV Sources** | 42 kamera hidup (12 mati dihapus). Format HLS m3u8 via `livepantau.semarangkota.go.id` |
| **HLS Error Handling** | 3x retry, timeout config, "Camera offline" setelah gagal |
| **WebSocket Server** | Asyncio-based, multiple clients, binary JPEG → JSON detection |
| **MIL Temporal Model** | Training final — ROC-AUC 0.9976, F1 0.9865 (file mode) |

### 🔄 Dalam Proses

- Sajam/weapon detection refinement

### 📋 Next Steps

- [ ] Sajam CNN — tuning threshold, reduce false positive
- [ ] Alert refinement — fight detection, weapon temporal confirmation
- [ ] Notifikasi Telegram/WhatsApp
- [ ] Test streaming panjang — FPS, memory, WS stability
- [ ] Auto camera health check (periodic ping)

---

## Cara Menjalankan

### Prasyarat

- **Python 3.14.4** di `C:\Python314\python.exe`
- **GPU RTX 4050 6GB** dengan CUDA 12.8
- **ONNX Runtime dengan CUDAExecutionProvider**
- **Chrome/Edge** (hls.js membutuhkan MSE)

### 1. Install Dependencies

```powershell
C:\Python314\python.exe -m pip install -r requirements.txt
```

Dependencies utama: `streamlit`, `onnxruntime-gpu`, `numpy`, `opencv-python`, `pyyaml`, `ultralytics`, `websockets`, `torch`, `pillow`.

### 2. Jalankan Dashboard

```powershell
C:\Python314\python.exe -m streamlit run app.py
```

- Dashboard: `http://localhost:8501`
- WebSocket server otomatis start di `ws://localhost:8765`
- YOLO model lazy-loaded saat frame pertama masuk (~800MB → ~2.1GB VRAM)

### 3. Pilih CCTV

1. Sidebar → **Sumber Video** → pilih **CCTV Langsung**
2. **Pilih Kamera** → centang 1 atau lebih CCTV
3. Tunggu stream HLS load (~5–10 detik)
4. Bounding box + anomaly score muncul otomatis

### Mode File (Simulasi)

```powershell
# Set config.yaml general.mode: "file", atau
# Pilih "File Video" di sidebar
# Upload file MP4 atau pilih dari sample_videos/
```

---

## Struktur File Penting

| File | Fungsi |
|------|--------|
| `app.py` | Streamlit dashboard, HLS HTML builder, JS tracking code, multi-camera grid |
| `ws_detect_server.py` | WebSocket server, YOLODetector, anomaly scoring, NMS |
| `core/yolo_detector.py` | YOLO11m ONNX + Indo weapon + Sajam CNN verifier |
| `cctv_sources.py` | 42 CCTV Semarang URL (HLS m3u8) |
| `inference.py` | AnomalyDetector class — MIL temporal fusion untuk file mode |
| `config.yaml` | Konfigurasi utama (threshold, model path, alert settings) |
| `core/sajam_cnn_verify.pt` | CNN verifier untuk senjata tajam (64×64 input) |
| `yolo11m.onnx` | YOLO11m ONNX FP16 model |

---

## Konfigurasi Threshold

`config.yaml`:

```yaml
thresholds:
  warning: 0.6
  alert: 0.8
  confirmation_frames: 10
```

Di JS tracking (app.py):

| Parameter | Mobil | Motor | Fungsi |
|-----------|-------|-------|--------|
| `IOU_THRESH` | 0.20 | 0.20 | IoU minimum match track-detection |
| `MAX_MISSED` | 5 | 5 | Track dihapus setelah 5 frame tanpa match |
| `NEW_TRACK_MIN_CONF` | 0.25 | 0.25 | Conf minimum untuk track baru |
| `ZOMBIE_REVIVE_DIST` | 10000 | 10000 | Radius revival track lama (100px) |
| Velocity EMA | 60% lama | 50% lama | Smoothing velocity update |
| Alpha range | 0.15–0.40 | 0.28–0.58 | Position smoothing adaptif |
| Size smoothing | 0.12 | 0.15 | Box size adaptation |
| Predict tracks | 6% velocity | 8% velocity | Prediksi posisi antar-frame |
| Age<3 alpha | 0.55 | 0.55 | Track baru converge cepat |

---

## Troubleshooting

### CCTV tidak muncul (0 frames)

1. Buka browser console (F12) — cek error JS
2. Pastikan `st.components.v1.html` masih berfungsi (deprecated setelah 2026-06-01)
3. Cek apakah kamera masih hidup:
   ```powershell
   curl -I https://livepantau.semarangkota.go.id/{uuid}/index.m3u8
   ```
4. Restart Streamlit: `Ctrl+C` → `streamlit run app.py`

### WebSocket tidak connect

- Pastikan WS server port 8765 tidak dipakai
- Cek log: `[DetectServer] WebSocket on ws://0.0.0.0:8765`
- Firewall mungkin blok localhost connection

### GPU memory full

- YOLO11m ONNX + CUDA ~2.1GB VRAM
- Jika OOM: turunkan `imgsz` ke 640 (fallback otomatis)
- Atau ganti model: `yolo11s.onnx` (lebih ringan)

---

## Catatan Teknis

- **Python path**: HARUS `C:\Python314\python.exe` — `python` resolve ke Miniconda3 (tanpa streamlit)
- **Server HEAD ditolak**: CCTV Semarang respond 405 pada HEAD request — hanya GET
- **st.components.v1.html deprecated**: Akan dihapus setelah 2026-06-01. Migrasi ke `st.iframe` direncanakan
- **Segmen HLS ~4 detik**: CCTV Semarang pakai segmen panjang bukan real-time chunked
- **Model path**: YOLO model `yolo11m.onnx` di root project, bukan di `models/`
