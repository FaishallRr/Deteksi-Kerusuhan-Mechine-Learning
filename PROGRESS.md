# PROGRESS — Deteksi Kerusuhan & Anomali (CCTV)

## Fase ML Temporal (MIL Model) ✅

| Stage | Status | ROC-AUC | F1 |
|-------|--------|---------|----|
| UCF-Crime Only | ✅ Selesai | 0.9984 | 0.9846 |
| Combined (UCF+SCVD+Indo) | ⚠️ Contaminated | — | — |
| Clean Training v2 | ✅ **FINAL** | **0.9976** | **0.9865** |

**Model**: `models/mil_model_v2_clean.pt` — digunakan di `inference.py` untuk file mode.

## Fase Real-time CCTV (ONNX + Streamlit) ✅

### Tracking Status

| Objek | Status | Detail |
|-------|--------|--------|
| 🚗 Mobil | ✅ **Smooth, akurat** | Alpha 0.15–0.40, velocity EMA 60/40 |
| 🏍️ Motor | ✅ **Responsif, stabil** | Alpha 0.28–0.58, velocity EMA 50/50, predict 8% |
| 🚚 Truck/Bus | ✅ **Akurat** | Class-size heuristic refinement |
| 🚶 Person | ✅ | Detection + velocity tracking |

### Vehicle Classification

- YOLO11m COCO + heuristic refinement
- Bicycle → selalu dikonversi ke motorcycle (konteks Indonesia)
- Motorcycle → car hanya jika area > 12000px (fisik impossible)
- Car → truck hanya jika area > 20000px + ratio > 2.0
- Tidak ada catch-all `conf < 0.5 → vehicle` (label asli dipertahankan)

### CCTV Pipeline

| Komponen | Detail |
|----------|--------|
| **CCTV Active** | 42 kamera (12 mati dihapus Juni 2026) |
| **Stream** | HLS m3u8 via `livepantau.semarangkota.go.id` |
| **Player** | hls.js (Chrome/Edge) |
| **Transport** | WebSocket ws://localhost:8765 |
| **Model** | YOLO11m ONNX FP16, CUDA, ~45fps |
| **Dashboard** | Streamlit 1.58.0, multi-camera grid |

## Next (Prioritas)

1. **Sajam/Weapon Detection** — tuning threshold sajam CNN, reduce false positive, conf interval
2. **Alert System** — Telegram + WhatsApp notification
3. **Fight Detection** — temporal confirmation untuk kerusuhan
4. **Long-run Stability** — FPS/memory monitoring, auto camera health check
