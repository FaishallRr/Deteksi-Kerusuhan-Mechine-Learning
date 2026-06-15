import streamlit as st
import cv2
import numpy as np
from pathlib import Path
from datetime import datetime
from collections import deque
import tempfile
import json
import threading

from ws_detect_server import start_ws_server as _start_ws

# Start detection WebSocket server (global, once per process)
if not getattr(threading, "_detect_ws_started", False):
    threading._detect_ws_started = True
    try:
        t = threading.Thread(target=_start_ws, daemon=True)
        t.start()
    except Exception as e:
        print(f"[WARN] Could not start detect WS server: {e}")

from inference import AnomalyDetector
from utils.config_loader import load_config
from cctv_sources import CCTV_NAMES

st.set_page_config(
    page_title="Sistem Deteksi Kerusuhan",
    page_icon="🚨",
    layout="wide",
)

st.title("🚨 Sistem Deteksi Kerusuhan & Anomali")
st.markdown("---")

config = load_config()

col1, col2, col3, col4 = st.columns(4)
col1.metric("Status", "🟢 Active")
mode = config["general"]["mode"].upper()
col2.metric("Mode", "🎬 " + ("Simulasi" if mode == "FILE" else "CCTV"))
col3.metric("Threshold Alert", config["thresholds"]["alert"])
col4.metric("Threshold Warning", config["thresholds"]["warning"])

st.sidebar.header("Konfigurasi")

source_mode = st.sidebar.radio("Sumber Video", ["CCTV Langsung", "File Video"])

def _build_hls_html(stream_url, cam_name, detect_host="http://localhost:8765"):
    LOCATION = "Kota Semarang, Jawa Tengah"
    return f"""<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<script src="https://cdn.jsdelivr.net/npm/hls.js@latest"></script>
<style>
* {{ margin:0; padding:0; box-sizing:border-box; }}
html,body {{ width:100%; height:100%; background:#000; overflow:hidden; font-family:Arial,sans-serif; }}
#container {{ position:relative; width:100%; height:100vh; background:#000; }}
video {{ width:100%; height:100%; display:block; object-fit:contain; }}
canvas {{ position:absolute; top:0; left:0; width:100%; height:100%; pointer-events:none; }}
#hud-top {{ position:absolute; top:0; left:0; right:0; height:44px; background:rgba(0,0,0,0.7); display:flex; align-items:center; padding:0 14px; color:#fff; font-size:15px; font-weight:600; }}
#hud-name {{ flex:1; }}
    #hud-score {{ font-weight:700; }}
    #hud-status {{ margin:0 12px; padding:2px 8px; border-radius:4px; font-size:11px; font-weight:700; text-transform:uppercase; }}
#hud-badges {{ position:absolute; top:50px; left:12px; display:flex; gap:8px; font-size:13px; font-weight:600; }}
.badge {{ padding:3px 10px; border-radius:4px; color:#fff; }}
.badge-w {{ background:#e74c3c; }}
.badge-p {{ background:#2ecc71; }}
.badge-v {{ background:#f39c12; }}
#hud-bottom {{ position:absolute; bottom:0; left:0; right:0; height:36px; background:rgba(0,0,0,0.7); display:flex; align-items:center; padding:0 14px; color:#ccc; font-size:12px; }}
#hud-time {{ flex:1; }}
#hud-loc {{ }}
#alert-banner {{ position:absolute; top:50%; left:50%; transform:translate(-50%,-50%); background:rgba(231,76,60,0.9); color:#fff; font-size:28px; font-weight:800; padding:20px 40px; border-radius:8px; display:none; text-align:center; white-space:nowrap; z-index:10; }}
#loading {{ position:absolute; inset:0; display:flex; align-items:center; justify-content:center; color:#888; font-size:18px; background:#000; z-index:5; }}
#loading.hidden {{ display:none; }}
.conn-status {{ position:absolute; top:50px; right:12px; font-size:11px; color:#888; }}
</style>
</head>
<body>
<div id="container">
<video id="video" muted playsinline></video>
<canvas id="overlay"></canvas>
<div id="hud-top"><span id="hud-name">{cam_name}</span><span id="hud-status">NORMAL</span><span id="hud-score">0.00</span></div>
<div id="hud-badges"><span class="badge badge-w" id="badge-w">W:0</span><span class="badge badge-p" id="badge-p">P:0</span><span class="badge badge-v" id="badge-v">V:0</span></div>
<div id="hud-bottom"><span id="hud-time"></span><span id="hud-loc">{LOCATION}</span></div>
<div id="alert-banner">⚠️ ANOMALI TERDETEKSI</div>
<div id="loading">Menghubungkan ke stream...</div>
<div class="conn-status" id="conn-status"></div>
</div>
<script>
window.name = 'hls-canvas-detection';
var video = document.getElementById('video');
var container = document.getElementById('container');
var canvas = document.getElementById('overlay');
var ctx = canvas.getContext('2d');
var hudScore = document.getElementById('hud-score');
var hudName = document.getElementById('hud-name');
var badgeW = document.getElementById('badge-w');
var badgeP = document.getElementById('badge-p');
var badgeV = document.getElementById('badge-v');
var hudTime = document.getElementById('hud-time');
var hudStatus = document.getElementById('hud-status');
var alertBanner = document.getElementById('alert-banner');
var loading = document.getElementById('loading');
var connStatus = document.getElementById('conn-status');

var currentScore = 0;
var currentBadges = {{w:0,p:0,v:0}};
var currentAlert = false;
var streamActive = false;

// Object tracker state
var tracks = [];
var nextTrackId = 1;
var IOU_THRESH = 0.20;
var MAX_MISSED = 5;

function resizeCanvas() {{
    var rect = container.getBoundingClientRect();
    canvas.width = rect.width;
    canvas.height = rect.height;
}}

function getVideoContentArea() {{
    var vw = video.videoWidth || 1280;
    var vh = video.videoHeight || 720;
    var cw = canvas.width;
    var ch = canvas.height;
    if (cw === 0 || ch === 0) return {{x:0, y:0, w:cw, h:ch, sx:1, sy:1}};
    var scale = Math.min(cw / vw, ch / vh);
    var cw2 = vw * scale;
    var ch2 = vh * scale;
    return {{
        x: (cw - cw2) / 2,
        y: (ch - ch2) / 2,
        w: cw2,
        h: ch2,
        sx: cw2 / 640,
        sy: ch2 / 360,
    }};
}}

function boxIou(a, b) {{
    var ix1 = Math.max(a.x, b.x), iy1 = Math.max(a.y, b.y);
    var ix2 = Math.min(a.x + a.w, b.x + b.w), iy2 = Math.min(a.y + a.h, b.y + b.h);
    var iw = Math.max(0, ix2 - ix1), ih = Math.max(0, iy2 - iy1);
    var inter = iw * ih, union = a.w * a.h + b.w * b.h - inter;
    return union > 0 ? inter / union : 0;
}}

var NEW_TRACK_MIN_CONF = 0.25;
var ZOMBIE_REVIVE_DIST = 10000;

function matchDetections(dets) {{
    var used = new Set();
    for (var i = 0; i < tracks.length; i++) {{
        var t = tracks[i];
        var best = -1, bestIou = IOU_THRESH;
        var mx = t.prevDetX !== undefined ? t.prevDetX : t.x;
        var my = t.prevDetY !== undefined ? t.prevDetY : t.y;
        var pCx = mx + t.w / 2 + t.vx, pCy = my + t.h / 2 + t.vy;
        for (var j = 0; j < dets.length; j++) {{
            if (used.has(j)) continue;
            var iou = boxIou({{x: mx + t.vx, y: my + t.vy, w: t.w, h: t.h}}, dets[j]);
            if (t.missed > 0 && iou < IOU_THRESH) {{
                var dCx = dets[j].x + dets[j].w / 2, dCy = dets[j].y + dets[j].h / 2;
                var dx = dCx - pCx, dy = dCy - pCy;
                if (dx * dx + dy * dy < ZOMBIE_REVIVE_DIST) {{
                    var ar = (t.w * t.h) / Math.max(dets[j].w * dets[j].h, 1);
                    if (ar > 0.35 && ar < 2.8) iou = Math.max(iou, 0.25);
                }}
            }}
            if (iou > bestIou) {{ bestIou = iou; best = j; }}
        }}
        if (best >= 0) {{
            var d = dets[best];
            t.prevTime = Date.now();

            var ema = (t.t === 'motorcycle') ? 0.5 : 0.6;
            t.vx = t.vx * ema + (d.x - t.prevDetX) * (1 - ema);
            t.vy = t.vy * ema + (d.y - t.prevDetY) * (1 - ema);
            var spd = Math.sqrt(t.vx * t.vx + t.vy * t.vy);
            var baseAlpha = 0.15, maxAlpha = 0.25;
            if (t.t === 'motorcycle') {{ baseAlpha = 0.28; maxAlpha = 0.30; }}
            var alpha = baseAlpha + Math.min(spd * 0.008, maxAlpha);
            if (t.age < 3) alpha = 0.55;

            t.x += (d.x - t.x) * alpha;
            t.y += (d.y - t.y) * alpha;
            var sizeAlpha = (t.t === 'motorcycle') ? 0.15 : 0.12;
            t.w += (d.w - t.w) * sizeAlpha;
            t.h += (d.h - t.h) * sizeAlpha;
            if (t.w < 10) t.w = 10;
            if (t.h < 10) t.h = 10;

            t.prevDetX = d.x;
            t.prevDetY = d.y;
            t.c = d.c; t.t = d.t;
            t.missed = 0; t.age++;
            used.add(best);
        }} else {{
            t.missed++;
        }}
    }}
    for (var j = 0; j < dets.length; j++) {{
        if (used.has(j)) continue;
        var d = dets[j];
        if (d.c < NEW_TRACK_MIN_CONF) continue;

        var reviveTarget = null;
        for (var k = 0; k < tracks.length; k++) {{
            var t2 = tracks[k];
            if (d.t !== t2.t) continue;
            var mmx = t2.prevDetX !== undefined ? t2.prevDetX : t2.x;
            var mmy = t2.prevDetY !== undefined ? t2.prevDetY : t2.y;
            if (boxIou({{x: mmx, y: mmy, w: t2.w, h: t2.h}}, d) > 0.15) {{
                reviveTarget = t2;
                break;
            }}
            if (t2.missed > 0) {{
                var tc2x = mmx + t2.w / 2, tc2y = mmy + t2.h / 2;
                var dcx = d.x + d.w / 2, dcy = d.y + d.h / 2;
                var dx = dcx - tc2x, dy = dcy - tc2y;
                if (dx * dx + dy * dy < ZOMBIE_REVIVE_DIST) {{
                    var ratio = (t2.w * t2.h) / Math.max(d.w * d.h, 1);
                    if (ratio > 0.3 && ratio < 3.0) {{
                        reviveTarget = t2;
                        break;
                    }}
                }}
            }}
        }}
        if (reviveTarget) {{
            reviveTarget.missed = 0;
            reviveTarget.x += (d.x - reviveTarget.x) * 0.40;
            reviveTarget.y += (d.y - reviveTarget.y) * 0.40;
            reviveTarget.w += (d.w - reviveTarget.w) * 0.15;
            reviveTarget.h += (d.h - reviveTarget.h) * 0.15;
            reviveTarget.prevDetX = d.x;
            reviveTarget.prevDetY = d.y;
            reviveTarget.c = d.c;
            reviveTarget.t = d.t;
            reviveTarget.prevTime = Date.now();
            reviveTarget.age++;
            continue;
        }}

        tracks.push({{id: nextTrackId++, x: d.x, y: d.y, w: d.w, h: d.h, vx: 0, vy: 0, age: 1, missed: 0, c: d.c, t: d.t, prevDetX: d.x, prevDetY: d.y, prevTime: Date.now()}});
    }}
    tracks = tracks.filter(function(t) {{ return t.missed < MAX_MISSED; }});

    if (tracks.length > 1) {{
        tracks.sort(function(a,b) {{ return a.id - b.id; }});
        var clean = [];
        for (var i = 0; i < tracks.length; i++) {{
            var keep = true;
            var ti = tracks[i];
            var mix = ti.prevDetX !== undefined ? ti.prevDetX : ti.x;
            var miy = ti.prevDetY !== undefined ? ti.prevDetY : ti.y;
            var tiArea = ti.w * ti.h;
            for (var j = 0; j < clean.length; j++) {{
                var tj = clean[j];
                if (ti.t !== tj.t) continue;
                var mjx = tj.prevDetX !== undefined ? tj.prevDetX : tj.x;
                var mjy = tj.prevDetY !== undefined ? tj.prevDetY : tj.y;
                var cx = mix + ti.w / 2, cy = miy + ti.h / 2;
                var cxj = mjx + tj.w / 2, cyj = mjy + tj.h / 2;
                var dx = cx - cxj, dy = cy - cyj;
                if (dx * dx + dy * dy < 3600) {{
                    if (ti.missed > tj.missed) {{ keep = false; break; }}
                }}
            }}
            if (keep) clean.push(ti);
        }}
        tracks = clean;
    }}
}}

function predictTracks() {{
    for (var i = 0; i < tracks.length; i++) {{
        var t = tracks[i];
        if (t.missed > 0) continue;
        var pred = (t.t === 'motorcycle') ? 0.08 : 0.06;
        t.x += t.vx * pred;
        t.y += t.vy * pred;
    }}
}}

function drawTracks() {{
    ctx.clearRect(0, 0, canvas.width, canvas.height);
    var area = getVideoContentArea();
    for (var i = 0; i < tracks.length; i++) {{
        var t = tracks[i];
        var x = area.x + t.x * area.sx;
        var y = area.y + t.y * area.sy;
        var w = t.w * area.sx;
        var h = t.h * area.sy;
        var color = t.t === 'person' ? '#2ecc71' : (t.t === 'weapon' ? '#e74c3c' : '#f39c12');
        ctx.strokeStyle = color;
        ctx.lineWidth = 3;
        ctx.strokeRect(x, y, w, h);
        ctx.fillStyle = color;
        var label = '#' + t.id + ' ' + t.t + ' ' + Math.round(t.c * 100) + '%';
        ctx.font = 'bold 14px Arial';
        var tw = ctx.measureText(label).width;
        ctx.fillRect(x, y - 22, tw + 8, 22);
        ctx.fillStyle = '#fff';
        ctx.fillText(label, x + 4, y - 6);
    }}
}}

function updateHUD(data) {{
    currentScore = data.s || 0;
    currentBadges = {{w: data.w || 0, p: data.p || 0, v: data.v || 0}};
    currentAlert = data.a || false;
    hudScore.textContent = currentScore.toFixed(2);
    hudScore.style.color = currentScore < 0.6 ? '#2ecc71' : (currentScore < 0.8 ? '#f1c40f' : '#e74c3c');
    var statusStr = data.st || 'normal';
    hudStatus.textContent = statusStr.toUpperCase();
    if (statusStr === 'bahaya') {{ hudStatus.style.background = '#e74c3c'; hudStatus.style.color = '#fff'; }}
    else if (statusStr === 'mencurigakan') {{ hudStatus.style.background = '#f39c12'; hudStatus.style.color = '#000'; }}
    else {{ hudStatus.style.background = '#2ecc71'; hudStatus.style.color = '#000'; }}
    badgeW.textContent = 'W:' + currentBadges.w;
    badgeP.textContent = 'P:' + currentBadges.p;
    badgeV.textContent = 'V:' + currentBadges.v;
    if (data.t) hudTime.textContent = data.t;
    alertBanner.style.display = currentAlert ? 'block' : 'none';
}}

function handleDetectionData(data) {{
    matchDetections(data.boxes || []);
    updateHUD(data);
}}

function renderLoop() {{
    predictTracks();
    drawTracks();
    requestAnimationFrame(renderLoop);
}}

window.addEventListener('message', function(e) {{
    if (e.data && e.data.type === 'detection') {{
        handleDetectionData(e.data);
    }}
}});

try {{
    if (window.BroadcastChannel) {{
        var bc = new BroadcastChannel('hls_canvas_ch');
        bc.onmessage = function(e) {{
            if (e.data && e.data.type === 'detection') {{
                handleDetectionData(e.data);
            }}
        }};
    }}
}} catch(e) {{}}

var streamUrl = '{stream_url}';
var hls = null;
var hlsRetries = 0;
var MAX_HLS_RETRIES = 3;

function setupHLS(url) {{
    if (hls) {{
        hls.destroy();
        hls = null;
    }}
    loading.textContent = 'Menghubungkan ke stream...';
    loading.classList.remove('hidden');
    connStatus.textContent = '';
    if (video.canPlayType('application/vnd.apple.mpegurl') === 'probably' && !Hls.isSupported()) {{
        video.src = url;
        loading.classList.add('hidden');
        streamActive = true;
    }} else if (Hls.isSupported()) {{
        try {{
            hls = new Hls({{
                enableWorker: false,
                lowLatencyMode: false,
                backBufferLength: 30,
                maxBufferLength: 30,
                manifestLoadingTimeOut: 10000,
                levelLoadingTimeOut: 10000,
                fragLoadingTimeOut: 15000,
                enableDateRangeTag: false,
            }});
            hls.on(Hls.Events.MEDIA_ATTACHED, function() {{
                console.log('[HLS] Media attached');
            }});
            hls.on(Hls.Events.MANIFEST_PARSED, function() {{
                console.log('[HLS] Manifest parsed OK');
                loading.classList.add('hidden');
                streamActive = true;
                connStatus.textContent = '\\u2001 Live';
                video.play().catch(function(e) {{}});
            }});
            hls.on(Hls.Events.LEVEL_LOADED, function(evt, data) {{
                console.log('[HLS] Level loaded:', data.details);
            }});
            hls.on(Hls.Events.FRAG_LOADED, function() {{
                if (!streamActive) {{
                    streamActive = true;
                    loading.classList.add('hidden');
                    connStatus.textContent = '\\u2001 Live';
                }}
            }});
            hls.on(Hls.Events.ERROR, function(evt, data) {{
                console.warn('[HLS] Error:', data.type, data.details, data.fatal);
                if (data.fatal) {{
                    hlsRetries++;
                    if (hlsRetries <= MAX_HLS_RETRIES) {{
                        connStatus.textContent = '\\u26a0\\ufe0f Error: ' + (data.details || 'unknown') + ' (percobaan ' + hlsRetries + '/' + MAX_HLS_RETRIES + ')';
                        loading.textContent = 'Koneksi terputus. Mencoba reconnect dalam 3 detik...';
                        loading.classList.remove('hidden');
                        setTimeout(function() {{ setupHLS(url); }}, 3000);
                    }} else {{
                        loading.textContent = 'Camera offline - gagal setelah ' + MAX_HLS_RETRIES + ' percobaan';
                        connStatus.textContent = '\\u274c Offline';
                        streamActive = false;
                    }}
                }} else {{
                    connStatus.textContent = '\\u26a0\\ufe0f ' + (data.details || 'buffer issue');
                }}
            }});
            hls.loadSource(url);
            hls.attachMedia(video);
        }} catch(e) {{
            console.error('[HLS] Init error:', e);
            loading.textContent = 'Gagal inisialisasi stream';
        }}
    }} else {{
        loading.textContent = 'Browser tidak support HLS';
        connStatus.textContent = '\\u274c Unsupported';
    }}
    video.addEventListener('loadedmetadata', function() {{
        resizeCanvas();
    }});
}}

setupHLS(streamUrl);
renderLoop();
renderLoop();

window.addEventListener('resize', resizeCanvas);
video.addEventListener('resize', resizeCanvas);

setInterval(function() {{
    if (streamActive) {{
        var buffered = video.buffered;
        if (buffered.length > 0) {{
            connStatus.textContent = '\\u2001 Live';
        }}
    }}
}}, 5000);

var detectCanvas = document.createElement('canvas');
detectCanvas.width = 640;
detectCanvas.height = 360;
var detectCtx = detectCanvas.getContext('2d');
var detectWs = null;
var detectProcessing = false;
var detectTimeout = null;

function connectDetectWS() {{
    if (detectWs && detectWs.readyState <= 1) return;
    detectWs = new WebSocket('ws://localhost:8765');
    detectWs.onopen = function() {{ console.log('[Detect] WS connected'); tracks = []; nextTrackId = 1; }};
    detectWs.onmessage = function(e) {{
        try {{
            var data = JSON.parse(e.data);
            if (data.type === 'detection') {{ handleDetectionData(data); }}
        }} catch(err) {{}}
        detectProcessing = false;
        if (detectTimeout) {{ clearTimeout(detectTimeout); detectTimeout = null; }}
    }};
    detectWs.onclose = function() {{ detectProcessing = false; setTimeout(connectDetectWS, 2000); }};
    detectWs.onerror = function() {{ detectProcessing = false; }};
}}

var captureRAF = null;

function captureLoop() {{
    if (detectWs && detectWs.readyState === 1 && streamActive && video.readyState >= 2 && !detectProcessing) {{
        detectProcessing = true;
        detectTimeout = setTimeout(function() {{
            detectProcessing = false;
        }}, 2000);
        detectCtx.drawImage(video, 0, 0, 640, 360);
        detectCanvas.toBlob(function(blob) {{
            if (blob && detectWs && detectWs.readyState === 1) detectWs.send(blob);
        }}, 'image/jpeg', 0.75);
    }}
    captureRAF = requestAnimationFrame(captureLoop);
}}

connectDetectWS();
if (video.readyState >= 2) {{
    captureRAF = requestAnimationFrame(captureLoop);
}} else {{
    video.addEventListener('playing', function() {{
        captureRAF = requestAnimationFrame(captureLoop);
    }}, {{once: true}});
}}
</script>
</body>
</html>"""


alert_history = []

if source_mode == "File Video":
    video_source = st.sidebar.radio("Pilih File", ["Sample Videos", "Upload File"])
    video_path = None
    if video_source == "Sample Videos":
        video_files = list(Path("sample_videos").glob("*.mp4"))
        video_names = [f.name for f in video_files] if video_files else ["Tidak ada video"]
        selected = st.sidebar.selectbox("Pilih Video", video_names)
        video_path = str(Path("sample_videos") / selected) if video_files else None
    else:
        uploaded = st.sidebar.file_uploader("Upload Video", type=["mp4", "avi", "mov"])
        if uploaded:
            tfile = tempfile.NamedTemporaryFile(delete=False, suffix=".mp4")
            tfile.write(uploaded.read())
            video_path = tfile.name

    threshold_override = st.sidebar.slider(
        "Alert Threshold (override)", 0.5, 0.95, config["thresholds"]["alert"], 0.05
    )

    main_col, info_col = st.columns([3, 1])

    with main_col:
        st.subheader("📹 Live Detection Feed")
        feed_placeholder = st.empty()
        video_placeholder = st.empty()

        detect_btn = st.button("▶️ Mulai Deteksi", type="primary", use_container_width=True)

        if detect_btn and video_path and Path(video_path).exists():
            try:
                detector = AnomalyDetector("config.yaml")
                detector.config["thresholds"]["alert"] = threshold_override

                cap = cv2.VideoCapture(video_path)
                score_buffer = deque(maxlen=10)

                progress = st.progress(0)
                total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

                frame_count = 0
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    resized = cv2.resize(frame, (448, 448))
                    frame_count += 1
                    progress.progress(min(frame_count / total_frames, 1.0))

                    yolo_objects = detector.yolo.detect(resized)
                    faces = detector.yolo.detect_faces(resized)
                    if faces:
                        yolo_objects["faces"] = faces
                    plate = detector.yolo.detect_plate(resized)
                    yolo_objects["plate"] = plate

                    display = resized.copy()
                    for obj_type, items in yolo_objects.items():
                        if obj_type == "plate":
                            continue
                        for item in items:
                            x1, y1, x2, y2 = map(int, item["bbox"])
                            conf = item["confidence"]
                            label = f"{obj_type} {conf:.0%}"
                            color = (0, 255, 0) if obj_type == "persons" else (0, 0, 255)
                            cv2.rectangle(display, (x1, y1), (x2, y2), color, 2)
                            cv2.putText(display, label, (x1, y1 - 10),
                                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)

                    if yolo_objects.get("weapons", []):
                        cv2.putText(display, "!!! SENJATA TERDETEKSI !!!", (50, 400),
                                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 3)

                    display_rgb = cv2.cvtColor(display, cv2.COLOR_BGR2RGB)
                    video_placeholder.image(display_rgb, channels="RGB", use_container_width=True)

                cap.release()
                progress.progress(1.0)
                st.success(f"✅ Selesai: {frame_count} frame diproses")
            except Exception as e:
                st.error(f"Error: {e}")

    with info_col:
        st.subheader("🚨 Riwayat Alert")
        if alert_history:
            for a in reversed(alert_history[-10:]):
                st.warning(f"**{a['time']}** - Score: {a['score']:.2f}")
        else:
            st.info("Belum ada alert")
        st.subheader("📍 Lokasi")
        st.caption(config["location"]["address"])
        st.markdown(f"[🗺️ Buka Google Maps]({config['location']['maps_link']})")
        st.subheader("⚙️ Info Sistem")
        st.caption(f"Waktu: {datetime.now().strftime('%d %B %Y %H:%M:%S')}")
        st.caption(f"Device: {config['general']['device'].upper()}")
        st.caption(f"Model: YOLO11m + Indo Weapon + Sajam CNN")
        st.caption(f"Alert via: Telegram & WhatsApp")

else:
    st.sidebar.subheader("Pilih CCTV")
    cam_names = list(CCTV_NAMES.keys())
    selected_cams = st.sidebar.multiselect(
        "Pilih Kamera",
        options=cam_names,
        default=cam_names[:2] if len(cam_names) >= 2 else cam_names,
    )

    selected_urls = {name: CCTV_NAMES[name] for name in selected_cams}
    st.sidebar.caption(f"Total kamera aktif: {len(selected_urls)}")
    st.sidebar.caption("Deteksi berjalan via WebSocket ws://localhost:8765")

    if selected_urls:
        cam_names_list = list(selected_urls.keys())
        n_cams = len(cam_names_list)
        cols_per_row = min(3, n_cams)
        rows = (n_cams + cols_per_row - 1) // cols_per_row

        for r in range(rows):
            row_cams = cam_names_list[r * cols_per_row:(r + 1) * cols_per_row]
            cols = st.columns(cols_per_row)
            for ci, cam_name in enumerate(row_cams):
                with cols[ci]:
                    html = _build_hls_html(selected_urls[cam_name], cam_name)
                    st.components.v1.html(html, height=360, scrolling=False)
    else:
        st.info("Pilih minimal 1 kamera CCTV dari sidebar")

st.markdown("---")
st.caption(
    "🚨 **Sistem Deteksi Kerusuhan & Anomali** | "
    "YOLO11m + Indo Weapon + Sajam CNN | "
    "Laporan dikirim ke Telegram & WhatsApp | "
    "© 2026"
)
