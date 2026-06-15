import json
import csv
import re
import sys
import os
from pathlib import Path
from http.server import HTTPServer, BaseHTTPRequestHandler
from urllib.parse import urlparse, unquote
import mimetypes


BASE_DIR = Path(__file__).resolve().parent.parent
DECISIONS_FILE = BASE_DIR / "curation" / "decisions.csv"
DECISIONS_FIELDS = [
    "file_path", "video_id", "title", "category", "label",
    "source", "channel", "duration_sec", "dataset", "decision", "reviewed_at",
]
PORT = 8765


def scan_videos():
    """Only scan ANOMALY videos for review.
    Normal v3 videos are auto-accepted (best quality CCTV).
    Normal v1/v2 are discarded (mostly tutorials/music/news).
    """
    videos = []
    seen_ids = set()

    datasets = {
        "v1": BASE_DIR / "sample_videos" / "indonesia",
        "v2": BASE_DIR / "sample_videos" / "indonesia_v2",
        "v3": BASE_DIR / "sample_videos" / "indonesia_v3",
        "tiktok": BASE_DIR / "sample_videos" / "tiktok",
        "twitter": BASE_DIR / "sample_videos" / "twitter",
    }

    for ds_name, ds_dir in datasets.items():
        metadata_csv = ds_dir / "metadata.csv"
        meta_lookup = {}
        if metadata_csv.exists():
            with open(metadata_csv, encoding="utf-8") as f:
                for row in csv.DictReader(f):
                    vid = row.get("video_id", "")
                    if vid:
                        meta_lookup[vid] = row

        for fpath in sorted(ds_dir.rglob("*.mp4")):
            vid = fpath.stem
            if vid in seen_ids:
                continue
            seen_ids.add(vid)

            meta = meta_lookup.get(vid, {})
            label = int(meta.get("label", 0))
            category = meta.get("category", "unknown")
            channel = meta.get("channel", "")
            title = meta.get("title", vid)
            dur = meta.get("duration_sec", "")
            source = meta.get("source", "YouTube")
            rel_path = str(fpath.relative_to(BASE_DIR))

            # Only include anomaly (label=1) for manual review
            if label != 1:
                continue

            videos.append({
                "file_path": rel_path,
                "video_id": vid,
                "title": title,
                "category": category,
                "label": label,
                "source": source,
                "channel": channel,
                "duration_sec": dur,
                "dataset": ds_name,
                "abs_path": str(fpath.resolve()),
            })

    return videos


def auto_accept_normal_v3():
    """Auto-save 'keep' for all normal v3 videos (best quality CCTV live feeds)."""
    from datetime import datetime
    v3_dir = BASE_DIR / "sample_videos" / "indonesia_v3"
    metadata_csv = v3_dir / "metadata.csv"
    if not metadata_csv.exists():
        return

    meta_lookup = {}
    with open(metadata_csv, encoding="utf-8") as f:
        for row in csv.DictReader(f):
            vid = row.get("video_id", "")
            if vid:
                meta_lookup[vid] = row

    for fpath in v3_dir.rglob("*.mp4"):
        vid = fpath.stem
        meta = meta_lookup.get(vid, {})
        label = int(meta.get("label", 0))
        existing = load_decisions()

        if label == 0 and vid not in existing:
            rel_path = str(fpath.relative_to(BASE_DIR))
            entry = {
                "file_path": rel_path,
                "video_id": vid,
                "title": meta.get("title", vid),
                "category": meta.get("category", "traffic"),
                "label": 0,
                "source": "YouTube",
                "channel": meta.get("channel", ""),
                "duration_sec": meta.get("duration_sec", ""),
                "dataset": "v3",
                "decision": "keep",
                "reviewed_at": datetime.now().isoformat(),
            }
            save_decision(entry)


def load_decisions():
    decisions = {}
    if DECISIONS_FILE.exists():
        with open(DECISIONS_FILE, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                vid = row.get("video_id", "")
                if vid:
                    decisions[vid] = row
    return decisions


def save_decision(decision: dict):
    exists = DECISIONS_FILE.exists()
    with open(DECISIONS_FILE, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=DECISIONS_FIELDS)
        if not exists:
            writer.writeheader()
        writer.writerow(decision)


videos = scan_videos()
decisions = load_decisions()


class CurationHandler(BaseHTTPRequestHandler):
    def do_GET(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        # API routes
        if path == "/api/videos":
            self._json_response(self._build_video_list())
            return

        if path == "/api/stats":
            total = len(videos)
            reviewed = sum(1 for v in videos if v["video_id"] in decisions)
            keep = sum(1 for v in videos if v["video_id"] in decisions and decisions[v["video_id"]]["decision"] == "keep")
            reject = sum(1 for v in videos if v["video_id"] in decisions and decisions[v["video_id"]]["decision"] == "reject")
            self._json_response({
                "total": total,
                "reviewed": reviewed,
                "remaining": total - reviewed,
                "keep": keep,
                "reject": reject,
                "progress_pct": round(reviewed / total * 100, 1) if total else 0,
            })
            return

        # Serve video files
        if path.startswith("/videos/"):
            rel_path = path[len("/videos/"):]
            abs_path = (BASE_DIR / rel_path).resolve()

            # Security: ensure it's within BASE_DIR
            try:
                abs_path.relative_to(BASE_DIR)
            except ValueError:
                self._send_error(403, "Forbidden")
                return

            if abs_path.exists():
                mime_type, _ = mimetypes.guess_type(str(abs_path))
                self.send_response(200)
                self.send_header("Content-Type", mime_type or "video/mp4")
                self.send_header("Accept-Ranges", "bytes")
                self.send_header("Content-Length", str(abs_path.stat().st_size))
                self.end_headers()
                with open(abs_path, "rb") as f:
                    self.wfile.write(f.read())
                return
            else:
                self._send_error(404, "File not found")
                return

        # Serve static files (curation dir)
        if path == "/" or path == "":
            self._serve_html()
            return

        self._send_error(404, "Not found")

    def do_POST(self):
        parsed = urlparse(self.path)
        path = unquote(parsed.path)

        content_length = int(self.headers.get("Content-Length", 0))
        body = self.rfile.read(content_length) if content_length else b"{}"
        data = json.loads(body) if body else {}

        # POST /api/keep or /api/reject
        if path in ("/api/keep", "/api/reject"):
            decision = path.split("/")[-1]
            video_id = data.get("video_id", "")
            if not video_id:
                self._json_response({"error": "missing video_id"}, 400)
                return

            vid = next((v for v in videos if v["video_id"] == video_id), None)
            if not vid:
                self._json_response({"error": "video not found"}, 404)
                return

            from datetime import datetime
            entry = {
                "file_path": vid["file_path"],
                "video_id": vid["video_id"],
                "title": vid["title"],
                "category": vid["category"],
                "label": vid["label"],
                "source": vid["source"],
                "channel": vid["channel"],
                "duration_sec": vid["duration_sec"],
                "dataset": vid["dataset"],
                "decision": decision,
                "reviewed_at": datetime.now().isoformat(),
            }
            save_decision(entry)
            decisions[video_id] = entry
            self._json_response({"status": "ok", "decision": decision})
            return

        self._send_error(404, "Not found")

    def _serve_html(self):
        self.send_response(200)
        self.send_header("Content-Type", "text/html; charset=utf-8")
        self.end_headers()
        self.wfile.write(HTML_PAGE.encode("utf-8"))

    def _json_response(self, data, status=200):
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Access-Control-Allow-Origin", "*")
        self.end_headers()
        self.wfile.write(json.dumps(data).encode("utf-8"))

    def _send_error(self, code, msg):
        self.send_response(code)
        self.send_header("Content-Type", "text/plain")
        self.end_headers()
        self.wfile.write(msg.encode("utf-8"))

    def log_message(self, format, *args):
        # Quieter logging — only log POST actions
        if self.command == "POST":
            super().log_message(format, *args)


HTML_PAGE = """<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>Curation Tool — Deteksi Kerusuhan</title>
<style>
* { margin: 0; padding: 0; box-sizing: border-box; }
body { font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
       background: #0f0f0f; color: #eee; min-height: 100vh; display: flex; }
.sidebar { width: 380px; background: #1a1a1a; padding: 20px; display: flex;
           flex-direction: column; border-right: 1px solid #333; }
.main { flex: 1; display: flex; flex-direction: column; align-items: center; padding: 20px; }
.video-container { width: 100%; max-width: 854px; background: #000; border-radius: 8px;
                   overflow: hidden; margin-bottom: 16px; position: relative; }
.video-container video { width: 100%; display: block; max-height: 70vh; }
.meta { flex: 1; overflow-y: auto; }
.meta h2 { font-size: 16px; margin-bottom: 8px; color: #fff; }
.meta-table { width: 100%; font-size: 13px; }
.meta-table td { padding: 4px 0; }
.meta-table td:first-child { color: #888; width: 90px; }
.meta-table td:last-child { color: #ddd; }
.badge { display: inline-block; padding: 2px 8px; border-radius: 4px; font-size: 11px;
         font-weight: 600; }
.badge-anomaly { background: #e74c3c22; color: #e74c3c; border: 1px solid #e74c3c44; }
.badge-normal { background: #2ecc7122; color: #2ecc71; border: 1px solid #2ecc7144; }
.badge-v1 { background: #f39c1222; color: #f39c12; }
.badge-v2 { background: #3498db22; color: #3498db; }
.badge-v3 { background: #9b59b622; color: #9b59b6; }
.actions { display: flex; gap: 10px; margin: 16px 0; flex-wrap: wrap; }
.btn { padding: 14px 32px; border: none; border-radius: 8px; font-size: 16px;
       font-weight: 700; cursor: pointer; transition: all 0.2s; flex: 1; min-width: 120px; }
.btn:active { transform: scale(0.96); }
.btn-keep { background: #2ecc71; color: #fff; }
.btn-keep:hover { background: #27ae60; }
.btn-reject { background: #e74c3c; color: #fff; }
.btn-reject:hover { background: #c0392b; }
.btn-skip { background: #555; color: #fff; }
.btn-skip:hover { background: #666; }
.btn-nav { background: #333; color: #aaa; padding: 10px 16px; border: none;
           border-radius: 6px; cursor: pointer; font-size: 13px; }
.btn-nav:hover { background: #444; color: #fff; }
.progress { margin-top: auto; padding-top: 16px; border-top: 1px solid #333; }
.progress-bar { height: 6px; background: #333; border-radius: 3px; margin: 8px 0; overflow: hidden; }
.progress-fill { height: 100%; background: #2ecc71; border-radius: 3px; transition: width 0.3s; }
.progress-text { font-size: 13px; color: #888; }
.done-message { display: none; text-align: center; padding: 60px 20px; }
.done-message h1 { font-size: 48px; margin-bottom: 16px; }
.done-message p { font-size: 18px; color: #888; }
.shortcut-hint { font-size: 11px; color: #555; margin-top: 8px; text-align: center; }
.empty-state { padding: 20px; text-align: center; color: #888; }
.nav-bar { display: flex; gap: 8px; align-items: center; }
.nav-bar input { width: 60px; padding: 6px; border-radius: 4px; border: 1px solid #444;
                 background: #222; color: #eee; text-align: center; font-size: 13px; }
.thumb-grid { display: flex; flex-wrap: wrap; gap: 4px; margin: 8px 0; }
.thumb-item { width: 28px; height: 20px; border-radius: 2px; cursor: pointer;
              position: relative; font-size: 7px; display: flex; align-items: center;
              justify-content: center; }
.thumb-unreviewed { background: #333; }
.thumb-keep { background: #2ecc71; }
.thumb-reject { background: #e74c3c; }
.thumb-current { outline: 2px solid #fff; outline-offset: 1px; }
</style>
</head>
<body>
<div class="sidebar" id="sidebar">
  <div class="meta" id="meta">
    <div class="empty-state">Select a video to review</div>
  </div>
  <div class="progress" id="progress">
    <div class="progress-text" id="progressText">Loading...</div>
    <div class="progress-bar"><div class="progress-fill" id="progressFill"></div></div>
    <div class="thumb-grid" id="thumbGrid"></div>
  </div>
</div>

<div class="main">
  <div class="nav-bar">
    <button class="btn-nav" onclick="goFirst()">|◀</button>
    <button class="btn-nav" onclick="goPrev()">◀</button>
    <span id="navIndex">0 / 0</span>
    <button class="btn-nav" onclick="goNext()">▶</button>
    <button class="btn-nav" onclick="goLast()">▶|</button>
    <span style="color:#555;font-size:12px">Go to:</span>
    <input type="number" id="jumpInput" min="0" value="0" onkeydown="if(event.key==='Enter')jumpTo(this.value)">
  </div>

  <div class="video-container" id="videoContainer">
    <video id="player" controls preload="auto"></video>
  </div>

  <div class="actions" id="actions">
    <button class="btn btn-keep" onclick="decide('keep')">✓ KEEP [K]</button>
    <button class="btn btn-reject" onclick="decide('reject')">✗ REJECT [R]</button>
    <button class="btn btn-skip" onclick="skipVideo()">→ SKIP [S]</button>
  </div>
  <div class="shortcut-hint">Keyboard: K=keep · R=reject · S=skip · ←/→ = prev/next · 1-0 = jump to video #</div>

  <div class="done-message" id="doneMessage">
    <h1>🎉</h1>
    <h1>All reviewed!</h1>
    <p>All videos have been reviewed. Great job!</p>
    <button class="btn btn-keep" onclick="showRemaining()">Show remaining</button>
  </div>
</div>

<script>
let videos = [];
let decisions = {};
let currentIndex = -1;
let currentVideoId = null;

async function loadData() {
  const [vidsRes, statsRes] = await Promise.all([
    fetch('/api/videos'),
    fetch('/api/stats')
  ]);
  const data = await vidsRes.json();
  videos = data.videos;
  decisions = data.decisions;

  // Find first unreviewed
  const firstUnreviewed = videos.findIndex(v => !decisions[v.video_id]);
  currentIndex = firstUnreviewed >= 0 ? firstUnreviewed : 0;

  updateProgress();
  renderThumbGrid();
  showVideo();
}

function getVideo() { return videos[currentIndex]; }

function showVideo() {
  const v = getVideo();
  if (!v) return;
  currentVideoId = v.video_id;

  const player = document.getElementById('player');
  const videoPath = '/videos/' + v.file_path.replace(/\\\\/g, '/');

  // Only reload if different video
  if (player.src !== window.location.origin + videoPath) {
    player.src = videoPath;
    player.load();
  }

  const meta = document.getElementById('meta');
  const labelClass = v.label == 1 ? 'badge-anomaly' : 'badge-normal';
  const labelText = v.label == 1 ? 'ANOMALY' : 'NORMAL';
  const dsClass = 'badge-' + v.dataset;

  const decision = decisions[v.video_id];
  const decisionBadge = decision
    ? `<span style="color:${decision.decision === 'keep' ? '#2ecc71' : '#e74c3c'};font-weight:700;margin-left:8px">${decision.decision.toUpperCase()}</span>`
    : '<span style="color:#555;margin-left:8px">unreviewed</span>';

  meta.innerHTML = `
    <h2>${escHtml(v.title || v.video_id)} ${decisionBadge}</h2>
    <table class="meta-table">
      <tr><td>Video ID</td><td><code>${v.video_id}</code></td></tr>
      <tr><td>Label</td><td><span class="badge ${labelClass}">${labelText}</span></td></tr>
      <tr><td>Category</td><td>${v.category}</td></tr>
      <tr><td>Dataset</td><td><span class="badge ${dsClass}">${v.dataset}</span></td></tr>
      <tr><td>Channel</td><td>${escHtml(v.channel || '-')}</td></tr>
      <tr><td>Duration</td><td>${v.duration_sec ? v.duration_sec + 's' : 'live'}</td></tr>
      <tr><td>Source</td><td>${v.source}</td></tr>
      <tr><td>Path</td><td style="font-size:11px;word-break:break-all">${v.file_path}</td></tr>
    </table>
  `;

  document.getElementById('navIndex').textContent = `${currentIndex + 1} / ${videos.length}`;
  document.getElementById('doneMessage').style.display = 'none';
  document.getElementById('actions').style.display = 'flex';
  document.getElementById('player').style.display = 'block';
  document.getElementById('videoContainer').style.display = 'block';
  updateThumbGrid();
  updateProgress();
}

function decide(action) {
  const v = getVideo();
  if (!v) return;

  fetch('/api/' + action, {
    method: 'POST',
    headers: {'Content-Type': 'application/json'},
    body: JSON.stringify({ video_id: v.video_id })
  }).then(r => r.json()).then(() => {
    decisions[v.video_id] = { decision: action };
    updateProgress();
    updateThumbGrid();
    showVideo();  // refresh badge
    autoAdvance();
  });
}

function autoAdvance() {
  const nextIdx = videos.findIndex((v, i) => i > currentIndex && !decisions[v.video_id]);
  if (nextIdx >= 0) {
    currentIndex = nextIdx;
    showVideo();
  } else {
    // Check if all done
    const remaining = videos.filter(v => !decisions[v.video_id]).length;
    if (remaining === 0) {
      document.getElementById('doneMessage').style.display = 'block';
      document.getElementById('actions').style.display = 'none';
    }
  }
}

function skipVideo() {
  const nextIdx = videos.findIndex((v, i) => i > currentIndex);
  if (nextIdx >= 0) {
    currentIndex = nextIdx;
    showVideo();
  }
}

function goNext() {
  if (currentIndex < videos.length - 1) {
    currentIndex++;
    showVideo();
  }
}

function goPrev() {
  if (currentIndex > 0) {
    currentIndex--;
    showVideo();
  }
}

function goFirst() { currentIndex = 0; showVideo(); }
function goLast() { currentIndex = videos.length - 1; showVideo(); }

function jumpTo(n) {
  const idx = parseInt(n) - 1;
  if (idx >= 0 && idx < videos.length) {
    currentIndex = idx;
    showVideo();
  }
}

function showRemaining() {
  const idx = videos.findIndex(v => !decisions[v.video_id]);
  if (idx >= 0) { currentIndex = idx; showVideo(); }
}

function updateProgress() {
  const total = videos.length;
  const reviewed = Object.keys(decisions).length;
  const pct = total ? (reviewed / total * 100) : 0;
  document.getElementById('progressText').textContent =
    `${reviewed} / ${total} reviewed (${pct.toFixed(1)}%) — ` +
    `${total - reviewed} remaining`;
  document.getElementById('progressFill').style.width = pct + '%';
}

function renderThumbGrid() {
  const grid = document.getElementById('thumbGrid');
  grid.innerHTML = '';
  videos.forEach((v, i) => {
    const el = document.createElement('div');
    el.className = 'thumb-item';
    const d = decisions[v.video_id];
    if (d && d.decision === 'keep') el.classList.add('thumb-keep');
    else if (d && d.decision === 'reject') el.classList.add('thumb-reject');
    else el.classList.add('thumb-unreviewed');
    el.title = `${i+1}: ${v.title || v.video_id}`;
    el.onclick = () => { currentIndex = i; showVideo(); };
    grid.appendChild(el);
  });
}

function updateThumbGrid() {
  const items = document.getElementById('thumbGrid').children;
  for (let i = 0; i < items.length; i++) {
    items[i].classList.remove('thumb-current');
    if (i === currentIndex) items[i].classList.add('thumb-current');
    const d = decisions[videos[i].video_id];
    items[i].className = 'thumb-item';
    if (d && d.decision === 'keep') items[i].classList.add('thumb-keep');
    else if (d && d.decision === 'reject') items[i].classList.add('thumb-reject');
    else items[i].classList.add('thumb-unreviewed');
    if (i === currentIndex) items[i].classList.add('thumb-current');
  }
}

function escHtml(s) {
  const d = document.createElement('div');
  d.textContent = s;
  return d.innerHTML;
}

// Keyboard shortcuts
document.addEventListener('keydown', (e) => {
  if (e.target.tagName === 'INPUT') return;
  switch (e.key.toLowerCase()) {
    case 'k': decide('keep'); e.preventDefault(); break;
    case 'r': decide('reject'); e.preventDefault(); break;
    case 's': skipVideo(); e.preventDefault(); break;
    case 'arrowright': goNext(); e.preventDefault(); break;
    case 'arrowleft': goPrev(); e.preventDefault(); break;
    case 'home': goFirst(); e.preventDefault(); break;
    case 'end': goLast(); e.preventDefault(); break;
  }
  // Number keys 1-9,0 for jump
  const n = parseInt(e.key);
  if (n >= 0 && n <= 9) {
    const ratio = n / 10;
    const idx = Math.floor(ratio * videos.length);
    if (n === 0) goLast();
    else { currentIndex = Math.min(idx, videos.length - 1); showVideo(); }
    e.preventDefault();
  }
});

loadData();
</script>
</body>
</html>"""


def build_video_list(self=None):
    """Build video list with decisions for API response."""
    vlist = []
    for v in videos:
        entry = {k: v[k] for k in ("file_path", "video_id", "title", "category",
                                     "label", "source", "channel", "duration_sec", "dataset")}
        entry["label"] = v["label"]
        entry["duration_sec"] = str(v["duration_sec"]) if v["duration_sec"] else ""
        vlist.append(entry)
    return {"videos": vlist, "decisions": {vid: {"decision": d["decision"]} for vid, d in decisions.items()}}


# Override to use the function properly
CurationHandler._build_video_list = staticmethod(build_video_list)


def main():
    # Fix build_video_list as classmethod
    CurationHandler._build_video_list = staticmethod(build_video_list)

    # Auto-accept normal v3 (best quality CCTV live feeds, no review needed)
    auto_accept_normal_v3()
    # Reload decisions after auto-accept
    global decisions
    decisions = load_decisions()

    server = HTTPServer(("0.0.0.0", PORT), CurationHandler)
    print(f"  {'='*50}")
    print(f"  CURATION TOOL — ANOMALY ONLY")
    print(f"  {'='*50}")
    print(f"  → http://localhost:{PORT}")
    print(f"")
    print(f"  Videos to review:    {len(videos)} anomaly")
    print(f"  Auto-accepted:       {len(decisions) - sum(1 for v in videos if v['video_id'] in decisions)} normal v3")
    print(f"  Reviewed so far:     {sum(1 for v in videos if v['video_id'] in decisions)}")
    print(f"  Decisions saved to:  {DECISIONS_FILE}")
    print(f"")
    print(f"  Controls:")
    print(f"    K = Keep    R = Reject    S = Skip")
    print(f"    ← → = Navigate    1-9 = Jump % through list")
    print(f"  {'='*50}")
    print(f"  Press Ctrl+C to stop\n")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\n  Stopped.")


if __name__ == "__main__":
    main()
