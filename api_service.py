from fastapi import FastAPI, UploadFile, File, HTTPException, Response, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
from pathlib import Path
import uuid
import uvicorn
import httpx
import time
import asyncio

from inference import AnomalyDetector
from utils.config_loader import load_config

app = FastAPI(title="Deteksi Kerusuhan API", version="1.0.0")

static_dir = Path(__file__).parent / "static"
if static_dir.exists():
    app.mount("/static", StaticFiles(directory=str(static_dir)), name="static")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

detector: Optional[AnomalyDetector] = None


class DetectionResponse(BaseModel):
    report_id: str
    timestamp: str
    anomaly_score: float
    status: str
    message: str


_HTTP_CLIENT = httpx.AsyncClient(
    verify=False,
    timeout=httpx.Timeout(30.0, connect=15.0),
    headers={
        "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
        "Origin": "https://pantausemar.semarangkota.go.id",
        "Referer": "https://pantausemar.semarangkota.go.id/",
    },
)

# Simple playlist cache: {key: (data, content_type, timestamp)}
_playlist_cache: dict = {}
_PLAYLIST_CACHE_TTL = 3.0  # seconds


@app.get("/hls-proxy/{rest:path}")
async def hls_proxy(rest: str):
    base_url = f"https://livepantau.semarangkota.go.id/{rest}"
    is_playlist = rest.endswith(".m3u8")

    # Check cache for playlists
    if is_playlist:
        cached = _playlist_cache.get(rest)
        if cached and (time.time() - cached[2]) < _PLAYLIST_CACHE_TTL:
            return Response(content=cached[0], media_type=cached[1])

    try:
        async with _HTTP_CLIENT.stream("GET", base_url) as resp:
            ct = resp.headers.get("content-type", "application/octet-stream")
            data = await resp.aread()

            # Rewrite sub-stream URLs in master playlist to use proxy
            if is_playlist and rest.endswith("index.m3u8"):
                text = data.decode("utf-8", errors="replace")
                base_path = "/".join(rest.split("/")[:-1]) + "/"
                lines = text.splitlines()
                rewritten = []
                for line in lines:
                    s = line.strip()
                    if s.endswith(".m3u8") and not s.startswith("#"):
                        rewritten.append(f"http://localhost:8000/hls-proxy/{base_path}{s}")
                    else:
                        rewritten.append(line)
                data = "\n".join(rewritten).encode("utf-8")

            # Store in cache for playlists
            if is_playlist:
                _playlist_cache[rest] = (data, ct, time.time())

            return Response(content=data, media_type=ct)
    except httpx.HTTPStatusError as e:
        return Response(content=e.response.content, status_code=e.response.status_code)
    except Exception:
        return Response(content=b"Proxy error", status_code=502)


async def _clean_cache():
    while True:
        await asyncio.sleep(30)
        now = time.time()
        stale = [k for k, v in _playlist_cache.items() if (now - v[2]) > _PLAYLIST_CACHE_TTL * 2]
        for k in stale:
            del _playlist_cache[k]


@app.on_event("startup")
async def startup():
    global detector
    config = load_config()
    detector = AnomalyDetector()
    asyncio.create_task(_clean_cache())


@app.post("/detect", response_model=DetectionResponse)
async def detect_video(file: UploadFile = File(...)):
    if not file.filename.endswith((".mp4", ".avi", ".mov")):
        raise HTTPException(400, "Format video tidak didukung")

    temp_path = f"/tmp/{uuid.uuid4()}_{file.filename}"
    with open(temp_path, "wb") as f:
        f.write(await file.read())

    report_id = f"ALRT-{datetime.now().strftime('%Y%m%d-%H%M%S')}"
    return DetectionResponse(
        report_id=report_id,
        timestamp=datetime.now().isoformat(),
        anomaly_score=0.0,
        status="processed",
        message="Video diterima untuk diproses",
    )


@app.on_event("shutdown")
async def shutdown():
    await _HTTP_CLIENT.aclose()


@app.get("/check-cctv")
async def check_cctv(url: str = Query(...)):
    t0 = time.time()
    try:
        rest = url.replace("https://livepantau.semarangkota.go.id/", "")
        if rest == url:
            return {"ok": False, "latency_ms": 0, "error": "invalid_url"}
        resp = await _HTTP_CLIENT.get(f"https://livepantau.semarangkota.go.id/{rest}", timeout=httpx.Timeout(8.0, connect=5.0))
        elapsed = int((time.time() - t0) * 1000)
        return {"ok": resp.is_success, "latency_ms": elapsed, "error": None if resp.is_success else f"http_{resp.status_code}"}
    except httpx.TimeoutException:
        elapsed = int((time.time() - t0) * 1000)
        return {"ok": False, "latency_ms": elapsed, "error": "timeout"}
    except Exception as e:
        elapsed = int((time.time() - t0) * 1000)
        return {"ok": False, "latency_ms": elapsed, "error": str(e)[:60]}


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


def start_api(host="0.0.0.0", port=8000):
    uvicorn.run(app, host=host, port=port, log_level="warning")

if __name__ == "__main__":
    start_api()
