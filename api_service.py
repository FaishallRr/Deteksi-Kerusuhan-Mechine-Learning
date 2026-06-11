from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
from datetime import datetime
import uuid
import uvicorn

from inference import AnomalyDetector
from utils.config_loader import load_config

app = FastAPI(title="Deteksi Kerusuhan API", version="1.0.0")

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


@app.on_event("startup")
async def startup():
    global detector
    config = load_config()
    detector = AnomalyDetector()


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


@app.get("/health")
async def health():
    return {"status": "ok", "timestamp": datetime.now().isoformat()}


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
