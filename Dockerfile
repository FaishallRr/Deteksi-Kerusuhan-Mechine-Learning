FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender-dev \
    libgomp1 \
    ffmpeg \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

RUN mkdir -p models sample_videos evidence docs

COPY . .

EXPOSE 8000
EXPOSE 8501

CMD ["sh", "-c", "streamlit run app.py --server.port=8501 --server.address=0.0.0.0 & uvicorn api_service:app --host 0.0.0.0 --port 8000"]
