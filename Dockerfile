FROM python:3.11-slim AS base

# Prevent Python from writing .pyc and enable unbuffered output for logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps required by faster-whisper (ctranslate2) and audio handling
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Copy backend source and install (hatchling needs the source to build)
COPY backend/ backend/
RUN pip install --no-cache-dir ./backend[cpu]

# Copy frontend
COPY frontend/ frontend/

# Create non-root user and writable data directory
RUN groupadd -r whisper && useradd -r -g whisper -u 1000 whisper && \
    mkdir -p /data && chown whisper:whisper /data

ENV WHISPER_ACTIVITY_LOG=/data/activity.jsonl \
    WHISPER_MODEL_SIZE=small \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=float32

USER 1000

EXPOSE 8000

CMD ["uvicorn", "whisper_streamer.main:app", "--host", "0.0.0.0", "--port", "8000"]
