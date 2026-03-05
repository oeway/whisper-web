FROM python:3.12-slim AS base

# Prevent Python from writing .pyc and enable unbuffered output for logging
ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1

WORKDIR /app

# Install system deps: ffmpeg for audio conversion, libsndfile for WAV I/O
RUN apt-get update && \
    apt-get install -y --no-install-recommends ffmpeg libsndfile1 && \
    rm -rf /var/lib/apt/lists/*

# Copy backend source and install (hatchling needs the source to build)
COPY backend/ backend/
RUN pip install --no-cache-dir ./backend[cpu]

# Copy frontend
COPY frontend/ frontend/

# Create non-root user and writable data directory
RUN groupadd -r whisper && useradd -r -g whisper -u 1000 -m whisper && \
    mkdir -p /data && chown whisper:whisper /data

# HF_HOME: redirect HuggingFace model cache to /data/models so it can be
# backed by a PersistentVolumeClaim in Kubernetes, avoiding re-downloads on
# pod restarts. Mount a PVC at /data in production.
ENV HF_HOME=/data/models \
    WHISPER_ACTIVITY_LOG=/data/activity.jsonl \
    WHISPER_MODEL_SIZE=small \
    WHISPER_DEVICE=cpu \
    WHISPER_COMPUTE_TYPE=int8_float32 \
    UVICORN_WORKERS=1

# Pre-download the default model during build so the first pod starts instantly.
# Override WHISPER_MODEL_SIZE at build time to bake in a different model.
# Skip with --build-arg PRELOAD_MODEL=false to keep the image small.
ARG PRELOAD_MODEL=true
RUN if [ "$PRELOAD_MODEL" = "true" ]; then \
        python -c "from faster_whisper import WhisperModel; WhisperModel('${WHISPER_MODEL_SIZE}', device='cpu', compute_type='float32')"; \
    fi

USER 1000

EXPOSE 8000

# UVICORN_WORKERS controls how many uvicorn worker processes to spawn.
# Each worker loads the model independently, so RAM = workers × model_size.
# For CPU-only: set UVICORN_WORKERS=1 and tune WHISPER_MAX_WORKERS instead.
# For high-throughput GPU nodes: keep UVICORN_WORKERS=1, WHISPER_MAX_WORKERS=1.
CMD ["sh", "-c", "uvicorn whisper_streamer.main:app --host 0.0.0.0 --port 8000 --workers ${UVICORN_WORKERS}"]
