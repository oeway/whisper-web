# Whisper Web

Web-based voice transcription powered by OpenAI's Whisper. Record audio in your browser, get transcripts back in seconds.

Runs on **Apple Silicon GPUs** (via MLX) and **Linux with CUDA** (via faster-whisper). Falls back to CPU anywhere else.

## Quick Start

### 1. Clone

```bash
git clone https://github.com/oeway/whisper-web.git
cd whisper-web
```

### 2. Install

**Apple Silicon (M1/M2/M3/M4):**

```bash
cd backend
pip install -e ".[apple]"
```

**Linux with CUDA:**

```bash
cd backend
pip install -e ".[cuda]"
```

**CPU only (any platform):**

```bash
cd backend
pip install -e ".[cpu]"
```

### 3. Run

```bash
uvicorn whisper_streamer.main:app --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** in your browser.

### 4. Use

1. Click the mic button to start recording
2. Speak
3. Click the stop button
4. Wait for the transcript

You can select a model size (tiny through large-v3), language, and an optional system prompt before recording.

After transcription completes, the UI shows stats: recording duration, file size, processing time, and realtime speed multiplier.

## Configuration

Environment variables:

| Variable | Default | Description |
|---|---|---|
| `WHISPER_MODEL_SIZE` | `small` | Default model (`tiny`, `base`, `small`, `medium`, `large-v3`) |
| `WHISPER_DEVICE` | `auto` | Device for faster-whisper (`auto`, `cpu`, `cuda`) |
| `WHISPER_COMPUTE_TYPE` | `float32` | Compute type for faster-whisper (`float32`, `float16`, `int8`) |
| `WHISPER_MAX_WORKERS` | `2` | Max parallel transcription threads |
| `WHISPER_SESSION_TTL` | `600` | Seconds before idle sessions are evicted |
| `WHISPER_MAX_CHUNK_BYTES` | `52428800` | Max total session size in bytes (50 MB) |

Example:

```bash
WHISPER_MODEL_SIZE=medium WHISPER_MAX_WORKERS=4 uvicorn whisper_streamer.main:app --host 0.0.0.0 --port 8000
```

## API

### `POST /api/transcribe` (single-shot)

Upload a complete audio file and get the transcript back in one request. Simplest way to use the API.

**Query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model_size` | `small` | Whisper model to use |
| `language` | `auto` | Language code (`en`, `zh`, `ja`, etc.) or `auto` |
| `prompt` | `""` | System prompt passed to Whisper as `initial_prompt` |

**Body:** multipart form with an `audio` file field.

```bash
curl -X POST "http://localhost:8000/api/transcribe?model_size=small&language=en" \
  -F "audio=@recording.wav"
```

**Response:**

```json
{
  "text": "Hello world this is a test.",
  "file_size_bytes": 170312,
  "processing_time_s": 0.42,
  "model": "small",
  "language": "en",
  "prompt": ""
}
```

### `POST /api/session/{session_id}/chunk` (streaming upload)

Upload audio chunks incrementally during recording. Useful for long recordings with VAD filtering.

**Body:** multipart form with a `chunk` file field (WAV).

**Response:**

```json
{
  "chunks": 3,
  "total_bytes": 48000
}
```

### `POST /api/session/{session_id}/transcribe`

Concatenate all uploaded chunks for this session and transcribe as one audio. Consumes and removes the session.

**Query parameters:** same as `/api/transcribe`.

**Response:** same as `/api/transcribe`, plus `"chunks": 3`.

### `GET /health`

Returns server status and active backend info.

```json
{
  "status": "ok",
  "backend": "mlx-whisper",
  "default_model": "small",
  "device": "apple-silicon-gpu",
  "compute_type": "mlx-float16"
}
```

## How It Works

### Architecture

```
Browser                              Server
┌──────────────────┐               ┌──────────────────────────┐
│  getUserMedia()   │               │  FastAPI + Uvicorn        │
│  ↓                │               │                          │
│  ScriptProcessor  │               │  POST .../chunk (×N)     │
│  (capture PCM)    │  WAV chunks   │    → session store       │
│  ↓                │ ────────────→ │                          │
│  VAD filter       │               │  POST .../transcribe     │
│  ↓                │               │    → concat WAV          │
│  encodeWav()      │               │    → ModelPool.get()     │
│  ↓                │               │    → transcribe_bytes()  │
│  on stop: trigger │    JSON       │    → mlx / faster-whisper│
│  display result   │ ←──────────── │                          │
└──────────────────┘               └──────────────────────────┘
```

### Frontend (`frontend/index.html`)

A single HTML file with inline JavaScript. No build step, no dependencies beyond Tailwind CSS (loaded from CDN).

- **Recording:** Uses `getUserMedia` to access the microphone and a `ScriptProcessor` to capture raw PCM audio samples into a buffer.
- **VAD:** Adaptive noise floor with RMS threshold filters out silence. Only speech chunks are sent to the server during recording.
- **Encoding:** PCM buffers are encoded into WAV (16-bit mono) in the browser and streamed as chunks.
- **Transcription:** On stop, the frontend triggers the server to concatenate all chunks and transcribe as one audio.
- **Display:** The JSON response is parsed and the transcript text and performance stats are rendered.

### Backend (`backend/whisper_streamer/main.py`)

A FastAPI application that auto-detects the best available Whisper backend at startup:

- **Apple Silicon** → `mlx-whisper`: Runs inference on the Metal GPU via Apple's MLX framework. Models are loaded from `mlx-community/whisper-*-mlx` repos on HuggingFace.
- **CUDA / CPU** → `faster-whisper`: Uses CTranslate2 for inference. Auto-detects CUDA GPUs; falls back to CPU with `float32` if the requested compute type isn't supported.

Key design decisions:

- **Configurable parallelism:** Transcription runs in a `ThreadPoolExecutor` with configurable workers (`WHISPER_MAX_WORKERS`, default 2). Increase for higher throughput on multi-GPU or CPU setups.
- **Session TTL:** Idle sessions are automatically evicted after `WHISPER_SESSION_TTL` seconds (default 10 minutes) to prevent memory leaks.
- **Lazy model loading:** Models are downloaded and loaded on the first request that uses them, then cached in memory for subsequent requests.
- **Automatic fallback:** If a compute type fails (e.g., `int8_float16` on CPU), the server retries with `float32`.

### System Prompt

The optional system prompt is passed to Whisper as `initial_prompt`. This helps with:

- **Proper nouns:** Providing names, acronyms, or domain terms helps Whisper spell them correctly.
- **Style guidance:** Hints about formatting or punctuation style.
- **Context:** Describing the topic can improve accuracy for domain-specific vocabulary.

### Supported Models

| Model | Parameters | Relative Speed | Best For |
|---|---|---|---|
| `tiny` | 39M | Fastest | Quick tests, low latency |
| `base` | 74M | Fast | Casual use |
| `small` | 244M | Balanced | General purpose (default) |
| `medium` | 769M | Slower | Higher accuracy |
| `large-v3` | 1.5B | Slowest | Maximum accuracy |

## License

MIT
