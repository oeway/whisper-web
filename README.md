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

Example:

```bash
WHISPER_MODEL_SIZE=medium uvicorn whisper_streamer.main:app --host 0.0.0.0 --port 8000
```

## API

### `POST /api/transcribe`

Upload a complete audio recording and get the transcript back.

**Query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model_size` | `small` | Whisper model to use |
| `language` | `auto` | Language code (`en`, `zh`, `ja`, etc.) or `auto` |
| `audio_format` | | Hint: `audio/wav`, `audio/webm`, etc. |
| `prompt` | | System prompt passed to Whisper as `initial_prompt` |

**Body:** multipart form with an `audio` file field.

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
Browser                          Server
┌─────────────────┐             ┌─────────────────────────┐
│  getUserMedia()  │             │  FastAPI + Uvicorn       │
│  ↓               │             │                         │
│  ScriptProcessor │             │  POST /api/transcribe   │
│  (capture PCM)   │             │    ↓                    │
│  ↓               │   WAV POST  │  ModelPool.get()        │
│  encodeWav()     │ ──────────→ │    ↓                    │
│  ↓               │             │  transcribe_bytes()     │
│  fetch()         │             │    ↓                    │
│  ↓               │   JSON      │  mlx_whisper.transcribe │
│  display result  │ ←────────── │  or model.transcribe()  │
└─────────────────┘             └─────────────────────────┘
```

### Frontend (`frontend/index.html`)

A single HTML file with inline JavaScript. No build step, no dependencies beyond Tailwind CSS (loaded from CDN).

- **Recording:** Uses `getUserMedia` to access the microphone and a `ScriptProcessor` to capture raw PCM audio samples into a buffer.
- **Encoding:** When the user clicks stop, the PCM buffer is encoded into a WAV file (16-bit mono) entirely in the browser.
- **Upload:** The WAV is sent as a multipart POST to `/api/transcribe` with the selected model, language, and prompt.
- **Display:** The JSON response is parsed and the transcript text and performance stats are rendered.

### Backend (`backend/whisper_streamer/main.py`)

A FastAPI application that auto-detects the best available Whisper backend at startup:

- **Apple Silicon** → `mlx-whisper`: Runs inference on the Metal GPU via Apple's MLX framework. Models are loaded from `mlx-community/whisper-*-mlx` repos on HuggingFace.
- **CUDA / CPU** → `faster-whisper`: Uses CTranslate2 for inference. Auto-detects CUDA GPUs; falls back to CPU with `float32` if the requested compute type isn't supported.

Key design decisions:

- **Single transcription thread:** All Whisper calls run in a dedicated `ThreadPoolExecutor(max_workers=1)` to avoid concurrency issues with MLX/CTranslate2 and to keep GPU memory usage predictable.
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
