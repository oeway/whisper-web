# Whisper Web

Web-based voice transcription and text-to-speech powered by OpenAI's Whisper. Record audio in your browser, get transcripts back in seconds — then speak them back with neural TTS.

Supports **batch recording**, **live streaming transcription**, **single-shot file transcription**, and **text-to-speech** (via Edge TTS / Google TTS). Runs on **Apple Silicon GPUs** (via MLX), **Linux with CUDA** (via faster-whisper/whisperX), or CPU.

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

Toggle **Live stream** for real-time utterance-by-utterance transcription as you speak.

You can select a model size (tiny through large-v3), language, and an optional system prompt before recording.

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
| `WHISPER_REQUIRE_AUTH` | `false` | Require Hypha token for all API requests |
| `HYPHA_TOKEN_URL` | `https://hypha.aicell.io/public/services/ws/parse_token` | Hypha token validation endpoint |
| `WHISPER_ACTIVITY_LOG` | `activity.jsonl` | Path to the JSON Lines activity log |
| `TTS_BACKEND` | `edge-tts` | Default TTS backend (`edge-tts` or `gtts`) |
| `TTS_VOICE` | `en-US-AriaNeural` | Default edge-tts voice |
| `TTS_MAX_TEXT_LENGTH` | `5000` | Max characters per TTS request |
| `TTS_MAX_CONCURRENT` | `8` | Max concurrent TTS requests |

Example:

```bash
WHISPER_REQUIRE_AUTH=true WHISPER_MODEL_SIZE=medium uvicorn whisper_streamer.main:app --host 0.0.0.0 --port 8000
```

## Authentication

Set `WHISPER_REQUIRE_AUTH=true` to require a valid [Hypha](https://hypha.aicell.io) token on every API request. Tokens can be passed via:

- **Header:** `Authorization: Bearer <token>`
- **Query parameter:** `?token=<token>`

Both Auth0 JWTs and Hypha client-credentials tokens are supported. When auth is disabled (the default), tokens are still accepted and logged if provided.

```bash
curl -X POST "http://localhost:8000/api/transcribe?model_size=small" \
  -H "Authorization: Bearer $HYPHA_TOKEN" \
  -F "audio=@recording.wav"
```

## Activity Log

Every chunk upload and transcription is logged as a JSON line to `activity.jsonl` (configurable via `WHISPER_ACTIVITY_LOG`). Each entry includes:

- `timestamp`, `action` (`chunk_upload` or `transcribe`)
- `user_id`, `user_email` (when authenticated)
- `client_ip`, `session_id`, `model`, `language`
- `chunks`, `file_size_bytes`, `processing_time_s`, `text_length`

Example entry:

```json
{"timestamp": "2026-02-25T20:55:02Z", "action": "transcribe", "user_id": "github|478667", "user_email": "oeway007@gmail.com", "client_ip": "127.0.0.1", "session_id": "s1", "model": "small", "language": "en", "chunks": 5, "file_size_bytes": 162000, "processing_time_s": 0.82, "text_length": 45}
```

## API

### `POST /api/transcribe` (single-shot)

Upload a complete audio file and get the transcript back in one request.

**Query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model_size` | `small` | Whisper model to use |
| `language` | `auto` | Language code (`en`, `zh`, `ja`, etc.) or `auto` |
| `context` | `""` | Prior transcript text used as Whisper initial_prompt |

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
  "language": "en"
}
```

### `POST /api/stream/utterance` (live streaming)

Transcribe a single utterance in real time. The client SDK's streaming mode sends each VAD-detected utterance to this endpoint as the user speaks.

**Query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model_size` | `small` | Whisper model to use |
| `language` | `auto` | Language code or `auto` |
| `context` | `""` | Previous transcript text (used as Whisper prompt for continuity) |
| `sequence` | `0` | Client-side utterance sequence number |

**Body:** multipart form with an `audio` file field (WAV).

**Response:**

```json
{
  "text": "Hello how are you?",
  "sequence": 1,
  "file_size_bytes": 32000,
  "processing_time_s": 0.32,
  "model": "small",
  "language": "en"
}
```

### `POST /api/session/{session_id}/chunk` (batch upload)

Upload audio chunks incrementally during recording. Useful for long recordings with VAD filtering.

**Body:** multipart form with a `chunk` file field (WAV).

**Response:**

```json
{
  "chunks": 3,
  "total_bytes": 48000
}
```

### `POST /api/session/{session_id}/transcribe` (batch transcribe)

Concatenate all uploaded chunks for this session and transcribe as one audio. Consumes and removes the session.

**Query parameters:** same as `/api/transcribe`.

**Response:** same as `/api/transcribe`, plus `"chunks": 3`.

### `POST /api/tts` (text-to-speech)

Convert text to speech audio (MP3). Two backends are supported:

- **edge-tts** (default): Microsoft Edge neural voices — high quality, many languages
- **gtts**: Google Translate TTS — simpler, fewer options

**Body** (JSON):

| Field | Default | Description |
|---|---|---|
| `text` | *(required)* | Text to synthesize |
| `language` | `en` | Language code (`en`, `zh`, `ja`, etc.) |
| `backend` | `edge-tts` | TTS engine: `edge-tts` or `gtts` |
| `voice` | *(auto)* | Edge-tts voice name (e.g. `en-US-GuyNeural`) |
| `slow` | `false` | Slow speech (gtts only) |

**Response:** `audio/mpeg` stream (MP3). Headers include `X-TTS-Backend` and `X-TTS-Processing-Time`.

```bash
curl -X POST "http://localhost:8000/api/tts" \
  -H "Content-Type: application/json" \
  -d '{"text": "Hello world", "language": "en", "backend": "edge-tts"}' \
  --output speech.mp3
```

### `GET /api/tts/voices`

List all available edge-tts voices with metadata.

### `GET /api/health`

Returns server status and feature flags.

```json
{
  "status": "ready",
  "backend": "whisperX (VAD+alignment)",
  "default_model": "small",
  "loaded_models": ["small"],
  "device": "auto",
  "compute_type": "float32",
  "auth_required": false,
  "features": {
    "streaming": true,
    "word_alignment": true
  }
}
```

## JavaScript SDK

A browser ES module is served at `/whisper-client.js`. No build step required.

```js
import { WhisperClient } from 'http://localhost:8000/whisper-client.js';

const client = new WhisperClient({
  server: 'http://localhost:8000',  // default: same origin
  token: 'your-hypha-token',        // optional auth token
  model: 'small',                    // tiny | base | small | medium | large-v3
  language: 'auto',                  // language code or 'auto'
  prompt: '',                        // optional initial prompt
});
```

### Mode 1: Live Streaming (real-time)

Audio is transcribed utterance-by-utterance as the user speaks.

```js
client.onTranscript = (result) => {
  console.log(`[${result.sequence}] ${result.text}`);
};

await client.startStreaming({
  utteranceGapMs: 1500,   // silence gap that ends an utterance (default 1500ms)
  maxUtteranceMs: 25000,  // force-flush after 25s of continuous speech
});

// ... user speaks, onTranscript fires for each utterance ...

const summary = await client.stopStreaming();
console.log(summary.text);       // full accumulated transcript
console.log(summary.utterances); // total utterance count
```

### Mode 2: Batch Recording with VAD

Record with VAD-filtered chunk uploads, then transcribe everything at once.

```js
client.onChunkUploaded = (info) => console.log(info.chunks, info.totalBytes);
client.onStatusChange = (status) => console.log(status);

await client.startRecording();
// ... user speaks ...
const result = await client.stopRecording();
console.log(result.text);
```

### Mode 3: Single-shot Transcription

Transcribe an existing audio file (Blob, File, or fetch response).

```js
const result = await client.transcribe(audioBlob, 'meeting.wav');
console.log(result.text);
```

### Text-to-Speech

Convert text to speech using the TTS API (no SDK class needed — it's a simple fetch):

```js
const resp = await fetch('/api/tts', {
  method: 'POST',
  headers: { 'Content-Type': 'application/json' },
  body: JSON.stringify({ text: 'Hello world', language: 'en', backend: 'edge-tts' }),
});
const audio = new Audio(URL.createObjectURL(await resp.blob()));
audio.play();
```

### Properties & Callbacks

| Property | Type | Description |
|---|---|---|
| `client.recording` | `boolean` | Whether batch recording is active |
| `client.streaming` | `boolean` | Whether live streaming is active |
| `client.onTranscript` | `Function` | Called per utterance (streaming mode) |
| `client.onChunkUploaded` | `Function` | Called per chunk upload (batch mode) |
| `client.onStatusChange` | `Function` | Called on status transitions |
| `client.destroy()` | method | Release all resources |

The `GET /sdk` endpoint redirects to `/whisper-client.js` for discoverability.

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
│  ↓                │    JSON       │    → transcribe_bytes()  │
│  on stop: trigger │ ←──────────── │    → mlx / faster-whisper│
│  display result   │               │                          │
└──────────────────┘               └──────────────────────────┘
```

### Frontend

- **`frontend/whisper-client.js`** — ES module SDK (`WhisperClient` class). Contains all recording, VAD, WAV encoding, chunk streaming, and API interaction logic.
- **`frontend/index.html`** — Demo UI that imports the SDK. No build step, no dependencies beyond Tailwind CSS (CDN).

The SDK handles:

- **Recording:** Uses `getUserMedia` + `ScriptProcessor` to capture raw PCM audio.
- **VAD:** Adaptive noise floor with RMS threshold filters out silence.
- **Encoding:** PCM buffers encoded to WAV (16-bit mono) in the browser.
- **Chunk streaming:** Speech chunks uploaded to the server during recording.
- **Live streaming:** Utterance-by-utterance transcription with context chaining.
- **Transcription:** On stop, triggers server-side concatenation and transcription.

### Backend (`backend/whisper_streamer/main.py`)

A FastAPI application that auto-detects the best available Whisper backend at startup:

- **Apple Silicon** → `mlx-whisper`: Runs inference on the Metal GPU via Apple's MLX framework.
- **CUDA / CPU** → `faster-whisper` or `whisperX`: Uses CTranslate2 for inference. Auto-detects CUDA GPUs; falls back to CPU with `float32` if needed.

Key design decisions:

- **Configurable parallelism:** Transcription runs in a `ThreadPoolExecutor` with configurable workers (`WHISPER_MAX_WORKERS`, default 2).
- **Session TTL:** Idle sessions are automatically evicted after `WHISPER_SESSION_TTL` seconds (default 10 minutes) to prevent memory leaks.
- **Lazy model loading:** Models are downloaded and loaded on the first request that uses them, then cached in memory.
- **Back-pressure:** Requests beyond `MAX_QUEUE` are rejected with HTTP 429 to prevent unbounded queuing.
- **Inference timeout:** Individual requests time out after `WHISPER_INFERENCE_TIMEOUT` seconds (default 120s).

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
