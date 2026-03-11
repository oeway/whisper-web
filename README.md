# Whisper Web

Web-based voice transcription powered by OpenAI's Whisper. Record audio in your browser, get transcripts back in seconds.

Supports **live streaming transcription**, **speaker diarization** (who said what), and **batch recording**. Runs on **Apple Silicon GPUs** (via MLX), **Linux with CUDA** (via faster-whisper/whisperX), or CPU.

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
| `WHISPER_REQUIRE_AUTH` | `false` | Require Hypha token for all API requests |
| `HYPHA_TOKEN_URL` | `https://hypha.aicell.io/public/services/ws/parse_token` | Hypha token validation endpoint |
| `WHISPER_BACKEND` | `auto` | Force backend (`auto`, `mlx`, `whisperx`, `faster-whisper`) |
| `WHISPER_HF_TOKEN` | — | HuggingFace token for pyannote speaker diarization |
| `WHISPERX_BATCH_SIZE` | `8` | Batch size for whisperX transcription |
| `WHISPER_ACTIVITY_LOG` | `activity.jsonl` | Path to the JSON Lines activity log |

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

Upload a complete audio file and get the transcript back in one request. Supports speaker diarization.

**Query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model_size` | `small` | Whisper model to use |
| `language` | `auto` | Language code (`en`, `zh`, `ja`, etc.) or `auto` |
| `prompt` | `""` | System prompt passed to Whisper as `initial_prompt` |
| `diarize` | `false` | Enable speaker diarization (whisperX backend + HF token required) |
| `align_words` | `false` | Enable word-level timestamps (whisperX only) |
| `min_speakers` | `1` | Minimum expected speakers (diarization hint) |
| `max_speakers` | `10` | Maximum expected speakers (diarization hint) |

**Body:** multipart form with an `audio` file field.

```bash
# Basic transcription
curl -X POST "http://localhost:8000/api/transcribe?model_size=small&language=en" \
  -F "audio=@recording.wav"

# With speaker diarization
curl -X POST "http://localhost:8000/api/transcribe?diarize=true&max_speakers=3" \
  -F "audio=@meeting.wav"
```

**Response:**

```json
{
  "text": "Hello world this is a test.",
  "file_size_bytes": 170312,
  "processing_time_s": 0.42,
  "model": "small",
  "language": "en",
  "segments": [
    {
      "start": 0.0, "end": 2.5,
      "text": "Hello world this is a test.",
      "speaker": "SPEAKER_00"
    }
  ]
}
```

### `POST /api/stream/utterance` (live streaming)

Transcribe a single utterance in real time. Designed for the client SDK's streaming mode where VAD detects speech boundaries and sends each utterance as it completes.

**Query parameters:**

| Parameter | Default | Description |
|---|---|---|
| `model_size` | `small` | Whisper model to use |
| `language` | `auto` | Language code or `auto` |
| `context` | `""` | Previous transcript text (used as Whisper prompt for continuity) |
| `sequence` | `0` | Client-side utterance sequence number |
| `diarize` | `false` | Enable speaker detection (adds ~1-3s latency per utterance) |
| `min_speakers` | `1` | Min speakers hint |
| `max_speakers` | `10` | Max speakers hint |

**Body:** multipart form with an `audio` file field (WAV).

**Response:**

```json
{
  "text": "Hello how are you?",
  "sequence": 1,
  "file_size_bytes": 32000,
  "processing_time_s": 0.32,
  "model": "small",
  "language": "en",
  "segments": [
    {
      "start": 0.0, "end": 1.8,
      "text": "Hello how are you?",
      "speaker": "SPEAKER_00"
    }
  ]
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

### `GET /api/health`

Returns server status, backend info, and feature flags.

```json
{
  "status": "ready",
  "backend": "whisperX (VAD+alignment+diarization)",
  "default_model": "small",
  "loaded_models": ["small"],
  "device": "auto",
  "compute_type": "float32",
  "auth_required": false,
  "features": {
    "diarization": true,
    "word_alignment": true,
    "streaming": true
  }
}
```

## JavaScript SDK

A browser ES module is served at `/whisper-client.js` for integrating transcription into your own web apps. No build step required.

```js
import { WhisperClient } from 'http://localhost:8000/whisper-client.js';

const client = new WhisperClient({
  server: 'http://localhost:8000',  // default: same origin
  token: 'your-hypha-token',        // optional auth token
  model: 'small',                    // tiny | base | small | medium | large-v3
  language: 'auto',                  // language code or 'auto'
  prompt: '',                        // optional initial prompt
  diarize: false,                    // enable speaker detection
  maxSpeakers: 10,                   // max expected speakers
});
```

### Mode 1: Live Streaming (real-time transcription)

Audio is transcribed utterance-by-utterance as the user speaks. Each completed utterance fires `onTranscript`.

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

### Mode 2: Live Streaming with Speaker Detection

Enable `diarize: true` to get speaker labels on each utterance in real time. Adds ~1-3s latency per utterance.

```js
const client = new WhisperClient({
  server: 'http://localhost:8000',
  diarize: true,
  maxSpeakers: 4,
});

client.onTranscript = (result) => {
  for (const seg of result.segments || []) {
    console.log(`${seg.speaker}: ${seg.text}`);
  }
};

client.onSpeakerChange = (info) => {
  console.log(`Speaker changed: ${info.previous} → ${info.current}`);
};

await client.startStreaming();
// ... conversation happens ...
const summary = await client.stopStreaming();
console.log(summary.speakers);       // ["SPEAKER_00", "SPEAKER_01"]
console.log(summary.speakerHistory); // [{speaker, sequence, text}, ...]
```

### Mode 3: Batch Recording with VAD

Record with VAD-filtered chunk uploads, then transcribe everything at once. Best for long recordings where you want the full context for maximum accuracy and diarization quality.

```js
client.onChunkUploaded = (info) => console.log(info.chunks, info.totalBytes);
client.onStatusChange = (status) => console.log(status);

await client.startRecording();
// ... user speaks ...
const result = await client.stopRecording();
console.log(result.text);
console.log(result.segments); // with speaker labels if diarize=true
```

### Mode 4: Single-shot Transcription

Transcribe an existing audio file (Blob, File, or fetch response).

```js
const result = await client.transcribe(audioBlob, 'meeting.wav');
console.log(result.text, result.segments);
```

### Properties & Callbacks

| Property | Type | Description |
|---|---|---|
| `client.recording` | `boolean` | Whether batch recording is active |
| `client.streaming` | `boolean` | Whether live streaming is active |
| `client.currentSpeaker` | `string\|null` | Last detected speaker label |
| `client.speakerHistory` | `Array` | History of speaker changes |
| `client.onTranscript` | `Function` | Called per utterance (streaming mode) |
| `client.onSpeakerChange` | `Function` | Called when speaker changes (streaming + diarize) |
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
│  ↓                │               │    → transcribe_bytes()  │
│  on stop: trigger │    JSON       │    → mlx / faster-whisper│
│  display result   │ ←──────────── │                          │
└──────────────────┘               └──────────────────────────┘
```

### Frontend

- **`frontend/whisper-client.js`** — ES module SDK (`WhisperClient` class). Contains all recording, VAD, WAV encoding, chunk streaming, and API interaction logic.
- **`frontend/whisper-client.d.ts`** — TypeScript type declarations for the SDK.
- **`frontend/index.html`** — Demo UI that imports the SDK. No build step, no dependencies beyond Tailwind CSS (CDN).

The SDK handles:

- **Recording:** Uses `getUserMedia` + `ScriptProcessor` to capture raw PCM audio.
- **VAD:** Adaptive noise floor with RMS threshold filters out silence.
- **Encoding:** PCM buffers encoded to WAV (16-bit mono) in the browser.
- **Chunk streaming:** Speech chunks uploaded to the server during recording.
- **Transcription:** On stop, triggers server-side concatenation and transcription.

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
