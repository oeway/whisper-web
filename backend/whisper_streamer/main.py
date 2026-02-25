import asyncio
import io
import os
import platform
import struct
import tempfile
import time
import logging
import concurrent.futures
from pathlib import Path
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, File, HTTPException, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles


# ---------------------------------------------------------------------------
# Detect backend engine: MLX on Apple Silicon, faster-whisper elsewhere
# ---------------------------------------------------------------------------
USE_MLX = False
_is_apple_silicon = platform.system() == "Darwin" and platform.machine() == "arm64"

if _is_apple_silicon:
    try:
        import mlx_whisper  # noqa: F401
        USE_MLX = True
    except ImportError:
        pass

if not USE_MLX:
    from faster_whisper import WhisperModel

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s: %(message)s",
)
log = logging.getLogger(__name__)
log.info(
    "Whisper backend: %s",
    "mlx-whisper (Apple Silicon GPU)" if USE_MLX else "faster-whisper (CPU/CUDA)",
)

FRONTEND_DIR = Path(__file__).resolve().parents[2] / "frontend"
DEFAULT_MODEL_SIZE = os.getenv("WHISPER_MODEL_SIZE", "small")
DEFAULT_DEVICE = os.getenv("WHISPER_DEVICE", "auto")
DEFAULT_COMPUTE_TYPE = os.getenv("WHISPER_COMPUTE_TYPE", "float32")
ALLOWED_MODELS = {"tiny", "base", "small", "medium", "large-v3"}

MLX_MODEL_MAP = {
    "tiny": "mlx-community/whisper-tiny-mlx",
    "base": "mlx-community/whisper-base-mlx",
    "small": "mlx-community/whisper-small-mlx",
    "medium": "mlx-community/whisper-medium-mlx",
    "large-v3": "mlx-community/whisper-large-v3-mlx",
}

# Single worker thread so transcription calls are serialised.
_transcription_executor = concurrent.futures.ThreadPoolExecutor(
    max_workers=1, thread_name_prefix="whisper"
)

# In-memory session store: session_id -> list of raw WAV bytes
_sessions: Dict[str, List[bytes]] = {}


# ---------------------------------------------------------------------------
# WAV concatenation helper
# ---------------------------------------------------------------------------
def _concat_wav_chunks(chunks: List[bytes]) -> bytes:
    """Concatenate multiple WAV files into a single WAV.

    Reads the header from the first chunk to determine sample rate / format,
    then strips headers from subsequent chunks and writes one combined file.
    """
    if not chunks:
        return b""
    if len(chunks) == 1:
        return chunks[0]

    # Parse header from the first chunk (standard 44-byte PCM WAV header)
    first = chunks[0]
    if len(first) < 44:
        return first
    sample_rate = struct.unpack_from("<I", first, 24)[0]
    num_channels = struct.unpack_from("<H", first, 22)[0]
    bits_per_sample = struct.unpack_from("<H", first, 34)[0]
    block_align = num_channels * (bits_per_sample // 8)
    byte_rate = sample_rate * block_align

    # Collect PCM data from all chunks (skip 44-byte headers)
    pcm_parts: List[bytes] = []
    for wav_bytes in chunks:
        if len(wav_bytes) > 44:
            pcm_parts.append(wav_bytes[44:])
        else:
            pcm_parts.append(wav_bytes)

    pcm_data = b"".join(pcm_parts)
    data_size = len(pcm_data)

    # Build new WAV header
    buf = io.BytesIO()
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))                # PCM chunk size
    buf.write(struct.pack("<H", 1))                 # PCM format
    buf.write(struct.pack("<H", num_channels))
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", byte_rate))
    buf.write(struct.pack("<H", block_align))
    buf.write(struct.pack("<H", bits_per_sample))
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(pcm_data)
    return buf.getvalue()


# ---------------------------------------------------------------------------
# Model pool
# ---------------------------------------------------------------------------
class ModelPool:
    def __init__(self, device: str, compute_type: str):
        self._device = device
        self._compute_type = compute_type
        self._models: Dict[str, Any] = {}
        self._lock = asyncio.Lock()

    async def get(self, model_size: str):
        if model_size not in ALLOWED_MODELS:
            model_size = DEFAULT_MODEL_SIZE
        if USE_MLX:
            return MLX_MODEL_MAP.get(model_size, MLX_MODEL_MAP["small"])
        return await self._get_ctranslate(model_size)

    async def _get_ctranslate(self, model_size: str):
        if model_size in self._models:
            return self._models[model_size]
        async with self._lock:
            if model_size in self._models:
                return self._models[model_size]
            loop = asyncio.get_event_loop()
            model = await loop.run_in_executor(
                _transcription_executor, self._load_ctranslate, model_size
            )
            self._models[model_size] = model
            return model

    def _load_ctranslate(self, model_size: str):
        log.info("Loading faster-whisper model %s ...", model_size)
        try:
            m = WhisperModel(
                model_size,
                device=self._device,
                compute_type=self._compute_type,
            )
        except Exception as exc:
            log.warning(
                "Primary load failed for %s (%s/%s), retrying float32: %s",
                model_size, self._device, self._compute_type, exc,
            )
            m = WhisperModel(model_size, device=self._device, compute_type="float32")
        log.info("faster-whisper model %s ready", model_size)
        return m


# ---------------------------------------------------------------------------
# Transcription helpers
# ---------------------------------------------------------------------------
def _transcribe_mlx(path: str, repo: str, language: Optional[str], prompt: Optional[str]) -> str:
    import mlx_whisper as _mlx_whisper
    kwargs: dict = {"path_or_hf_repo": repo}
    if language:
        kwargs["language"] = language
    if prompt:
        kwargs["initial_prompt"] = prompt
    result = _mlx_whisper.transcribe(path, **kwargs)
    return (result.get("text") or "").strip()


def _transcribe_ctranslate(path: str, model, language: Optional[str], prompt: Optional[str]) -> str:
    try:
        segments, _ = model.transcribe(
            path, beam_size=1, language=language,
            initial_prompt=prompt or None, vad_filter=False,
        )
    except ValueError as exc:
        if "max() arg is an empty sequence" not in str(exc):
            raise
        segments, _ = model.transcribe(
            path, beam_size=1, language=language,
            initial_prompt=prompt or None, vad_filter=False,
        )
    return " ".join(seg.text.strip() for seg in segments if seg.text.strip())


def transcribe_bytes(model_or_repo, audio_bytes: bytes, suffix: str,
                     language: Optional[str], prompt: Optional[str]) -> str:
    lang = language if language and language != "auto" else None
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=True) as tmp:
        tmp.write(audio_bytes)
        tmp.flush()
        if USE_MLX:
            return _transcribe_mlx(tmp.name, model_or_repo, lang, prompt)
        return _transcribe_ctranslate(tmp.name, model_or_repo, lang, prompt)


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
model_pool = ModelPool(device=DEFAULT_DEVICE, compute_type=DEFAULT_COMPUTE_TYPE)
app = FastAPI(title="Whisper Transcription Service")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"], allow_credentials=True,
    allow_methods=["*"], allow_headers=["*"],
)


@app.on_event("shutdown")
async def _shutdown():
    _transcription_executor.shutdown(wait=False)


@app.get("/health")
async def health():
    return {
        "status": "ok",
        "backend": "mlx-whisper" if USE_MLX else "faster-whisper",
        "default_model": DEFAULT_MODEL_SIZE,
        "device": "apple-silicon-gpu" if USE_MLX else DEFAULT_DEVICE,
        "compute_type": "mlx-float16" if USE_MLX else DEFAULT_COMPUTE_TYPE,
    }


@app.post("/api/session/{session_id}/chunk")
async def upload_chunk(
    session_id: str,
    chunk: UploadFile = File(...),
):
    """Receive a single WAV chunk and append it to the session buffer."""
    audio_bytes = await chunk.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty chunk")
    if session_id not in _sessions:
        _sessions[session_id] = []
    _sessions[session_id].append(audio_bytes)
    total = sum(len(c) for c in _sessions[session_id])
    return {
        "chunks": len(_sessions[session_id]),
        "total_bytes": total,
    }


@app.post("/api/session/{session_id}/transcribe")
async def transcribe_session(
    session_id: str,
    model_size: str = DEFAULT_MODEL_SIZE,
    language: str = "auto",
    prompt: str = "",
):
    """Concatenate all chunks for this session and transcribe as one audio."""
    chunks = _sessions.pop(session_id, [])
    if not chunks:
        raise HTTPException(status_code=404, detail="no audio for this session")

    combined = _concat_wav_chunks(chunks)
    file_size = len(combined)
    num_chunks = len(chunks)

    model_or_repo = await model_pool.get(model_size)
    t0 = time.monotonic()

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        _transcription_executor,
        transcribe_bytes,
        model_or_repo, combined, ".wav", language, prompt,
    )

    elapsed = time.monotonic() - t0
    log.info(
        "Transcribed %d chunks (%d bytes) in %.2fs (model=%s, lang=%s)",
        num_chunks, file_size, elapsed, model_size, language,
    )
    return {
        "text": text,
        "chunks": num_chunks,
        "file_size_bytes": file_size,
        "processing_time_s": round(elapsed, 3),
        "model": model_size,
        "language": language,
        "prompt": prompt,
    }


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
