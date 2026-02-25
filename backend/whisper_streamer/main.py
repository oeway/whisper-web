import asyncio
import os
import platform
import tempfile
import time
import logging
import concurrent.futures
from pathlib import Path
from typing import Dict, Optional, Any

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


def _detect_suffix(format_hint: str, filename: str, content_type: str) -> str:
    fh, fn, ct = format_hint.lower(), filename.lower(), content_type.lower()
    if "wav" in fh or fn.endswith(".wav") or "wav" in ct:
        return ".wav"
    if "mp4" in fh or "m4a" in fh or fn.endswith(".mp4") or "mp4" in ct:
        return ".mp4"
    if "ogg" in fh or fn.endswith(".ogg") or "ogg" in ct:
        return ".ogg"
    if "webm" in fh or fn.endswith(".webm") or "webm" in ct:
        return ".webm"
    return ".webm"


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


@app.post("/api/transcribe")
async def transcribe(
    audio: UploadFile = File(...),
    model_size: str = DEFAULT_MODEL_SIZE,
    language: str = "auto",
    audio_format: str = "",
    prompt: str = "",
):
    """Accept a complete audio recording and return the transcript."""
    audio_bytes = await audio.read()
    if not audio_bytes:
        raise HTTPException(status_code=400, detail="empty audio")

    suffix = _detect_suffix(audio_format, audio.filename or "", audio.content_type or "")
    model_or_repo = await model_pool.get(model_size)

    file_size = len(audio_bytes)
    t0 = time.monotonic()

    loop = asyncio.get_event_loop()
    text = await loop.run_in_executor(
        _transcription_executor,
        transcribe_bytes,
        model_or_repo, audio_bytes, suffix, language, prompt,
    )

    elapsed = time.monotonic() - t0
    log.info(
        "Transcribed %s bytes in %.2fs (model=%s, lang=%s, prompt=%r)",
        file_size, elapsed, model_size, language, prompt[:40] if prompt else "",
    )
    return {
        "text": text,
        "file_size_bytes": file_size,
        "processing_time_s": round(elapsed, 3),
        "model": model_size,
        "language": language,
        "prompt": prompt,
    }


if FRONTEND_DIR.exists():
    app.mount("/", StaticFiles(directory=FRONTEND_DIR, html=True), name="frontend")
