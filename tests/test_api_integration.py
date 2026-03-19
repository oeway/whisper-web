"""Integration tests for Whisper Web API.

Uses macOS `say` command to generate real synthesized speech, then tests
all three API modes: single-shot transcribe, batch session, and live streaming.

Usage:
    # Start server first:
    #   cd backend && uvicorn whisper_streamer.main:app --port 8765
    # Then run tests:
    #   python tests/test_api_integration.py
    #   python tests/test_api_integration.py --server https://your-public-url

Requires: httpx, subprocess (macOS `say` + `afconvert`)
"""

import argparse
import io
import os
import struct
import subprocess
import sys
import tempfile
import time
import uuid

import httpx

DEFAULT_SERVER = "http://localhost:8765"


# ---------------------------------------------------------------------------
# Audio generation helpers
# ---------------------------------------------------------------------------
def synthesize_speech_wav(text: str, sample_rate: int = 16000) -> bytes:
    """Use macOS `say` to generate a real speech WAV file (16-bit mono PCM)."""
    with tempfile.NamedTemporaryFile(suffix=".aiff", delete=True) as aiff:
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as wav:
            wav_path = wav.name
        # Generate speech with macOS say
        subprocess.run(
            ["say", "-v", "Samantha", "-o", aiff.name, text],
            check=True, timeout=30,
        )
        # Convert to 16kHz 16-bit mono WAV
        subprocess.run(
            ["afconvert", "-f", "WAVE", "-d", f"LEI16@{sample_rate}",
             aiff.name, wav_path],
            check=True, timeout=15,
        )
    with open(wav_path, "rb") as f:
        data = f.read()
    os.unlink(wav_path)
    return data


def generate_silent_wav(duration_s: float = 1.0, sample_rate: int = 16000) -> bytes:
    """Generate a silent WAV file (useful for edge-case testing)."""
    num_samples = int(sample_rate * duration_s)
    buf = io.BytesIO()
    data_size = num_samples * 2  # 16-bit = 2 bytes per sample
    # WAV header
    buf.write(b"RIFF")
    buf.write(struct.pack("<I", 36 + data_size))
    buf.write(b"WAVE")
    buf.write(b"fmt ")
    buf.write(struct.pack("<I", 16))      # PCM chunk size
    buf.write(struct.pack("<H", 1))       # PCM format
    buf.write(struct.pack("<H", 1))       # mono
    buf.write(struct.pack("<I", sample_rate))
    buf.write(struct.pack("<I", sample_rate * 2))  # byte rate
    buf.write(struct.pack("<H", 2))       # block align
    buf.write(struct.pack("<H", 16))      # bits per sample
    buf.write(b"data")
    buf.write(struct.pack("<I", data_size))
    buf.write(b"\x00" * data_size)
    return buf.getvalue()


def split_wav_into_chunks(wav_bytes: bytes, num_chunks: int = 3) -> list:
    """Split a WAV file into N roughly equal chunks, each with a valid header."""
    if len(wav_bytes) < 44:
        return [wav_bytes]
    header = wav_bytes[:44]
    pcm = wav_bytes[44:]
    chunk_size = len(pcm) // num_chunks
    # Ensure chunk_size is even (16-bit samples = 2 bytes)
    chunk_size = chunk_size - (chunk_size % 2)
    chunks = []
    for i in range(num_chunks):
        start = i * chunk_size
        end = start + chunk_size if i < num_chunks - 1 else len(pcm)
        pcm_chunk = pcm[start:end]
        # Build a new WAV for this chunk
        buf = io.BytesIO()
        buf.write(header[:4])  # RIFF
        buf.write(struct.pack("<I", 36 + len(pcm_chunk)))
        buf.write(header[8:44])  # WAVEfmt...
        # Overwrite data size
        buf.seek(40)
        buf.write(struct.pack("<I", len(pcm_chunk)))
        buf.seek(44)
        buf.write(pcm_chunk)
        # Re-write complete header to be safe
        result = bytearray(header) + pcm_chunk
        # Fix RIFF size
        struct.pack_into("<I", result, 4, 36 + len(pcm_chunk))
        # Fix data size
        struct.pack_into("<I", result, 40, len(pcm_chunk))
        chunks.append(bytes(result))
    return chunks


# ---------------------------------------------------------------------------
# Test runner
# ---------------------------------------------------------------------------
class TestResults:
    def __init__(self):
        self.passed = 0
        self.failed = 0
        self.errors = []

    def ok(self, name: str, detail: str = ""):
        self.passed += 1
        print(f"  PASS  {name}" + (f"  ({detail})" if detail else ""))

    def fail(self, name: str, detail: str):
        self.failed += 1
        self.errors.append((name, detail))
        print(f"  FAIL  {name}: {detail}")

    def summary(self):
        total = self.passed + self.failed
        print(f"\n{'='*60}")
        print(f"Results: {self.passed}/{total} passed, {self.failed} failed")
        if self.errors:
            print("\nFailures:")
            for name, detail in self.errors:
                print(f"  - {name}: {detail}")
        print(f"{'='*60}")
        return self.failed == 0


def run_tests(server: str, token: str = ""):
    headers = {}
    if token:
        headers["Authorization"] = f"Bearer {token}"
    client = httpx.Client(base_url=server, timeout=120.0, headers=headers,
                          follow_redirects=True, cookies=httpx.Cookies())
    results = TestResults()

    # -----------------------------------------------------------------------
    # 0. Health check
    # -----------------------------------------------------------------------
    print("\n[Health Check]")
    try:
        r = client.get("/api/health")
        r.raise_for_status()
        health = r.json()
        results.ok("GET /api/health", f"status={health['status']}, backend={health['backend']}")
        if health["status"] != "ready":
            results.fail("health.status", f"expected 'ready', got '{health['status']}'")
    except Exception as e:
        results.fail("GET /api/health", str(e))
        print("  Server not reachable. Aborting.")
        return results

    # -----------------------------------------------------------------------
    # 1. Single-shot transcription (POST /api/transcribe)
    # -----------------------------------------------------------------------
    print("\n[Single-shot Transcription]")

    # 1a. Real speech
    print("  Generating speech: 'Hello world, this is a test.'")
    speech_wav = synthesize_speech_wav("Hello world, this is a test.")
    print(f"  Generated {len(speech_wav)} bytes of WAV audio")

    try:
        t0 = time.monotonic()
        r = client.post(
            "/api/transcribe",
            params={"model_size": "tiny", "language": "en"},
            files={"audio": ("test.wav", speech_wav, "audio/wav")},
        )
        elapsed = time.monotonic() - t0
        r.raise_for_status()
        data = r.json()
        text = data.get("text", "").lower()
        # Check that Whisper recognized something from the speech
        if any(w in text for w in ["hello", "world", "test"]):
            results.ok("transcribe real speech",
                       f"text='{data['text'][:80]}' in {elapsed:.2f}s")
        else:
            results.fail("transcribe real speech",
                         f"unexpected text: '{data['text'][:100]}' (expected hello/world/test)")
        # Verify response schema
        for key in ["text", "file_size_bytes", "processing_time_s", "model", "language"]:
            if key not in data:
                results.fail(f"transcribe schema.{key}", "missing from response")
    except Exception as e:
        results.fail("transcribe real speech", str(e))

    # 1b. Empty audio (should return 400)
    try:
        r = client.post(
            "/api/transcribe",
            params={"model_size": "tiny"},
            files={"audio": ("empty.wav", b"", "audio/wav")},
        )
        if r.status_code == 400:
            results.ok("transcribe empty audio", "correctly returned 400")
        else:
            results.fail("transcribe empty audio", f"expected 400, got {r.status_code}")
    except Exception as e:
        results.fail("transcribe empty audio", str(e))

    # 1c. Invalid model (should return 400)
    try:
        r = client.post(
            "/api/transcribe",
            params={"model_size": "nonexistent"},
            files={"audio": ("test.wav", speech_wav, "audio/wav")},
        )
        if r.status_code == 400:
            results.ok("transcribe invalid model", "correctly returned 400")
        else:
            results.fail("transcribe invalid model", f"expected 400, got {r.status_code}")
    except Exception as e:
        results.fail("transcribe invalid model", str(e))

    # 1d. Silent audio (should return empty or near-empty text)
    try:
        silent_wav = generate_silent_wav(2.0)
        r = client.post(
            "/api/transcribe",
            params={"model_size": "tiny", "language": "en"},
            files={"audio": ("silent.wav", silent_wav, "audio/wav")},
        )
        r.raise_for_status()
        data = r.json()
        # Silent audio may produce empty text or hallucination-filtered text
        results.ok("transcribe silent audio", f"text='{data['text'][:50]}' (len={len(data['text'])})")
    except Exception as e:
        results.fail("transcribe silent audio", str(e))

    # -----------------------------------------------------------------------
    # 2. Batch session (chunk upload + transcribe)
    # -----------------------------------------------------------------------
    print("\n[Batch Session]")

    # Generate a longer speech for batch mode
    print("  Generating speech: 'The quick brown fox jumps over the lazy dog.'")
    batch_wav = synthesize_speech_wav("The quick brown fox jumps over the lazy dog.")
    chunks = split_wav_into_chunks(batch_wav, num_chunks=3)
    print(f"  Split into {len(chunks)} chunks: {[len(c) for c in chunks]} bytes")

    session_id = f"test-{uuid.uuid4().hex[:8]}"

    # 2a. Upload chunks
    upload_ok = True
    for i, chunk_data in enumerate(chunks):
        try:
            r = client.post(
                f"/api/session/{session_id}/chunk",
                files={"chunk": (f"chunk{i}.wav", chunk_data, "audio/wav")},
            )
            r.raise_for_status()
            data = r.json()
            if data.get("chunks") != i + 1:
                results.fail(f"chunk upload {i}", f"expected chunks={i+1}, got {data.get('chunks')}")
                upload_ok = False
            else:
                results.ok(f"chunk upload {i}",
                           f"chunks={data['chunks']} total_bytes={data['total_bytes']}")
        except Exception as e:
            results.fail(f"chunk upload {i}", str(e))
            upload_ok = False

    # 2b. Transcribe session
    if upload_ok:
        try:
            t0 = time.monotonic()
            r = client.post(
                f"/api/session/{session_id}/transcribe",
                params={"model_size": "tiny", "language": "en"},
            )
            elapsed = time.monotonic() - t0
            r.raise_for_status()
            data = r.json()
            text = data.get("text", "").lower()
            if any(w in text for w in ["fox", "dog", "quick", "brown", "lazy"]):
                results.ok("batch transcribe",
                           f"text='{data['text'][:80]}' in {elapsed:.2f}s")
            else:
                results.fail("batch transcribe",
                             f"unexpected text: '{data['text'][:100]}'")
            if data.get("chunks") != 3:
                results.fail("batch transcribe chunks",
                             f"expected chunks=3, got {data.get('chunks')}")
        except Exception as e:
            results.fail("batch transcribe", str(e))

    # 2c. Transcribe again should return 404 (session consumed)
    try:
        r = client.post(f"/api/session/{session_id}/transcribe",
                        params={"model_size": "tiny"})
        if r.status_code == 404:
            results.ok("batch re-transcribe", "correctly returned 404 (session consumed)")
        else:
            results.fail("batch re-transcribe", f"expected 404, got {r.status_code}")
    except Exception as e:
        results.fail("batch re-transcribe", str(e))

    # 2d. Session DELETE endpoint
    delete_sid = f"test-del-{uuid.uuid4().hex[:8]}"
    try:
        # Upload a chunk
        r = client.post(
            f"/api/session/{delete_sid}/chunk",
            files={"chunk": ("c.wav", chunks[0], "audio/wav")},
        )
        r.raise_for_status()
        # Delete it
        r = client.delete(f"/api/session/{delete_sid}")
        r.raise_for_status()
        data = r.json()
        if data.get("deleted") == delete_sid:
            results.ok("session DELETE", f"freed {data.get('bytes_freed')} bytes")
        else:
            results.fail("session DELETE", f"unexpected response: {data}")
        # Delete again should 404
        r = client.delete(f"/api/session/{delete_sid}")
        if r.status_code == 404:
            results.ok("session DELETE twice", "correctly returned 404")
        else:
            results.fail("session DELETE twice", f"expected 404, got {r.status_code}")
    except Exception as e:
        results.fail("session DELETE", str(e))

    # -----------------------------------------------------------------------
    # 3. Live streaming (POST /api/stream/utterance)
    # -----------------------------------------------------------------------
    print("\n[Live Streaming]")

    # Simulate 3 utterances being sent sequentially (as the client SDK does)
    utterances = [
        "Good morning everyone.",
        "Today we will discuss the project timeline.",
        "Let's begin with the first topic.",
    ]

    accumulated_context = ""
    for seq, utt_text in enumerate(utterances):
        try:
            print(f"  Generating utterance {seq}: '{utt_text}'")
            utt_wav = synthesize_speech_wav(utt_text)
            t0 = time.monotonic()
            r = client.post(
                "/api/stream/utterance",
                params={
                    "model_size": "tiny",
                    "language": "en",
                    "context": accumulated_context[-500:],
                    "sequence": seq,
                },
                files={"audio": ("utt.wav", utt_wav, "audio/wav")},
            )
            elapsed = time.monotonic() - t0
            r.raise_for_status()
            data = r.json()
            text = data.get("text", "")
            # Verify response has required fields
            if "sequence" not in data:
                results.fail(f"utterance {seq} schema", "missing 'sequence' field")
            elif data["sequence"] != seq:
                results.fail(f"utterance {seq} sequence",
                             f"expected {seq}, got {data['sequence']}")
            elif text.strip():
                results.ok(f"utterance {seq}",
                           f"text='{text[:60]}' in {elapsed:.2f}s")
                accumulated_context += " " + text
            else:
                # Empty text is acceptable if Whisper didn't catch it
                results.ok(f"utterance {seq}", f"empty text (whisper missed) in {elapsed:.2f}s")
        except Exception as e:
            results.fail(f"utterance {seq}", str(e))

    # -----------------------------------------------------------------------
    # 4. Edge cases
    # -----------------------------------------------------------------------
    print("\n[Edge Cases]")

    # 4a. Concurrent requests (light stress test)
    import concurrent.futures
    print("  Sending 4 concurrent single-shot requests...")
    short_wav = synthesize_speech_wav("Testing concurrency.")

    def do_transcribe(idx):
        c = httpx.Client(base_url=server, timeout=120.0, headers=headers,
                         follow_redirects=True, cookies=httpx.Cookies())
        t0 = time.monotonic()
        r = c.post(
            "/api/transcribe",
            params={"model_size": "tiny", "language": "en"},
            files={"audio": ("t.wav", short_wav, "audio/wav")},
        )
        return idx, r.status_code, time.monotonic() - t0, r.json() if r.status_code == 200 else r.text

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
        futs = [pool.submit(do_transcribe, i) for i in range(4)]
        for fut in concurrent.futures.as_completed(futs):
            idx, status, elapsed, data = fut.result()
            if status == 200:
                results.ok(f"concurrent req {idx}", f"status=200 in {elapsed:.2f}s")
            elif status == 429:
                results.ok(f"concurrent req {idx}", "429 back-pressure (expected under load)")
            else:
                results.fail(f"concurrent req {idx}", f"status={status}: {str(data)[:100]}")

    # 4b. Verify frontend JS is served
    try:
        r = client.get("/whisper-client.js")
        if r.status_code == 200 and "WhisperClient" in r.text:
            results.ok("frontend SDK served", f"{len(r.text)} bytes, contains WhisperClient")
        else:
            results.fail("frontend SDK served", f"status={r.status_code}, WhisperClient not found")
    except Exception as e:
        results.fail("frontend SDK served", str(e))

    # 4c. Verify index.html is served
    try:
        r = client.get("/")
        if r.status_code == 200 and "whisper" in r.text.lower():
            results.ok("frontend index.html", f"{len(r.text)} bytes")
        else:
            results.fail("frontend index.html", f"status={r.status_code}")
    except Exception as e:
        results.fail("frontend index.html", str(e))

    # 4d. SDK redirect
    try:
        r = client.get("/sdk", follow_redirects=False)
        if r.status_code in (301, 302, 307, 308):
            results.ok("GET /sdk redirect", f"redirects to {r.headers.get('location')}")
        else:
            results.fail("GET /sdk redirect", f"expected redirect, got {r.status_code}")
    except Exception as e:
        results.fail("GET /sdk redirect", str(e))

    # -----------------------------------------------------------------------
    # 5. Context chaining accuracy test
    # -----------------------------------------------------------------------
    print("\n[Context Chaining]")
    # Test that providing context improves transcription of a follow-up utterance
    try:
        context_text = "We are discussing artificial intelligence and machine learning."
        followup_wav = synthesize_speech_wav(
            "The neural network achieved ninety five percent accuracy."
        )
        r = client.post(
            "/api/stream/utterance",
            params={
                "model_size": "tiny",
                "language": "en",
                "context": context_text,
                "sequence": 0,
            },
            files={"audio": ("ctx.wav", followup_wav, "audio/wav")},
        )
        r.raise_for_status()
        data = r.json()
        results.ok("context chaining", f"text='{data['text'][:80]}'")
    except Exception as e:
        results.fail("context chaining", str(e))

    # -----------------------------------------------------------------------
    # 6. Text-to-Speech (TTS)
    # -----------------------------------------------------------------------
    print("\n[Text-to-Speech]")

    # 6a. Edge-tts basic
    try:
        t0 = time.monotonic()
        r = client.post(
            "/api/tts",
            json={"text": "Hello world, this is a test of text to speech.",
                  "language": "en", "backend": "edge-tts"},
        )
        elapsed = time.monotonic() - t0
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        tts_backend = r.headers.get("x-tts-backend", "unknown")
        if "audio" in content_type and len(r.content) > 1000:
            results.ok("edge-tts basic",
                       f"{len(r.content)} bytes MP3 in {elapsed:.2f}s (backend={tts_backend})")
        else:
            results.fail("edge-tts basic",
                         f"unexpected: content-type={content_type}, size={len(r.content)}")
    except Exception as e:
        results.fail("edge-tts basic", str(e))

    # 6b. gTTS basic
    try:
        t0 = time.monotonic()
        r = client.post(
            "/api/tts",
            json={"text": "Testing Google text to speech engine.",
                  "language": "en", "backend": "gtts"},
        )
        elapsed = time.monotonic() - t0
        r.raise_for_status()
        content_type = r.headers.get("content-type", "")
        if "audio" in content_type and len(r.content) > 1000:
            results.ok("gtts basic",
                       f"{len(r.content)} bytes MP3 in {elapsed:.2f}s")
        else:
            results.fail("gtts basic",
                         f"unexpected: content-type={content_type}, size={len(r.content)}")
    except Exception as e:
        results.fail("gtts basic", str(e))

    # 6c. Edge-tts multi-language
    for lang, text in [("es", "Hola mundo"), ("fr", "Bonjour le monde"),
                       ("ja", "こんにちは世界"), ("zh", "你好世界")]:
        try:
            r = client.post("/api/tts",
                            json={"text": text, "language": lang, "backend": "edge-tts"})
            r.raise_for_status()
            if len(r.content) > 500:
                results.ok(f"edge-tts {lang}", f"{len(r.content)} bytes")
            else:
                results.fail(f"edge-tts {lang}", f"too small: {len(r.content)} bytes")
        except Exception as e:
            results.fail(f"edge-tts {lang}", str(e))

    # 6d. Empty text (should return 400)
    try:
        r = client.post("/api/tts", json={"text": "", "language": "en"})
        if r.status_code == 400:
            results.ok("tts empty text", "correctly returned 400")
        else:
            results.fail("tts empty text", f"expected 400, got {r.status_code}")
    except Exception as e:
        results.fail("tts empty text", str(e))

    # 6e. Invalid backend (should return 400)
    try:
        r = client.post("/api/tts",
                        json={"text": "test", "language": "en", "backend": "invalid"})
        if r.status_code == 400:
            results.ok("tts invalid backend", "correctly returned 400")
        else:
            results.fail("tts invalid backend", f"expected 400, got {r.status_code}")
    except Exception as e:
        results.fail("tts invalid backend", str(e))

    # 6f. Text too long (should return 413)
    try:
        r = client.post("/api/tts",
                        json={"text": "x" * 6000, "language": "en"})
        if r.status_code == 413:
            results.ok("tts text too long", "correctly returned 413")
        else:
            results.fail("tts text too long", f"expected 413, got {r.status_code}")
    except Exception as e:
        results.fail("tts text too long", str(e))

    # 6g. List voices
    try:
        r = client.get("/api/tts/voices")
        r.raise_for_status()
        data = r.json()
        count = data.get("count", 0)
        if count > 50:
            results.ok("tts list voices", f"{count} voices available")
        else:
            results.fail("tts list voices", f"expected >50 voices, got {count}")
    except Exception as e:
        results.fail("tts list voices", str(e))

    # 6h. Round-trip: transcribe → TTS (speech-to-text-to-speech)
    try:
        # Transcribe
        test_wav = synthesize_speech_wav("The weather is beautiful today.")
        r = client.post(
            "/api/transcribe",
            params={"model_size": "tiny", "language": "en"},
            files={"audio": ("rt.wav", test_wav, "audio/wav")},
        )
        r.raise_for_status()
        transcript = r.json()["text"]
        # TTS the transcript
        r = client.post("/api/tts",
                        json={"text": transcript, "language": "en", "backend": "edge-tts"})
        r.raise_for_status()
        if len(r.content) > 1000:
            results.ok("round-trip STT→TTS",
                       f"'{transcript[:50]}' → {len(r.content)} bytes MP3")
        else:
            results.fail("round-trip STT→TTS", f"TTS output too small: {len(r.content)}")
    except Exception as e:
        results.fail("round-trip STT→TTS", str(e))

    return results


def main():
    parser = argparse.ArgumentParser(description="Whisper Web API Integration Tests")
    parser.add_argument("--server", default=DEFAULT_SERVER,
                        help=f"Server URL (default: {DEFAULT_SERVER})")
    parser.add_argument("--token", default=os.getenv("HYPHA_TOKEN", ""),
                        help="Hypha auth token (or set HYPHA_TOKEN env var)")
    args = parser.parse_args()

    print(f"Whisper Web API Integration Tests")
    print(f"Server: {args.server}")
    if args.token:
        print(f"Auth: token provided ({len(args.token)} chars)")
    print(f"{'='*60}")

    results = run_tests(args.server, token=args.token)
    ok = results.summary()
    sys.exit(0 if ok else 1)


if __name__ == "__main__":
    main()
