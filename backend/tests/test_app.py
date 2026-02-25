import asyncio
from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

import whisper_streamer.main as main_module
from whisper_streamer.main import app, _sessions


@pytest.fixture()
def client():
    return TestClient(app)


@pytest.fixture(autouse=True)
def clear_sessions():
    _sessions.clear()
    yield
    _sessions.clear()


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "backend" in data
    assert "auth_required" in data


def test_chunk_upload(client):
    files = {"chunk": ("chunk.wav", b"RIFF" + b"\x00" * 100, "audio/wav")}
    resp = client.post("/api/session/test-1/chunk", files=files)
    assert resp.status_code == 200
    assert resp.json()["chunks"] == 1

    resp = client.post("/api/session/test-1/chunk", files=files)
    assert resp.json()["chunks"] == 2


def test_chunk_empty(client):
    files = {"chunk": ("chunk.wav", b"", "audio/wav")}
    resp = client.post("/api/session/test-1/chunk", files=files)
    assert resp.status_code == 400


def test_transcribe_no_session(client):
    resp = client.post("/api/session/nonexistent/transcribe?model_size=tiny")
    assert resp.status_code == 404


def test_transcribe_invalid_model(client):
    resp = client.post("/api/session/nonexistent/transcribe?model_size=bogus")
    assert resp.status_code == 400
    assert "invalid model_size" in resp.json()["detail"]


def test_transcribe_returns_stats(client, monkeypatch):
    monkeypatch.setattr(
        "whisper_streamer.main.transcribe_bytes",
        lambda *args, **kwargs: "hello world",
    )
    # Upload two chunks
    wav = b"RIFF" + b"\x00" * 100
    client.post("/api/session/s1/chunk", files={"chunk": ("c.wav", wav, "audio/wav")})
    client.post("/api/session/s1/chunk", files={"chunk": ("c.wav", wav, "audio/wav")})

    resp = client.post("/api/session/s1/transcribe?model_size=tiny&language=en&prompt=test")
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "hello world"
    assert data["chunks"] == 2
    assert "file_size_bytes" in data
    assert "processing_time_s" in data
    assert data["model"] == "tiny"
    assert data["prompt"] == "test"


def test_single_shot_transcribe(client, monkeypatch):
    monkeypatch.setattr(
        "whisper_streamer.main.transcribe_bytes",
        lambda *args, **kwargs: "single shot",
    )
    wav = b"RIFF" + b"\x00" * 100
    resp = client.post(
        "/api/transcribe?model_size=tiny&language=en",
        files={"audio": ("test.wav", wav, "audio/wav")},
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "single shot"
    assert data["model"] == "tiny"
    assert "file_size_bytes" in data
    assert "processing_time_s" in data


def test_single_shot_empty(client):
    resp = client.post(
        "/api/transcribe?model_size=tiny",
        files={"audio": ("test.wav", b"", "audio/wav")},
    )
    assert resp.status_code == 400


def test_single_shot_invalid_model(client):
    wav = b"RIFF" + b"\x00" * 100
    resp = client.post(
        "/api/transcribe?model_size=invalid",
        files={"audio": ("test.wav", wav, "audio/wav")},
    )
    assert resp.status_code == 400
    assert "invalid model_size" in resp.json()["detail"]


# ---------------------------------------------------------------------------
# Auth tests
# ---------------------------------------------------------------------------
def test_auth_required_rejects_without_token(client, monkeypatch):
    """When REQUIRE_AUTH=true, requests without a token get 401."""
    monkeypatch.setattr(main_module, "REQUIRE_AUTH", True)
    resp = client.get("/api/session/nonexistent/transcribe")
    # POST endpoints
    wav = b"RIFF" + b"\x00" * 100
    resp = client.post("/api/session/s1/chunk", files={"chunk": ("c.wav", wav, "audio/wav")})
    assert resp.status_code == 401


def test_auth_required_accepts_valid_token(client, monkeypatch):
    """When REQUIRE_AUTH=true and a valid token is provided, request succeeds."""
    monkeypatch.setattr(main_module, "REQUIRE_AUTH", True)
    monkeypatch.setattr(
        main_module, "_validate_token",
        AsyncMock(return_value={"id": "test-user", "email": "test@example.com", "roles": []}),
    )
    wav = b"RIFF" + b"\x00" * 100
    resp = client.post(
        "/api/session/s1/chunk",
        files={"chunk": ("c.wav", wav, "audio/wav")},
        headers={"Authorization": "Bearer fake-token"},
    )
    assert resp.status_code == 200


def test_auth_not_required_allows_anonymous(client, monkeypatch):
    """When REQUIRE_AUTH=false, requests without tokens work fine."""
    monkeypatch.setattr(main_module, "REQUIRE_AUTH", False)
    wav = b"RIFF" + b"\x00" * 100
    resp = client.post(
        "/api/session/s1/chunk",
        files={"chunk": ("c.wav", wav, "audio/wav")},
    )
    assert resp.status_code == 200
