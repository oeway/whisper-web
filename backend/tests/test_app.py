import pytest
from fastapi.testclient import TestClient

from whisper_streamer.main import app, transcribe_bytes


@pytest.fixture()
def client():
    return TestClient(app)


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    data = resp.json()
    assert data["status"] == "ok"
    assert "backend" in data
    assert "default_model" in data


def test_transcribe_empty_audio(client):
    files = {"audio": ("recording.wav", b"", "audio/wav")}
    resp = client.post("/api/transcribe", params={"audio_format": "audio/wav"}, files=files)
    assert resp.status_code == 400


def test_transcribe_returns_stats(client, monkeypatch):
    monkeypatch.setattr(
        "whisper_streamer.main.transcribe_bytes",
        lambda *args, **kwargs: "hello world",
    )

    files = {"audio": ("recording.wav", b"fake-audio-data", "audio/wav")}
    resp = client.post(
        "/api/transcribe",
        params={"model_size": "tiny", "language": "en", "audio_format": "audio/wav", "prompt": "test"},
        files=files,
    )
    assert resp.status_code == 200
    data = resp.json()
    assert data["text"] == "hello world"
    assert data["file_size_bytes"] == len(b"fake-audio-data")
    assert "processing_time_s" in data
    assert data["model"] == "tiny"
    assert data["language"] == "en"
    assert data["prompt"] == "test"
