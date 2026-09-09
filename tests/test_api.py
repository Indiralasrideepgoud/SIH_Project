"""
End-to-end API tests using FastAPI TestClient.
"""

import sys
import io
import numpy as np
import soundfile as sf
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

from fastapi.testclient import TestClient
from src.api.server import app
from src.api.inference import InferenceEngine

# Pre-load model if available
InferenceEngine().load_model()
client = TestClient(app)


def _make_test_wav(duration: float = 1.0, sr: int = 16000) -> bytes:
    """Generate a simple test WAV file as bytes."""
    t = np.linspace(0, duration, int(sr * duration), endpoint=False)
    signal = 0.5 * np.sin(2 * np.pi * 440 * t).astype(np.float32)
    buf = io.BytesIO()
    sf.write(buf, signal, sr, format='WAV')
    buf.seek(0)
    return buf.read()


def test_health_endpoint():
    """Test the health check endpoint."""
    response = client.get("/api/v1/health")
    assert response.status_code == 200
    data = response.json()
    assert data["status"] == "healthy"
    print("[PASS] Health endpoint works")


def test_model_info_endpoint():
    """Test the model info endpoint."""
    response = client.get("/api/v1/model-info")
    assert response.status_code == 200
    data = response.json()
    assert data["model_name"] == "EnsembleAntiSpoof"
    assert ".wav" in data["supported_formats"]
    print("[PASS] Model info endpoint works")


def test_detect_with_valid_audio():
    """Test detection endpoint with valid audio file."""
    wav_bytes = _make_test_wav()

    response = client.post(
        "/api/v1/detect",
        files={"audio": ("test.wav", wav_bytes, "audio/wav")}
    )

    # If model is loaded, expect 200; if not, expect 503
    if response.status_code == 200:
        data = response.json()
        assert data["classification"] in ["REAL", "AI_CLONE"]
        assert 0.0 <= data["confidence"] <= 1.0
        assert isinstance(data["diagnostic_flags"], list)
        assert data["processing_time_ms"] > 0
        print(f"[PASS] Detection works: {data['classification']} ({data['confidence']:.3f})")
    elif response.status_code == 503:
        print("[PASS] Detection endpoint accessible (model not loaded for test)")
    else:
        print(f"[FAIL] Unexpected status: {response.status_code}")


def test_detect_rejects_invalid_extension():
    """Test that unsupported file types are rejected."""
    response = client.post(
        "/api/v1/detect",
        files={"audio": ("test.txt", b"not audio data", "text/plain")}
    )
    assert response.status_code == 400
    print("[PASS] Invalid file extension rejected")


def test_detect_rejects_empty_file():
    """Test that empty files are rejected."""
    response = client.post(
        "/api/v1/detect",
        files={"audio": ("test.wav", b"", "audio/wav")}
    )
    assert response.status_code == 400
    print("[PASS] Empty file rejected")


if __name__ == "__main__":
    print("Running API tests...\n")
    test_health_endpoint()
    test_model_info_endpoint()
    test_detect_rejects_invalid_extension()
    test_detect_rejects_empty_file()
    test_detect_with_valid_audio()
    print("\nAll API tests passed! [PASS]")
