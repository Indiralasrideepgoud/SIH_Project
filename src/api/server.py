"""
FastAPI Application — Audio Anti-Spoofing REST API.

Endpoints:
  POST /api/v1/detect   — Upload audio, get spoofing detection result
  GET  /api/v1/health   — Health check
  GET  /api/v1/model-info — Model metadata and architecture info

Security:
  - File type validation (whitelist)
  - File size limits (10MB max)
  - Audio duration limits (30s max)
  - Input sanitization
"""

import os
import sys
from pathlib import Path
from contextlib import asynccontextmanager

from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.responses import HTMLResponse
from fastapi.middleware.cors import CORSMiddleware

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
import config
from src.api.schemas import (
    SpoofDetectionResponse,
    HealthResponse,
    ModelInfoResponse,
    ErrorResponse,
)
from src.api.inference import InferenceEngine
from src.api.web_ui import HTML_CONTENT


# ─── Lifespan: load model at startup ─────────────────────────────────────────
@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load model into memory on startup."""
    engine = InferenceEngine()
    success = engine.load_model()
    if success:
        print(f"[API] Model loaded successfully on {config.DEVICE}")
    else:
        print(f"[API] WARNING: Model not loaded. Train the model first.")
    yield
    # Cleanup (if needed)
    print("[API] Shutting down...")


# ─── FastAPI Application ─────────────────────────────────────────────────────
app = FastAPI(
    title="Audio Anti-Spoofing API",
    description=(
        "Enterprise-grade AI voice clone & deepfake detector. "
        "Detects synthetic speech (TTS), voice conversion, and replay attacks "
        "using an ensemble CNN+LSTM deep learning architecture."
    ),
    version="1.0.0-mvp",
    lifespan=lifespan,
)

# CORS middleware (restrict in production)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ─── Endpoints ───────────────────────────────────────────────────────────────

@app.get("/", response_class=HTMLResponse, include_in_schema=False)
async def web_ui():
    """Serve the presentation-ready forensic detector dashboard."""
    return HTML_CONTENT

@app.post(
    "/api/v1/detect",
    response_model=SpoofDetectionResponse,
    responses={
        400: {"model": ErrorResponse},
        413: {"model": ErrorResponse},
        503: {"model": ErrorResponse},
    },
    summary="Detect AI Voice Clones & Deepfakes",
    description=(
        "Upload an audio file to detect whether it contains genuine human "
        "speech or AI-generated synthetic voice (TTS/voice conversion). "
        "Returns classification, confidence score, and diagnostic flags."
    ),
)
async def detect_spoofing(
    audio: UploadFile = File(..., description="Audio file (.wav, .mp3, .flac, .ogg, .m4a)")
):
    """Main spoofing detection endpoint."""
    engine = InferenceEngine()

    # Validate file extension
    filename = audio.filename or ""
    ext = Path(filename).suffix.lower()
    if ext not in config.ALLOWED_AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=400,
            detail=(
                f"Unsupported file type: '{ext}'. "
                f"Allowed: {', '.join(config.ALLOWED_AUDIO_EXTENSIONS)}"
            ),
        )

    # Read and validate file size
    audio_bytes = await audio.read()
    max_bytes = config.MAX_UPLOAD_SIZE_MB * 1024 * 1024
    if len(audio_bytes) > max_bytes:
        raise HTTPException(
            status_code=413,
            detail=(
                f"File too large: {len(audio_bytes) / 1e6:.1f}MB. "
                f"Maximum: {config.MAX_UPLOAD_SIZE_MB}MB"
            ),
        )

    if len(audio_bytes) == 0:
        raise HTTPException(
            status_code=400,
            detail="Empty audio file received.",
        )

    # Validate model is loaded
    if not engine.is_loaded:
        raise HTTPException(
            status_code=503,
            detail="Model not loaded. Please train the model first.",
        )

    # Run inference
    try:
        result = engine.predict(audio_bytes)
    except Exception as e:
        raise HTTPException(
            status_code=400,
            detail=f"Failed to process audio: {str(e)}",
        )

    return SpoofDetectionResponse(**result)


@app.get(
    "/api/v1/health",
    response_model=HealthResponse,
    summary="Health Check",
)
async def health_check():
    """Check API and model health status."""
    engine = InferenceEngine()
    return HealthResponse(
        status="healthy",
        model_loaded=engine.is_loaded,
        device=str(config.DEVICE),
    )


@app.get(
    "/api/v1/model-info",
    response_model=ModelInfoResponse,
    summary="Model Information",
)
async def model_info():
    """Get model architecture and configuration details."""
    engine = InferenceEngine()
    params = engine.model.count_parameters() if engine.is_loaded else 0

    return ModelInfoResponse(
        model_name="EnsembleAntiSpoof",
        version="1.0.0-mvp",
        architecture="CNN + Bi-LSTM Ensemble with Attention Fusion",
        total_parameters=params,
        device=str(config.DEVICE),
        supported_formats=[".wav", ".mp3", ".flac", ".ogg", ".m4a"],
        max_audio_duration_sec=config.MAX_AUDIO_DURATION_SEC,
        max_upload_size_mb=config.MAX_UPLOAD_SIZE_MB,
    )
