"""
Pydantic schemas for the Anti-Spoofing API.

Defines request/response models for type-safe API contracts.
"""

from pydantic import BaseModel, Field
from typing import List, Literal, Optional


class SpoofDetectionResponse(BaseModel):
    """Response from the spoofing detection endpoint."""
    classification: Literal["REAL", "AI_CLONE"] = Field(
        description="Binary classification result"
    )
    confidence: float = Field(
        ge=0.0, le=1.0,
        description="Confidence score (0.0 = uncertain, 1.0 = certain)"
    )
    diagnostic_flags: List[str] = Field(
        description="List of diagnostic flags explaining the decision"
    )
    processing_time_ms: float = Field(
        ge=0.0,
        description="Total processing time in milliseconds"
    )


class HealthResponse(BaseModel):
    """Response from the health check endpoint."""
    status: str = "healthy"
    model_loaded: bool = False
    device: str = "cpu"
    version: str = "1.0.0-mvp"


class ModelInfoResponse(BaseModel):
    """Response from the model info endpoint."""
    model_name: str = "EnsembleAntiSpoof"
    version: str = "1.0.0-mvp"
    architecture: str = "CNN + Bi-LSTM Ensemble with Attention Fusion"
    total_parameters: int = 0
    device: str = "cpu"
    supported_formats: List[str] = [".wav", ".mp3", ".flac", ".ogg", ".m4a"]
    max_audio_duration_sec: float = 30.0
    max_upload_size_mb: float = 10.0
    training_metrics: Optional[dict] = None


class ErrorResponse(BaseModel):
    """Standard error response."""
    error: str
    detail: str
