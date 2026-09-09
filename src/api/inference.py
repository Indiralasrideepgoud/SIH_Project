"""
Inference Pipeline — Single-File Audio Spoofing Detection.

Provides a thread-safe singleton model loader and a complete inference
pipeline: audio bytes → preprocess → extract features → model forward →
post-process → classification + confidence + diagnostics.
"""

import time
import torch
import numpy as np
from pathlib import Path
from typing import Dict, Optional
import threading
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
import config
from src.models.ensemble import EnsembleAntiSpoof
from src.models.diagnostics import DiagnosticAnalyzer
from src.feature_extraction.extractor import FeatureExtractor


class InferenceEngine:
    """
    Singleton inference engine for anti-spoofing detection.

    Loads the trained model once and provides thread-safe inference.
    """

    _instance = None
    _lock = threading.Lock()

    def __new__(cls):
        if cls._instance is None:
            with cls._lock:
                if cls._instance is None:
                    cls._instance = super().__new__(cls)
                    cls._instance._initialized = False
        return cls._instance

    def __init__(self):
        if self._initialized:
            return
        self._initialized = True

        self.model: Optional[EnsembleAntiSpoof] = None
        self.extractor = FeatureExtractor()
        self.diagnostics = DiagnosticAnalyzer()
        self.device = config.DEVICE
        self._model_lock = threading.Lock()

    def load_model(self, model_path: Path = config.BEST_MODEL_PATH) -> bool:
        """
        Load the trained model from checkpoint.

        Returns:
            True if model loaded successfully, False otherwise.
        """
        if not model_path.exists():
            print(f"[INFERENCE] Model not found at {model_path}")
            return False

        try:
            checkpoint = torch.load(
                str(model_path),
                map_location=self.device,
                weights_only=False,
            )

            # Reconstruct model from saved config
            saved_config = checkpoint.get("config", {})
            self.model = EnsembleAntiSpoof(
                cnn_embedding_dim=saved_config.get("cnn_embedding_dim",
                                                    config.CNN_EMBEDDING_DIM),
                lstm_embedding_dim=saved_config.get("lstm_embedding_dim",
                                                     config.LSTM_EMBEDDING_DIM),
                hidden_dim=saved_config.get("ensemble_hidden_dim",
                                            config.ENSEMBLE_HIDDEN_DIM),
                num_classes=saved_config.get("num_classes", config.NUM_CLASSES),
            )
            self.model.load_state_dict(checkpoint["model_state_dict"])
            self.model.to(self.device)
            self.model.eval()

            params = self.model.count_parameters()
            val_acc = checkpoint.get("val_accuracy", "N/A")
            print(f"[INFERENCE] Model loaded: {params:,} params, "
                  f"val_acc: {val_acc}, device: {self.device}")
            return True

        except Exception as e:
            print(f"[INFERENCE] Failed to load model: {e}")
            return False

    @property
    def is_loaded(self) -> bool:
        return self.model is not None

    def predict(self, audio_bytes: bytes) -> Dict:
        """
        Run full inference pipeline on raw audio bytes.

        Args:
            audio_bytes: Raw audio file bytes (wav, mp3, flac, etc.)

        Returns:
            Dict with keys: classification, confidence, diagnostic_flags,
                            processing_time_ms
        """
        if not self.is_loaded:
            raise RuntimeError("Model not loaded. Call load_model() first.")

        start_time = time.time()

        # 1. Load and preprocess audio
        waveform = self.extractor.load_audio_from_bytes(audio_bytes)

        # 2. Extract model features
        features = self.extractor.extract_all_features(waveform)

        # 3. Extract diagnostic features
        diag_features = self.extractor.extract_diagnostic_features(waveform)

        # 4. Model inference (thread-safe)
        with self._model_lock:
            with torch.no_grad():
                mel_spec = features["mel_spectrogram"].unsqueeze(0).to(self.device)
                mfcc_phase = features["mfcc_phase"].unsqueeze(0).to(self.device)

                probs = self.model.predict_proba(mel_spec, mfcc_phase)
                probs = probs.cpu().numpy()[0]  # [P(REAL), P(AI_CLONE)]

        # 5. Post-process
        predicted_class = int(np.argmax(probs))
        classification = config.CLASS_NAMES[predicted_class]
        confidence = float(probs[predicted_class])

        # 6. Generate diagnostic flags
        flags = self.diagnostics.analyze(
            diagnostic_features=diag_features,
            confidence=confidence,
            classification=classification,
        )

        processing_time = (time.time() - start_time) * 1000  # ms

        return {
            "classification": classification,
            "confidence": round(confidence, 4),
            "diagnostic_flags": flags,
            "processing_time_ms": round(processing_time, 2),
        }

    def predict_file(self, filepath: str) -> Dict:
        """
        Run inference on a file path (convenience method).

        Args:
            filepath: Path to audio file.

        Returns:
            Same as predict()
        """
        with open(filepath, "rb") as f:
            audio_bytes = f.read()
        return self.predict(audio_bytes)
