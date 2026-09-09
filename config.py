"""
Central configuration for the Audio Anti-Spoofing MVP.
All hyperparameters, feature dimensions, and paths are defined here.
"""

import os
from pathlib import Path

# ─── Project Paths ───────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).parent.resolve()
DATA_DIR = PROJECT_ROOT / "data"
RAW_DATA_DIR = DATA_DIR / "raw"
REAL_AUDIO_DIR = RAW_DATA_DIR / "real"
SPOOF_AUDIO_DIR = RAW_DATA_DIR / "spoof"
PROCESSED_DATA_DIR = DATA_DIR / "processed"
MODEL_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
for d in [REAL_AUDIO_DIR, SPOOF_AUDIO_DIR, PROCESSED_DATA_DIR, MODEL_DIR]:
    d.mkdir(parents=True, exist_ok=True)

# ─── Audio Parameters ────────────────────────────────────────────────────────
SAMPLE_RATE = 16000          # 16kHz — standard for speech processing
AUDIO_DURATION_SEC = 3.0     # Fixed duration per sample
AUDIO_NUM_SAMPLES = int(SAMPLE_RATE * AUDIO_DURATION_SEC)  # 48000 samples

# ─── Feature Extraction Parameters ───────────────────────────────────────────
N_FFT = 2048                 # FFT window size
HOP_LENGTH = 512             # Hop length between frames
N_MELS = 128                 # Number of mel filter banks
N_MFCC = 40                  # Number of MFCC coefficients
FMIN = 0                     # Minimum frequency for mel scale
FMAX = 8000                  # Maximum frequency for mel scale

# Derived: number of time frames for fixed-length audio
NUM_TIME_FRAMES = 1 + (AUDIO_NUM_SAMPLES // HOP_LENGTH)  # ~94 frames for 3s

# ─── Dataset Parameters ──────────────────────────────────────────────────────
NUM_SAMPLES_PER_CLASS = 200  # Samples per class (real / spoof)
TRAIN_SPLIT = 0.70
VAL_SPLIT = 0.15
TEST_SPLIT = 0.15
RANDOM_SEED = 42

# ─── Model Hyperparameters ───────────────────────────────────────────────────
# CNN Branch
CNN_CHANNELS = [1, 32, 64, 128]
CNN_KERNEL_SIZE = 3
CNN_DROPOUT = 0.3
CNN_EMBEDDING_DIM = 128

# LSTM Branch
LSTM_INPUT_DIM = N_MFCC + 1  # MFCC (40) + phase derivative (1) = 41
LSTM_HIDDEN_DIM = 64
LSTM_NUM_LAYERS = 2
LSTM_DROPOUT = 0.3
LSTM_EMBEDDING_DIM = 128

# Ensemble
ENSEMBLE_HIDDEN_DIM = 64
NUM_CLASSES = 2
CLASS_NAMES = ["REAL", "AI_CLONE"]

# ─── Training Hyperparameters ────────────────────────────────────────────────
LEARNING_RATE = 1e-3
WEIGHT_DECAY = 1e-4
BATCH_SIZE = 16
NUM_EPOCHS = 50
EARLY_STOPPING_PATIENCE = 5
BEST_MODEL_PATH = MODEL_DIR / "best_model.pth"

# ─── Device Configuration ────────────────────────────────────────────────────
import torch
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# ─── API Configuration ───────────────────────────────────────────────────────
API_HOST = "0.0.0.0"
API_PORT = 8000
MAX_UPLOAD_SIZE_MB = 10
MAX_AUDIO_DURATION_SEC = 30.0
ALLOWED_AUDIO_EXTENSIONS = {".wav", ".mp3", ".flac", ".ogg", ".m4a"}

# ─── Diagnostic Thresholds ───────────────────────────────────────────────────
VOCODER_ARTIFACT_THRESHOLD = 0.65
PHASE_DISCONTINUITY_THRESHOLD = 0.60
TEMPORAL_SMOOTHNESS_THRESHOLD = 0.55
BORDERLINE_CONFIDENCE_LOW = 0.4
BORDERLINE_CONFIDENCE_HIGH = 0.6
