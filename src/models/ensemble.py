"""
Ensemble Fusion Model — CNN + LSTM Anti-Spoofing Classifier.

Fuses the CNN spatial embedding and LSTM temporal embedding through
concatenation, then classifies via a 2-layer MLP head.

Architecture:
  CNN(128-dim) ─┐
                ├── Concat → 256-dim
  LSTM(128-dim)─┘
  Linear(256, 64) → ReLU → Dropout
  Linear(64, 2)   → Softmax → [REAL, AI_CLONE]
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
import config
from src.models.cnn_branch import CNNBranch
from src.models.lstm_branch import LSTMBranch


class EnsembleAntiSpoof(nn.Module):
    """
    Ensemble anti-spoofing model combining CNN and LSTM branches.

    The CNN branch processes mel spectrograms to detect spatial frequency
    patterns (vocoder fingerprints). The LSTM branch processes MFCC + phase
    derivative sequences to detect temporal irregularities (phase jumps,
    prosody anomalies). Both embeddings are fused and classified.
    """

    def __init__(
        self,
        cnn_embedding_dim: int = config.CNN_EMBEDDING_DIM,
        lstm_embedding_dim: int = config.LSTM_EMBEDDING_DIM,
        hidden_dim: int = config.ENSEMBLE_HIDDEN_DIM,
        num_classes: int = config.NUM_CLASSES,
        dropout: float = 0.3,
    ):
        super().__init__()

        self.cnn_branch = CNNBranch(embedding_dim=cnn_embedding_dim)
        self.lstm_branch = LSTMBranch(embedding_dim=lstm_embedding_dim)

        fusion_dim = cnn_embedding_dim + lstm_embedding_dim

        self.classifier = nn.Sequential(
            nn.Linear(fusion_dim, hidden_dim),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, mel_spectrogram: torch.Tensor,
                mfcc_phase: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through both branches and fusion classifier.

        Args:
            mel_spectrogram: (batch, 1, n_mels, T) for CNN branch
            mfcc_phase: (batch, n_mfcc+1, T) for LSTM branch

        Returns:
            Logits tensor of shape (batch, num_classes)
        """
        cnn_emb = self.cnn_branch(mel_spectrogram)   # (B, 128)
        lstm_emb = self.lstm_branch(mfcc_phase)       # (B, 128)

        # Fusion: concatenate embeddings
        fused = torch.cat([cnn_emb, lstm_emb], dim=1)  # (B, 256)

        # Classification
        logits = self.classifier(fused)  # (B, 2)
        return logits

    def predict_proba(self, mel_spectrogram: torch.Tensor,
                      mfcc_phase: torch.Tensor) -> torch.Tensor:
        """Get softmax probabilities for each class."""
        logits = self.forward(mel_spectrogram, mfcc_phase)
        return F.softmax(logits, dim=1)

    def count_parameters(self) -> int:
        """Count total trainable parameters."""
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
