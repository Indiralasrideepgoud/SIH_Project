"""
CNN Branch — Spatial Spectrogram Pattern Detector.

A 3-layer CNN operating on mel spectrograms to detect spatial patterns
characteristic of vocoder synthesis: spectral banding, unnatural formant
structure, and frequency-domain artifacts invisible to the human ear.

Architecture:
  Conv2d(1,32) → BN → ReLU → MaxPool
  Conv2d(32,64) → BN → ReLU → MaxPool
  Conv2d(64,128) → BN → ReLU → AdaptiveAvgPool
  Dropout → Flatten → 128-dim embedding
"""

import torch
import torch.nn as nn
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
import config


class CNNBranch(nn.Module):
    """CNN branch for spatial spectrogram analysis."""

    def __init__(
        self,
        in_channels: int = 1,
        embedding_dim: int = config.CNN_EMBEDDING_DIM,
        dropout: float = config.CNN_DROPOUT,
    ):
        super().__init__()

        self.features = nn.Sequential(
            # Block 1: (1, 128, T) → (32, 64, T//2)
            nn.Conv2d(in_channels, 32, kernel_size=3, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 2: (32, 64, T//2) → (64, 32, T//4)
            nn.Conv2d(32, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),

            # Block 3: (64, 32, T//4) → (128, 1, 1) via adaptive pool
            nn.Conv2d(64, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d((1, 1)),
        )

        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(128, embedding_dim)

    def forward(self, mel_spectrogram: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            mel_spectrogram: Tensor of shape (batch, 1, n_mels, T)

        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        x = self.features(mel_spectrogram)  # (B, 128, 1, 1)
        x = x.view(x.size(0), -1)           # (B, 128)
        x = self.dropout(x)
        x = self.fc(x)                       # (B, embedding_dim)
        return x
