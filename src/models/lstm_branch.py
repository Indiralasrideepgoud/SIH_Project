"""
LSTM Branch — Temporal Sequence Pattern Detector.

A bidirectional LSTM with self-attention operating on MFCC + phase derivative
sequences to capture temporal irregularities: phase jumps between vocoder
frames, unnatural prosody transitions, and temporal smoothness anomalies.

Architecture:
  Bi-LSTM(input=41, hidden=64, layers=2, dropout=0.3)
  Self-Attention (learnable query over hidden states)
  128-dim embedding output
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
import config


class SelfAttention(nn.Module):
    """Learnable self-attention over LSTM hidden states."""

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.attention = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output: torch.Tensor) -> torch.Tensor:
        """
        Args:
            lstm_output: (batch, seq_len, hidden_dim)

        Returns:
            Weighted context vector: (batch, hidden_dim)
        """
        # Compute attention weights
        attn_weights = self.attention(lstm_output)    # (B, T, 1)
        attn_weights = F.softmax(attn_weights, dim=1) # (B, T, 1)

        # Weighted sum of hidden states
        context = torch.sum(lstm_output * attn_weights, dim=1)  # (B, hidden_dim)
        return context


class LSTMBranch(nn.Module):
    """Bi-LSTM branch with self-attention for temporal pattern analysis."""

    def __init__(
        self,
        input_dim: int = config.LSTM_INPUT_DIM,
        hidden_dim: int = config.LSTM_HIDDEN_DIM,
        num_layers: int = config.LSTM_NUM_LAYERS,
        dropout: float = config.LSTM_DROPOUT,
        embedding_dim: int = config.LSTM_EMBEDDING_DIM,
    ):
        super().__init__()

        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if num_layers > 1 else 0.0,
        )

        # Bi-directional doubles hidden dim
        self.attention = SelfAttention(hidden_dim * 2)
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * 2, embedding_dim)

    def forward(self, mfcc_phase: torch.Tensor) -> torch.Tensor:
        """
        Forward pass.

        Args:
            mfcc_phase: Tensor of shape (batch, features, T)
                        where features = n_mfcc + 1 (phase derivative)

        Returns:
            Embedding tensor of shape (batch, embedding_dim)
        """
        # Transpose to (batch, T, features) for LSTM
        x = mfcc_phase.transpose(1, 2)  # (B, T, 41)

        # LSTM forward
        lstm_out, _ = self.lstm(x)  # (B, T, hidden*2)

        # Self-attention aggregation
        context = self.attention(lstm_out)  # (B, hidden*2)

        # Project to embedding
        context = self.dropout(context)
        embedding = self.fc(context)  # (B, embedding_dim)
        return embedding
