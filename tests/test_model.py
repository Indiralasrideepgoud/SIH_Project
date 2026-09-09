"""
Tests for model architecture (shape validation and forward pass).
"""

import sys
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import config
from src.models.cnn_branch import CNNBranch
from src.models.lstm_branch import LSTMBranch
from src.models.ensemble import EnsembleAntiSpoof


def test_cnn_branch_shapes():
    """Verify CNN branch input/output shapes."""
    model = CNNBranch()
    batch_size = 4

    # Input: (B, 1, n_mels, T)
    x = torch.randn(batch_size, 1, config.N_MELS, config.NUM_TIME_FRAMES)
    out = model(x)

    assert out.shape == (batch_size, config.CNN_EMBEDDING_DIM), \
        f"CNN output shape: {out.shape}"
    print(f"[PASS] CNN branch: input {x.shape} -> output {out.shape}")


def test_lstm_branch_shapes():
    """Verify LSTM branch input/output shapes."""
    model = LSTMBranch()
    batch_size = 4

    # Input: (B, features, T) where features = n_mfcc + 1
    x = torch.randn(batch_size, config.LSTM_INPUT_DIM, config.NUM_TIME_FRAMES)
    out = model(x)

    assert out.shape == (batch_size, config.LSTM_EMBEDDING_DIM), \
        f"LSTM output shape: {out.shape}"
    print(f"[PASS] LSTM branch: input {x.shape} -> output {out.shape}")


def test_ensemble_forward_pass():
    """Verify ensemble model forward pass and output shapes."""
    model = EnsembleAntiSpoof()
    batch_size = 4

    mel = torch.randn(batch_size, 1, config.N_MELS, config.NUM_TIME_FRAMES)
    mfcc = torch.randn(batch_size, config.LSTM_INPUT_DIM, config.NUM_TIME_FRAMES)

    logits = model(mel, mfcc)
    assert logits.shape == (batch_size, config.NUM_CLASSES), \
        f"Ensemble logits shape: {logits.shape}"

    probs = model.predict_proba(mel, mfcc)
    assert probs.shape == (batch_size, config.NUM_CLASSES), \
        f"Ensemble probs shape: {probs.shape}"

    # Probabilities should sum to 1
    prob_sums = probs.sum(dim=1)
    assert torch.allclose(prob_sums, torch.ones(batch_size), atol=1e-5), \
        f"Probabilities don't sum to 1: {prob_sums}"

    print(f"[PASS] Ensemble: logits {logits.shape}, probs {probs.shape}")
    print(f"  Parameters: {model.count_parameters():,}")


def test_model_gradient_flow():
    """Verify gradients flow through the entire model."""
    model = EnsembleAntiSpoof()
    model.train()

    mel = torch.randn(2, 1, config.N_MELS, config.NUM_TIME_FRAMES)
    mfcc = torch.randn(2, config.LSTM_INPUT_DIM, config.NUM_TIME_FRAMES)
    labels = torch.tensor([0, 1])

    logits = model(mel, mfcc)
    loss = torch.nn.CrossEntropyLoss()(logits, labels)
    loss.backward()

    # Check all parameters have gradients
    for name, param in model.named_parameters():
        if param.requires_grad:
            assert param.grad is not None, f"No gradient for {name}"
            assert not torch.all(param.grad == 0), f"Zero gradient for {name}"

    print("[PASS] Gradients flow through all model parameters")


if __name__ == "__main__":
    print("Running model architecture tests...\n")
    test_cnn_branch_shapes()
    test_lstm_branch_shapes()
    test_ensemble_forward_pass()
    test_model_gradient_flow()
    print("\nAll model tests passed! [PASS]")
