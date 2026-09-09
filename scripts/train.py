"""
CLI script to train the ensemble anti-spoofing model.

Usage:
    python scripts/train.py [--epochs N] [--batch-size N] [--lr F]
"""

import sys
import argparse
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import config
from src.data_engineering.dataset import create_dataloaders
from src.models.ensemble import EnsembleAntiSpoof
from src.training.trainer import Trainer
from src.training.evaluate import evaluate_model


def main():
    parser = argparse.ArgumentParser(
        description="Train the ensemble anti-spoofing model"
    )
    parser.add_argument("--epochs", type=int, default=config.NUM_EPOCHS)
    parser.add_argument("--batch-size", type=int, default=config.BATCH_SIZE)
    parser.add_argument("--lr", type=float, default=config.LEARNING_RATE)
    args = parser.parse_args()

    print("=" * 60)
    print("AUDIO ANTI-SPOOFING — MODEL TRAINING")
    print("=" * 60)

    # 1. Create data loaders
    print("\n[1/4] Loading dataset...")
    train_loader, val_loader, test_loader = create_dataloaders(
        batch_size=args.batch_size
    )

    # 2. Initialize model
    print("\n[2/4] Initializing model...")
    model = EnsembleAntiSpoof()
    print(f"  Architecture: CNN + Bi-LSTM Ensemble")
    print(f"  Parameters: {model.count_parameters():,}")
    print(f"  Device: {config.DEVICE}")

    # 3. Train
    print("\n[3/4] Training...")
    trainer = Trainer(
        model=model,
        train_loader=train_loader,
        val_loader=val_loader,
        lr=args.lr,
        num_epochs=args.epochs,
    )
    history = trainer.train()

    # 4. Evaluate on test set
    print("\n[4/4] Evaluating on test set...")
    # Reload best model for evaluation
    import torch
    checkpoint = torch.load(
        str(config.BEST_MODEL_PATH),
        map_location=config.DEVICE,
        weights_only=False,
    )
    model.load_state_dict(checkpoint["model_state_dict"])

    results = evaluate_model(model, test_loader)

    print("\nTraining pipeline complete!")
    print(f"Best model saved to: {config.BEST_MODEL_PATH}")
    print(f"Test accuracy: {results['accuracy']:.4f}")
    print(f"Test F1-Score: {results['f1_score']:.4f}")
    print(f"Test EER: {results['eer']:.4f}")


if __name__ == "__main__":
    main()
