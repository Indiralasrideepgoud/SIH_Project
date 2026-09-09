"""
Training Loop for the Ensemble Anti-Spoofing Model.

Features:
- Adam optimizer with weight decay
- Cross-entropy loss with class weights
- Early stopping on validation loss
- Best model checkpointing
- Per-epoch logging of loss, accuracy, and learning rate
"""

import time
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from pathlib import Path
from typing import Dict, Optional
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
import config
from src.models.ensemble import EnsembleAntiSpoof


class Trainer:
    """Training manager for the ensemble anti-spoofing model."""

    def __init__(
        self,
        model: EnsembleAntiSpoof,
        train_loader: DataLoader,
        val_loader: DataLoader,
        lr: float = config.LEARNING_RATE,
        weight_decay: float = config.WEIGHT_DECAY,
        num_epochs: int = config.NUM_EPOCHS,
        patience: int = config.EARLY_STOPPING_PATIENCE,
        device: torch.device = config.DEVICE,
        save_path: Path = config.BEST_MODEL_PATH,
    ):
        self.model = model.to(device)
        self.train_loader = train_loader
        self.val_loader = val_loader
        self.num_epochs = num_epochs
        self.patience = patience
        self.device = device
        self.save_path = save_path

        # Optimizer
        self.optimizer = torch.optim.Adam(
            model.parameters(), lr=lr, weight_decay=weight_decay
        )

        # Learning rate scheduler
        self.scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.optimizer, mode='min', factor=0.5, patience=3
        )

        # Class-weighted cross-entropy loss
        if hasattr(train_loader.dataset, 'labels'):
            label_array = np.array(train_loader.dataset.labels)
        else:
            label_array = np.array([label for _, label in train_loader.dataset])
        class_counts = np.bincount(label_array, minlength=config.NUM_CLASSES)
        if 0 in class_counts:
            class_counts = np.ones(config.NUM_CLASSES)
        class_weights = 1.0 / class_counts
        class_weights = class_weights / class_weights.sum() * config.NUM_CLASSES
        weight_tensor = torch.FloatTensor(class_weights).to(device)
        self.criterion = nn.CrossEntropyLoss(weight=weight_tensor)

        # Training history
        self.history = {
            "train_loss": [], "train_acc": [],
            "val_loss": [], "val_acc": [],
        }

    def _train_epoch(self) -> Dict[str, float]:
        """Run one training epoch."""
        self.model.train()
        running_loss = 0.0
        correct = 0
        total = 0

        for batch_idx, (features, labels) in enumerate(self.train_loader):
            mel_spec = features["mel_spectrogram"].to(self.device)
            mfcc_phase = features["mfcc_phase"].to(self.device)
            labels = labels.to(self.device)

            # Forward pass
            self.optimizer.zero_grad()
            logits = self.model(mel_spec, mfcc_phase)
            loss = self.criterion(logits, labels)

            # Backward pass
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            # Metrics
            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return {"loss": epoch_loss, "accuracy": epoch_acc}

    @torch.no_grad()
    def _validate_epoch(self) -> Dict[str, float]:
        """Run one validation epoch."""
        self.model.eval()
        running_loss = 0.0
        correct = 0
        total = 0

        for features, labels in self.val_loader:
            mel_spec = features["mel_spectrogram"].to(self.device)
            mfcc_phase = features["mfcc_phase"].to(self.device)
            labels = labels.to(self.device)

            logits = self.model(mel_spec, mfcc_phase)
            loss = self.criterion(logits, labels)

            running_loss += loss.item() * labels.size(0)
            _, predicted = torch.max(logits, 1)
            correct += (predicted == labels).sum().item()
            total += labels.size(0)

        epoch_loss = running_loss / total
        epoch_acc = correct / total
        return {"loss": epoch_loss, "accuracy": epoch_acc}

    def train(self) -> Dict:
        """
        Full training loop with early stopping.

        Returns:
            Training history dict
        """
        print(f"\n{'='*60}")
        print(f"TRAINING — Ensemble Anti-Spoofing Model")
        print(f"{'='*60}")
        print(f"Device: {self.device}")
        print(f"Parameters: {self.model.count_parameters():,}")
        print(f"Epochs: {self.num_epochs} (patience: {self.patience})")
        print(f"{'='*60}\n")

        best_val_loss = float("inf")
        patience_counter = 0
        start_time = time.time()

        for epoch in range(1, self.num_epochs + 1):
            epoch_start = time.time()

            # Train
            train_metrics = self._train_epoch()
            self.history["train_loss"].append(train_metrics["loss"])
            self.history["train_acc"].append(train_metrics["accuracy"])

            # Validate
            val_metrics = self._validate_epoch()
            self.history["val_loss"].append(val_metrics["loss"])
            self.history["val_acc"].append(val_metrics["accuracy"])

            # Learning rate scheduling
            self.scheduler.step(val_metrics["loss"])

            epoch_time = time.time() - epoch_start
            lr = self.optimizer.param_groups[0]["lr"]

            print(
                f"Epoch {epoch:3d}/{self.num_epochs} | "
                f"Train Loss: {train_metrics['loss']:.4f} Acc: {train_metrics['accuracy']:.4f} | "
                f"Val Loss: {val_metrics['loss']:.4f} Acc: {val_metrics['accuracy']:.4f} | "
                f"LR: {lr:.6f} | {epoch_time:.1f}s"
            )

            # Early stopping check
            if val_metrics["loss"] < best_val_loss:
                best_val_loss = val_metrics["loss"]
                patience_counter = 0
                # Save best model
                self._save_checkpoint(epoch, val_metrics)
                print(f"  [+] Best model saved (val_loss: {best_val_loss:.4f})")
            else:
                patience_counter += 1
                if patience_counter >= self.patience:
                    print(f"\n  ✗ Early stopping at epoch {epoch} "
                          f"(no improvement for {self.patience} epochs)")
                    break

        total_time = time.time() - start_time
        print(f"\n{'='*60}")
        print(f"Training complete in {total_time:.1f}s")
        print(f"Best validation loss: {best_val_loss:.4f}")
        print(f"Model saved to: {self.save_path}")
        print(f"{'='*60}\n")

        return self.history

    def _save_checkpoint(self, epoch: int, metrics: Dict) -> None:
        """Save model checkpoint."""
        self.save_path.parent.mkdir(parents=True, exist_ok=True)
        checkpoint = {
            "epoch": epoch,
            "model_state_dict": self.model.state_dict(),
            "optimizer_state_dict": self.optimizer.state_dict(),
            "val_loss": metrics["loss"],
            "val_accuracy": metrics["accuracy"],
            "config": {
                "cnn_embedding_dim": config.CNN_EMBEDDING_DIM,
                "lstm_embedding_dim": config.LSTM_EMBEDDING_DIM,
                "ensemble_hidden_dim": config.ENSEMBLE_HIDDEN_DIM,
                "num_classes": config.NUM_CLASSES,
                "n_mels": config.N_MELS,
                "n_mfcc": config.N_MFCC,
                "sample_rate": config.SAMPLE_RATE,
            },
        }
        torch.save(checkpoint, str(self.save_path))
