"""
PyTorch Dataset and DataLoader factory for audio anti-spoofing.

Provides:
  - AntiSpoofDataset: loads audio files, extracts features on-the-fly
  - create_dataloaders: creates stratified train/val/test splits with
    class-balanced sampling
"""

import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from sklearn.model_selection import train_test_split
from pathlib import Path
from typing import Tuple, Dict, List
import sys

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
import config
from src.feature_extraction.extractor import FeatureExtractor
from src.data_engineering.augmentation import AudioAugmentor


class AntiSpoofDataset(Dataset):
    """
    PyTorch Dataset for audio anti-spoofing.

    Loads audio files from real/ and spoof/ directories,
    extracts features using FeatureExtractor, and applies
    optional augmentation during training.
    """

    def __init__(
        self,
        file_paths: List[str],
        labels: List[int],
        augment: bool = False,
    ):
        """
        Args:
            file_paths: List of absolute paths to audio files.
            labels: List of labels (0=REAL, 1=AI_CLONE).
            augment: Whether to apply data augmentation.
        """
        self.file_paths = file_paths
        self.labels = labels
        self.augment = augment
        self.extractor = FeatureExtractor()
        self.augmentor = AudioAugmentor(p=0.4) if augment else None

    def __len__(self) -> int:
        return len(self.file_paths)

    def __getitem__(self, idx: int) -> Tuple[Dict[str, torch.Tensor], int]:
        filepath = self.file_paths[idx]
        label = self.labels[idx]

        # Load and preprocess audio
        waveform = self.extractor.load_audio(filepath)

        # Apply waveform augmentation during training
        if self.augmentor is not None:
            waveform = self.augmentor.augment_waveform(waveform)

        # Extract features
        features = self.extractor.extract_all_features(waveform)

        # Apply spectrogram augmentation during training
        if self.augmentor is not None:
            features["mel_spectrogram"] = self.augmentor.augment_spectrogram(
                features["mel_spectrogram"].squeeze(0)
            ).unsqueeze(0)

        return features, label


def _collect_file_paths_and_labels() -> Tuple[List[str], List[int]]:
    """Scan data directories and collect all audio files with labels."""
    file_paths = []
    labels = []

    # Real samples (label = 0)
    real_dir = config.REAL_AUDIO_DIR
    if real_dir.exists():
        for f in sorted(real_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in config.ALLOWED_AUDIO_EXTENSIONS:
                file_paths.append(str(f))
                labels.append(0)

    # Spoof samples (label = 1)
    spoof_dir = config.SPOOF_AUDIO_DIR
    if spoof_dir.exists():
        for f in sorted(spoof_dir.iterdir()):
            if f.is_file() and f.suffix.lower() in config.ALLOWED_AUDIO_EXTENSIONS:
                file_paths.append(str(f))
                labels.append(1)

    return file_paths, labels


def create_dataloaders(
    batch_size: int = config.BATCH_SIZE,
    train_split: float = config.TRAIN_SPLIT,
    val_split: float = config.VAL_SPLIT,
    seed: int = config.RANDOM_SEED,
) -> Tuple[DataLoader, DataLoader, DataLoader]:
    """
    Create train, validation, and test DataLoaders with stratified splits
    and class-balanced sampling.

    Returns:
        (train_loader, val_loader, test_loader)
    """
    file_paths, labels = _collect_file_paths_and_labels()

    if len(file_paths) == 0:
        raise FileNotFoundError(
            f"No audio files found in {config.REAL_AUDIO_DIR} or "
            f"{config.SPOOF_AUDIO_DIR}. Run dataset generation first."
        )

    print(f"[DATASET] Found {len(file_paths)} total samples "
          f"(Real: {labels.count(0)}, Spoof: {labels.count(1)})")

    # Stratified train/temp split
    train_paths, temp_paths, train_labels, temp_labels = train_test_split(
        file_paths, labels,
        test_size=(1 - train_split),
        stratify=labels,
        random_state=seed,
    )

    # Split temp into val and test
    relative_val = val_split / (val_split + (1 - train_split - val_split))
    val_paths, test_paths, val_labels, test_labels = train_test_split(
        temp_paths, temp_labels,
        test_size=(1 - relative_val),
        stratify=temp_labels,
        random_state=seed,
    )

    print(f"[DATASET] Splits — Train: {len(train_paths)}, "
          f"Val: {len(val_paths)}, Test: {len(test_paths)}")

    # Create datasets
    train_dataset = AntiSpoofDataset(train_paths, train_labels, augment=True)
    val_dataset = AntiSpoofDataset(val_paths, val_labels, augment=False)
    test_dataset = AntiSpoofDataset(test_paths, test_labels, augment=False)

    # Class-balanced sampler for training
    train_label_array = np.array(train_labels)
    class_counts = np.bincount(train_label_array)
    class_weights = 1.0 / class_counts
    sample_weights = class_weights[train_label_array]
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(train_label_array),
        replacement=True,
    )

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        sampler=sampler,
        num_workers=0,  # Keep 0 for Windows compatibility
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=0,
        pin_memory=True,
    )

    return train_loader, val_loader, test_loader
