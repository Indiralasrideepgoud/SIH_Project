"""
Audio augmentation transforms for anti-spoofing training robustness.

Applies data augmentation to improve model generalization:
- Additive Gaussian noise
- Time masking (simulates dropout)
- Frequency masking (SpecAugment-style)
- Gain variation
"""

import numpy as np
import torch


class AudioAugmentor:
    """Applies a chain of audio augmentations with configurable probability."""

    def __init__(self, p: float = 0.5, seed: int = None):
        """
        Args:
            p: Probability of applying each augmentation.
            seed: Random seed for reproducibility.
        """
        self.p = p
        self.rng = np.random.RandomState(seed)

    def add_gaussian_noise(self, waveform: np.ndarray,
                           snr_db: float = None) -> np.ndarray:
        """Add Gaussian noise at a random SNR between 15-40 dB."""
        if self.rng.random() > self.p:
            return waveform
        if snr_db is None:
            snr_db = self.rng.uniform(15, 40)
        signal_power = np.mean(waveform ** 2) + 1e-10
        noise_power = signal_power / (10 ** (snr_db / 10))
        noise = self.rng.randn(len(waveform)) * np.sqrt(noise_power)
        return (waveform + noise).astype(np.float32)

    def random_gain(self, waveform: np.ndarray,
                    min_gain: float = 0.6,
                    max_gain: float = 1.4) -> np.ndarray:
        """Apply random gain scaling."""
        if self.rng.random() > self.p:
            return waveform
        gain = self.rng.uniform(min_gain, max_gain)
        return (waveform * gain).astype(np.float32)

    def time_mask(self, spectrogram: torch.Tensor,
                  max_mask_pct: float = 0.15) -> torch.Tensor:
        """
        Apply time masking to a spectrogram (SpecAugment-style).
        Zeros out a contiguous block of time frames.

        Args:
            spectrogram: Tensor of shape (freq_bins, time_frames).
            max_mask_pct: Maximum percentage of time frames to mask.
        """
        if self.rng.random() > self.p:
            return spectrogram
        _, t = spectrogram.shape
        mask_len = int(t * self.rng.uniform(0, max_mask_pct))
        if mask_len == 0:
            return spectrogram
        start = self.rng.randint(0, max(t - mask_len, 1))
        spectrogram = spectrogram.clone()
        spectrogram[:, start:start + mask_len] = 0
        return spectrogram

    def frequency_mask(self, spectrogram: torch.Tensor,
                       max_mask_pct: float = 0.15) -> torch.Tensor:
        """
        Apply frequency masking to a spectrogram (SpecAugment-style).
        Zeros out a contiguous block of frequency bins.

        Args:
            spectrogram: Tensor of shape (freq_bins, time_frames).
            max_mask_pct: Maximum percentage of freq bins to mask.
        """
        if self.rng.random() > self.p:
            return spectrogram
        f, _ = spectrogram.shape
        mask_len = int(f * self.rng.uniform(0, max_mask_pct))
        if mask_len == 0:
            return spectrogram
        start = self.rng.randint(0, max(f - mask_len, 1))
        spectrogram = spectrogram.clone()
        spectrogram[start:start + mask_len, :] = 0
        return spectrogram

    def augment_waveform(self, waveform: np.ndarray) -> np.ndarray:
        """Apply all waveform-level augmentations."""
        waveform = self.add_gaussian_noise(waveform)
        waveform = self.random_gain(waveform)
        return waveform

    def augment_spectrogram(self, spectrogram: torch.Tensor) -> torch.Tensor:
        """Apply all spectrogram-level augmentations."""
        spectrogram = self.time_mask(spectrogram)
        spectrogram = self.frequency_mask(spectrogram)
        return spectrogram
