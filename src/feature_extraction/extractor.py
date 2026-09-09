"""
Multi-Stream Feature Extraction Pipeline.

Extracts three parallel feature streams from raw audio:
  1. Mel Spectrogram (128 × T) — spatial spectral patterns, vocoder fingerprints
  2. MFCCs (40 × T) — timbral envelope, unnatural smoothness detection
  3. Phase Derivative (1 × T) — instantaneous frequency deviation, phase coherence

All features are normalized (z-score) and padded/truncated to fixed length.
Includes high-frequency emphasis to amplify vocoder artifacts above 6kHz.
"""

import numpy as np
import librosa
import torch
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
import config


class FeatureExtractor:
    """Extracts mel spectrogram, MFCC, and phase features from audio."""

    def __init__(
        self,
        sr: int = config.SAMPLE_RATE,
        n_fft: int = config.N_FFT,
        hop_length: int = config.HOP_LENGTH,
        n_mels: int = config.N_MELS,
        n_mfcc: int = config.N_MFCC,
        target_length: int = config.NUM_TIME_FRAMES,
    ):
        self.sr = sr
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.n_mfcc = n_mfcc
        self.target_length = target_length

    def load_audio(self, filepath: str, duration: float = config.AUDIO_DURATION_SEC) -> np.ndarray:
        """
        Load and preprocess audio file.

        - Resamples to target sample rate
        - Converts to mono
        - Pads or truncates to fixed duration
        - Applies pre-emphasis filter for high-frequency boost
        """
        waveform, sr = librosa.load(
            filepath, sr=self.sr, mono=True, duration=duration
        )

        # Pad or truncate to exact length
        target_samples = int(self.sr * duration)
        if len(waveform) < target_samples:
            waveform = np.pad(waveform, (0, target_samples - len(waveform)))
        else:
            waveform = waveform[:target_samples]

        # Pre-emphasis filter: boost high frequencies where vocoder artifacts live
        waveform = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])

        return waveform.astype(np.float32)

    def load_audio_from_bytes(self, audio_bytes: bytes,
                              duration: float = config.AUDIO_DURATION_SEC) -> np.ndarray:
        """Load audio from in-memory bytes (for API inference)."""
        import io
        import soundfile as sf

        # Read audio from bytes
        data, sr = sf.read(io.BytesIO(audio_bytes))

        # Convert to mono if stereo
        if data.ndim > 1:
            data = np.mean(data, axis=1)

        # Resample if needed
        if sr != self.sr:
            data = librosa.resample(data, orig_sr=sr, target_sr=self.sr)

        # Pad or truncate
        target_samples = int(self.sr * duration)
        if len(data) < target_samples:
            data = np.pad(data, (0, target_samples - len(data)))
        else:
            data = data[:target_samples]

        # Pre-emphasis
        data = np.append(data[0], data[1:] - 0.97 * data[:-1])

        return data.astype(np.float32)

    def extract_mel_spectrogram(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract log-mel spectrogram.

        Returns:
            np.ndarray of shape (n_mels, target_length)
        """
        mel_spec = librosa.feature.melspectrogram(
            y=waveform,
            sr=self.sr,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=config.FMIN,
            fmax=config.FMAX,
        )
        # Convert to log scale (dB)
        log_mel = librosa.power_to_db(mel_spec, ref=np.max)

        # Pad or truncate time axis
        log_mel = self._fix_length(log_mel)

        # Z-score normalization per frequency bin
        mean = np.mean(log_mel, axis=1, keepdims=True)
        std = np.std(log_mel, axis=1, keepdims=True) + 1e-8
        log_mel = (log_mel - mean) / std

        return log_mel.astype(np.float32)

    def extract_mfcc(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract MFCCs with delta coefficients.

        Returns:
            np.ndarray of shape (n_mfcc, target_length)
        """
        mfcc = librosa.feature.mfcc(
            y=waveform,
            sr=self.sr,
            n_mfcc=self.n_mfcc,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Pad or truncate time axis
        mfcc = self._fix_length(mfcc)

        # Z-score normalization
        mean = np.mean(mfcc, axis=1, keepdims=True)
        std = np.std(mfcc, axis=1, keepdims=True) + 1e-8
        mfcc = (mfcc - mean) / std

        return mfcc.astype(np.float32)

    def extract_phase_derivative(self, waveform: np.ndarray) -> np.ndarray:
        """
        Extract instantaneous frequency deviation (group delay derivative).

        Phase coherence violations are a strong indicator of neural vocoder
        synthesis — real speech maintains smooth phase evolution while
        vocoders introduce frame-boundary discontinuities.

        Returns:
            np.ndarray of shape (1, target_length)
        """
        # Compute STFT
        stft = librosa.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
        )

        # Extract phase and compute instantaneous frequency
        phase = np.angle(stft)

        # Phase difference across time (instantaneous frequency)
        inst_freq = np.diff(phase, axis=1)
        # Wrap to [-pi, pi]
        inst_freq = np.angle(np.exp(1j * inst_freq))

        # Compute deviation: variance of instantaneous frequency across freq bins
        # High variance = phase discontinuity (vocoder artifact)
        phase_dev = np.var(inst_freq, axis=0, keepdims=False)

        # Reshape to (1, T)
        phase_dev = phase_dev.reshape(1, -1)

        # Pad or truncate
        phase_dev = self._fix_length(phase_dev)

        # Normalize
        mean = np.mean(phase_dev)
        std = np.std(phase_dev) + 1e-8
        phase_dev = (phase_dev - mean) / std

        return phase_dev.astype(np.float32)

    def extract_all_features(self, waveform: np.ndarray) -> dict:
        """
        Extract all three feature streams from a waveform.

        Returns:
            dict with keys:
              'mel_spectrogram': torch.Tensor (1, n_mels, T)
              'mfcc_phase': torch.Tensor (n_mfcc+1, T) — for LSTM input
        """
        mel_spec = self.extract_mel_spectrogram(waveform)
        mfcc = self.extract_mfcc(waveform)
        phase_dev = self.extract_phase_derivative(waveform)

        # CNN input: mel spectrogram as single-channel "image"
        mel_tensor = torch.FloatTensor(mel_spec).unsqueeze(0)  # (1, n_mels, T)

        # LSTM input: concatenate MFCC and phase derivative along feature axis
        mfcc_phase = np.concatenate([mfcc, phase_dev], axis=0)  # (n_mfcc+1, T)
        mfcc_phase_tensor = torch.FloatTensor(mfcc_phase)  # (n_mfcc+1, T)

        return {
            "mel_spectrogram": mel_tensor,
            "mfcc_phase": mfcc_phase_tensor,
        }

    def extract_diagnostic_features(self, waveform: np.ndarray) -> dict:
        """
        Extract additional features for post-inference diagnostic analysis.

        Returns statistics used by the diagnostics module to generate flags.
        """
        # Spectral flatness in high-frequency band (6-8kHz)
        stft = np.abs(librosa.stft(waveform, n_fft=self.n_fft,
                                    hop_length=self.hop_length))
        freqs = librosa.fft_frequencies(sr=self.sr, n_fft=self.n_fft)
        high_band_mask = (freqs >= 6000) & (freqs <= 8000)
        high_band = stft[high_band_mask, :]
        if high_band.size > 0:
            spectral_flatness_high = float(np.exp(np.mean(np.log(high_band + 1e-10)))
                                           / (np.mean(high_band) + 1e-10))
        else:
            spectral_flatness_high = 0.0

        # Phase discontinuity metric
        phase = np.angle(librosa.stft(waveform, n_fft=self.n_fft,
                                       hop_length=self.hop_length))
        inst_freq = np.diff(phase, axis=1)
        inst_freq_wrapped = np.angle(np.exp(1j * inst_freq))
        phase_var = float(np.mean(np.var(inst_freq_wrapped, axis=0)))

        # MFCC delta variance (temporal smoothness)
        mfcc = librosa.feature.mfcc(y=waveform, sr=self.sr, n_mfcc=self.n_mfcc)
        mfcc_delta = librosa.feature.delta(mfcc)
        mfcc_delta_var = float(np.mean(np.var(mfcc_delta, axis=1)))

        return {
            "spectral_flatness_high_band": spectral_flatness_high,
            "phase_variance": phase_var,
            "mfcc_delta_variance": mfcc_delta_var,
        }

    def _fix_length(self, feature: np.ndarray) -> np.ndarray:
        """Pad or truncate the time axis (axis=1) to target_length."""
        current_length = feature.shape[1]
        if current_length < self.target_length:
            pad_width = self.target_length - current_length
            feature = np.pad(feature, ((0, 0), (0, pad_width)), mode='constant')
        elif current_length > self.target_length:
            feature = feature[:, :self.target_length]
        return feature
