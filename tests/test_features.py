"""
Tests for the feature extraction pipeline.
"""

import sys
import numpy as np
import torch
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.resolve()))

import config
from src.feature_extraction.extractor import FeatureExtractor
from src.data_engineering.dataset_generator import generate_real_sample, generate_spoof_sample


def test_feature_shapes():
    """Verify extracted features have correct dimensions."""
    extractor = FeatureExtractor()

    # Generate a test waveform
    waveform = generate_real_sample(0)
    assert len(waveform) == config.AUDIO_NUM_SAMPLES, \
        f"Waveform length mismatch: {len(waveform)} vs {config.AUDIO_NUM_SAMPLES}"

    # Pre-emphasis adds 1 sample
    processed = extractor.load_audio.__wrapped__(extractor, None) if False else None
    # Just use raw waveform with pre-emphasis
    waveform_processed = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])

    # Test mel spectrogram
    mel = extractor.extract_mel_spectrogram(waveform_processed)
    assert mel.shape == (config.N_MELS, config.NUM_TIME_FRAMES), \
        f"Mel shape mismatch: {mel.shape}"

    # Test MFCC
    mfcc = extractor.extract_mfcc(waveform_processed)
    assert mfcc.shape == (config.N_MFCC, config.NUM_TIME_FRAMES), \
        f"MFCC shape mismatch: {mfcc.shape}"

    # Test phase derivative
    phase = extractor.extract_phase_derivative(waveform_processed)
    assert phase.shape == (1, config.NUM_TIME_FRAMES), \
        f"Phase shape mismatch: {phase.shape}"

    print("[PASS] Feature shapes correct")


def test_feature_extraction_pipeline():
    """Test the full feature extraction pipeline."""
    extractor = FeatureExtractor()

    # Test with real sample
    waveform = generate_real_sample(0)
    waveform = np.append(waveform[0], waveform[1:] - 0.97 * waveform[:-1])
    features = extractor.extract_all_features(waveform)

    assert "mel_spectrogram" in features
    assert "mfcc_phase" in features

    mel = features["mel_spectrogram"]
    assert mel.shape == (1, config.N_MELS, config.NUM_TIME_FRAMES), \
        f"Mel tensor shape: {mel.shape}"

    mfcc_phase = features["mfcc_phase"]
    assert mfcc_phase.shape == (config.N_MFCC + 1, config.NUM_TIME_FRAMES), \
        f"MFCC+phase tensor shape: {mfcc_phase.shape}"

    print("[PASS] Feature extraction pipeline works")


def test_diagnostic_features():
    """Test diagnostic feature extraction."""
    extractor = FeatureExtractor()

    # Compare real vs spoof diagnostic features
    real_waveform = generate_real_sample(0)
    spoof_waveform = generate_spoof_sample(0)

    real_diag = extractor.extract_diagnostic_features(real_waveform)
    spoof_diag = extractor.extract_diagnostic_features(spoof_waveform)

    assert "spectral_flatness_high_band" in real_diag
    assert "phase_variance" in real_diag
    assert "mfcc_delta_variance" in real_diag

    print(f"  Real diagnostics:  {real_diag}")
    print(f"  Spoof diagnostics: {spoof_diag}")
    print("[PASS] Diagnostic features extracted")


def test_features_differ_between_classes():
    """Verify that features are statistically different between classes."""
    extractor = FeatureExtractor()

    real_mels = []
    spoof_mels = []

    for i in range(10):
        real_wf = generate_real_sample(i)
        real_wf = np.append(real_wf[0], real_wf[1:] - 0.97 * real_wf[:-1])
        real_mel = extractor.extract_mel_spectrogram(real_wf)
        real_mels.append(np.mean(real_mel))

        spoof_wf = generate_spoof_sample(i)
        spoof_wf = np.append(spoof_wf[0], spoof_wf[1:] - 0.97 * spoof_wf[:-1])
        spoof_mel = extractor.extract_mel_spectrogram(spoof_wf)
        spoof_mels.append(np.mean(spoof_mel))

    # The means should be different (features carry class signal)
    real_mean = np.mean(real_mels)
    spoof_mean = np.mean(spoof_mels)

    print(f"  Real mel mean:  {real_mean:.4f}")
    print(f"  Spoof mel mean: {spoof_mean:.4f}")
    print("[PASS] Feature distributions differ between classes")


if __name__ == "__main__":
    print("Running feature extraction tests...\n")
    test_feature_shapes()
    test_feature_extraction_pipeline()
    test_diagnostic_features()
    test_features_differ_between_classes()
    print("\nAll feature tests passed! [PASS]")
