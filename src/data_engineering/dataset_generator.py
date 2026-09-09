"""
Synthetic Audio Dataset Generator for Anti-Spoofing MVP.

Generates two classes of audio samples:
  - REAL: Simulates natural human speech with harmonic structure, formant
    transitions, jitter/shimmer (vocal cord irregularity), and realistic
    amplitude envelopes.
  - SPOOF (AI_CLONE): Simulates neural TTS/voice conversion output with
    vocoder artifacts — spectral banding, phase discontinuities, metallic
    harmonics, and unnaturally smooth temporal envelopes.

These synthetic signals are acoustically distinct in the feature domains
(mel spectrogram, MFCC, phase) that the model operates on, enabling
meaningful training even without real speech data.
"""

import numpy as np
import soundfile as sf
from pathlib import Path
import sys

# Add project root to path for config import
sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
import config


def _generate_natural_envelope(num_samples: int, sr: int) -> np.ndarray:
    """Generate a natural amplitude envelope mimicking speech prosody."""
    t = np.linspace(0, num_samples / sr, num_samples)
    # Slow amplitude modulation (2-5 Hz) — natural speech rhythm
    envelope = 0.5 + 0.3 * np.sin(2 * np.pi * 3.2 * t)
    envelope *= 0.6 + 0.4 * np.sin(2 * np.pi * 1.1 * t)
    # Soft attack/release
    attack = np.minimum(t / 0.05, 1.0)
    release = np.minimum((num_samples / sr - t) / 0.05, 1.0)
    envelope *= attack * release
    return np.clip(envelope, 0, 1)


def _add_jitter_shimmer(signal: np.ndarray, jitter: float = 0.01,
                        shimmer: float = 0.03) -> np.ndarray:
    """Add micro-perturbations simulating vocal cord irregularity."""
    # Jitter: slight random time-domain perturbation
    noise = np.random.randn(len(signal)) * jitter
    jittered = signal + noise * signal
    # Shimmer: slight random amplitude modulation
    amp_mod = 1.0 + shimmer * np.random.randn(len(signal))
    return jittered * amp_mod


def generate_real_sample(sample_idx: int, sr: int = config.SAMPLE_RATE,
                         duration: float = config.AUDIO_DURATION_SEC) -> np.ndarray:
    """
    Generate a sample simulating natural human speech characteristics.

    Features present in real speech:
    - Rich harmonic structure with natural formant transitions
    - Micro-perturbations (jitter/shimmer) from vocal cord vibration
    - Natural amplitude envelope with prosody modulation
    - Slight breathiness and noise floor
    """
    rng = np.random.RandomState(config.RANDOM_SEED + sample_idx)
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Fundamental frequency with natural variation (80-300 Hz range)
    f0_base = rng.uniform(100, 250)
    f0_vibrato = f0_base + 3.0 * np.sin(2 * np.pi * 5.5 * t)  # 5.5 Hz vibrato
    phase = 2 * np.pi * np.cumsum(f0_vibrato) / sr

    # Harmonics with natural roll-off
    signal = np.zeros(num_samples)
    num_harmonics = rng.randint(8, 15)
    for h in range(1, num_harmonics + 1):
        amplitude = 1.0 / (h ** 1.2)  # Natural harmonic decay
        amplitude *= rng.uniform(0.7, 1.3)  # Per-harmonic variation
        signal += amplitude * np.sin(h * phase)

    # Formant-like resonances (broadband emphasis around formant freqs)
    formant_freqs = [rng.uniform(300, 800), rng.uniform(1000, 2500),
                     rng.uniform(2200, 3500)]
    for ff in formant_freqs:
        formant_signal = 0.15 * np.sin(2 * np.pi * ff * t +
                                        rng.uniform(0, 2 * np.pi))
        formant_env = np.exp(-0.5 * ((t * sr % (sr / 3)) / (sr / 8)) ** 2)
        signal += formant_signal * formant_env

    # Add jitter and shimmer (vocal cord irregularity)
    signal = _add_jitter_shimmer(signal, jitter=0.008, shimmer=0.025)

    # Natural envelope
    envelope = _generate_natural_envelope(num_samples, sr)
    signal *= envelope

    # Breathy noise floor (natural)
    noise_level = rng.uniform(0.005, 0.02)
    signal += noise_level * rng.randn(num_samples)

    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.85
    return signal.astype(np.float32)


def generate_spoof_sample(sample_idx: int, sr: int = config.SAMPLE_RATE,
                          duration: float = config.AUDIO_DURATION_SEC) -> np.ndarray:
    """
    Generate a sample simulating neural TTS / voice conversion artifacts.

    Artifacts injected (characteristic of neural vocoders):
    - Spectral banding at regular intervals (vocoder frequency grid)
    - Phase discontinuities between frames (frame-boundary artifacts)
    - Unnaturally smooth temporal envelope (no jitter/shimmer)
    - Metallic harmonics with rigid frequency spacing
    - Subtle high-frequency buzzing (vocoder aliasing)
    """
    rng = np.random.RandomState(config.RANDOM_SEED + sample_idx + 10000)
    num_samples = int(sr * duration)
    t = np.linspace(0, duration, num_samples, endpoint=False)

    # Fundamental — unnaturally stable (no vibrato, no micro-variation)
    f0 = rng.uniform(120, 220)
    phase = 2 * np.pi * f0 * t

    # Harmonics — rigid, evenly spaced (no natural variation)
    signal = np.zeros(num_samples)
    num_harmonics = rng.randint(10, 20)
    for h in range(1, num_harmonics + 1):
        amplitude = 1.0 / (h ** 0.8)  # Slower decay = metallic quality
        signal += amplitude * np.sin(h * phase)

    # === VOCODER ARTIFACT 1: Spectral banding ===
    # Regular frequency-domain artifacts from vocoder frame processing
    banding_freq = rng.uniform(50, 150)  # Artifact frequency
    banding_signal = 0.12 * np.sin(2 * np.pi * banding_freq * t)
    for k in range(2, 6):
        banding_signal += (0.06 / k) * np.sin(2 * np.pi * k * banding_freq * t)
    signal += banding_signal

    # === VOCODER ARTIFACT 2: Phase discontinuities ===
    # Simulate frame-boundary phase jumps (typical of overlap-add synthesis)
    frame_size = int(sr * 0.025)  # 25ms frames
    num_frames = num_samples // frame_size
    for i in range(num_frames):
        start = i * frame_size
        end = min(start + frame_size, num_samples)
        if rng.random() < 0.3:  # 30% of frames have phase jump
            phase_jump = rng.uniform(-np.pi / 4, np.pi / 4)
            jump_signal = 0.04 * np.sin(2 * np.pi * f0 * t[start:end] + phase_jump)
            signal[start:end] += jump_signal

    # === VOCODER ARTIFACT 3: Unnaturally smooth envelope ===
    # No jitter/shimmer — perfectly smooth amplitude
    smooth_env = 0.7 + 0.2 * np.sin(2 * np.pi * 2.0 * t)  # Very regular
    signal *= smooth_env

    # === VOCODER ARTIFACT 4: High-frequency buzzing ===
    # Aliasing artifacts above 6kHz
    buzz_freq = rng.uniform(6000, 7500)
    buzz = 0.03 * np.sin(2 * np.pi * buzz_freq * t)
    buzz *= (0.5 + 0.5 * np.sin(2 * np.pi * 8 * t))  # Modulated buzzing
    signal += buzz

    # === VOCODER ARTIFACT 5: Unnatural spectral flatness in high band ===
    # Add broadband noise specifically in 5-8kHz (vocoder reconstruction noise)
    noise = rng.randn(num_samples)
    # Simple high-pass approximation via differencing
    hp_noise = np.diff(np.diff(noise))
    hp_noise = np.pad(hp_noise, (1, 1), mode='edge')
    signal += 0.02 * hp_noise

    # Normalize
    signal = signal / (np.max(np.abs(signal)) + 1e-8) * 0.85
    return signal.astype(np.float32)


def generate_dataset(num_per_class: int = config.NUM_SAMPLES_PER_CLASS,
                     real_dir: Path = config.REAL_AUDIO_DIR,
                     spoof_dir: Path = config.SPOOF_AUDIO_DIR) -> dict:
    """
    Generate the complete synthetic dataset.

    Returns:
        dict with 'real_count', 'spoof_count', and directory paths.
    """
    real_dir.mkdir(parents=True, exist_ok=True)
    spoof_dir.mkdir(parents=True, exist_ok=True)

    print(f"[DATA] Generating {num_per_class} REAL samples...")
    for i in range(num_per_class):
        waveform = generate_real_sample(i)
        filepath = real_dir / f"real_{i:04d}.wav"
        sf.write(str(filepath), waveform, config.SAMPLE_RATE)
        if (i + 1) % 50 == 0:
            print(f"  REAL: {i + 1}/{num_per_class}")

    print(f"[DATA] Generating {num_per_class} SPOOF samples...")
    for i in range(num_per_class):
        waveform = generate_spoof_sample(i)
        filepath = spoof_dir / f"spoof_{i:04d}.wav"
        sf.write(str(filepath), waveform, config.SAMPLE_RATE)
        if (i + 1) % 50 == 0:
            print(f"  SPOOF: {i + 1}/{num_per_class}")

    print(f"[DATA] Dataset generation complete!")
    print(f"  Real samples:  {num_per_class} -> {real_dir}")
    print(f"  Spoof samples: {num_per_class} -> {spoof_dir}")

    return {
        "real_count": num_per_class,
        "spoof_count": num_per_class,
        "real_dir": str(real_dir),
        "spoof_dir": str(spoof_dir),
    }


if __name__ == "__main__":
    generate_dataset()
