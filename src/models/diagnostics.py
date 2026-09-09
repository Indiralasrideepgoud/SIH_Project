"""
Post-Inference Diagnostic Analyzer.

Analyzes extracted audio features to generate human-readable diagnostic
flags explaining why a sample was classified as REAL or AI_CLONE.

Flags are based on acoustic feature statistics:
- Spectral flatness in high-frequency band (vocoder reconstruction noise)
- Phase instantaneous frequency variance (phase discontinuities)
- MFCC delta variance (temporal smoothness anomalies)
- Confidence borderline detection
"""

import sys
from pathlib import Path
from typing import List

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
import config


class DiagnosticAnalyzer:
    """Generates diagnostic flags from audio features and model confidence."""

    def __init__(self):
        self.vocoder_threshold = config.VOCODER_ARTIFACT_THRESHOLD
        self.phase_threshold = config.PHASE_DISCONTINUITY_THRESHOLD
        self.smoothness_threshold = config.TEMPORAL_SMOOTHNESS_THRESHOLD
        self.borderline_low = config.BORDERLINE_CONFIDENCE_LOW
        self.borderline_high = config.BORDERLINE_CONFIDENCE_HIGH

    def analyze(self, diagnostic_features: dict,
                confidence: float,
                classification: str) -> List[str]:
        """
        Generate diagnostic flags based on feature statistics.

        Args:
            diagnostic_features: dict from FeatureExtractor.extract_diagnostic_features()
                Keys: spectral_flatness_high_band, phase_variance, mfcc_delta_variance
            confidence: Model confidence score (0.0 to 1.0)
            classification: "REAL" or "AI_CLONE"

        Returns:
            List of diagnostic flag strings
        """
        flags = []

        sf_high = diagnostic_features.get("spectral_flatness_high_band", 0.0)
        phase_var = diagnostic_features.get("phase_variance", 0.0)
        mfcc_delta_var = diagnostic_features.get("mfcc_delta_variance", 0.0)

        # Vocoder artifact detection
        if sf_high > self.vocoder_threshold:
            flags.append(
                f"High vocoder artifact probability "
                f"(spectral flatness: {sf_high:.3f})"
            )

        # Phase discontinuity detection
        if phase_var > self.phase_threshold:
            flags.append(
                f"Phase discontinuity detected "
                f"(IF variance: {phase_var:.4f})"
            )

        # Temporal smoothness anomaly
        if mfcc_delta_var < self.smoothness_threshold:
            flags.append(
                f"Unnatural temporal smoothness "
                f"(MFCC-Δ variance: {mfcc_delta_var:.4f})"
            )

        # Borderline confidence warning
        if self.borderline_low <= confidence <= self.borderline_high:
            flags.append(
                f"Low confidence — borderline sample "
                f"(confidence: {confidence:.3f})"
            )

        # High confidence confirmation
        if confidence > 0.9 and classification == "AI_CLONE":
            flags.append("Strong synthetic signature detected")
        elif confidence > 0.9 and classification == "REAL":
            flags.append("Strong natural speech signature confirmed")

        # If no specific flags, provide a generic summary
        if not flags:
            flags.append(
                f"Classification: {classification} "
                f"(confidence: {confidence:.3f})"
            )

        return flags
