"""
Model Evaluation Module.

Computes classification metrics on the test set:
- Accuracy, Precision, Recall, F1-Score (per-class and macro)
- Confusion Matrix
- Equal Error Rate (EER)
"""

import numpy as np
import torch
from torch.utils.data import DataLoader
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report
)
from typing import Dict, Tuple
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent.parent.resolve()))
import config
from src.models.ensemble import EnsembleAntiSpoof


def compute_eer(y_true: np.ndarray, y_scores: np.ndarray) -> float:
    """
    Compute Equal Error Rate (EER).

    The EER is the point where the False Acceptance Rate (FAR) equals
    the False Rejection Rate (FRR). Lower EER = better performance.
    """
    from sklearn.metrics import roc_curve

    fpr, tpr, thresholds = roc_curve(y_true, y_scores, pos_label=1)
    fnr = 1 - tpr

    # Find the threshold where FAR ≈ FRR
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2
    return float(eer)


@torch.no_grad()
def evaluate_model(
    model: EnsembleAntiSpoof,
    test_loader: DataLoader,
    device: torch.device = config.DEVICE,
) -> Dict:
    """
    Evaluate the model on the test set.

    Returns:
        Dictionary with all metrics
    """
    model.eval()
    model.to(device)

    all_preds = []
    all_labels = []
    all_probs = []

    for features, labels in test_loader:
        mel_spec = features["mel_spectrogram"].to(device)
        mfcc_phase = features["mfcc_phase"].to(device)

        logits = model(mel_spec, mfcc_phase)
        probs = torch.softmax(logits, dim=1)

        _, predicted = torch.max(logits, 1)
        all_preds.extend(predicted.cpu().numpy())
        all_labels.extend(labels.numpy())
        all_probs.extend(probs[:, 1].cpu().numpy())  # P(AI_CLONE)

    all_preds = np.array(all_preds)
    all_labels = np.array(all_labels)
    all_probs = np.array(all_probs)

    # Core metrics
    accuracy = accuracy_score(all_labels, all_preds)
    precision = precision_score(all_labels, all_preds, average='macro',
                                zero_division=0)
    recall = recall_score(all_labels, all_preds, average='macro',
                          zero_division=0)
    f1 = f1_score(all_labels, all_preds, average='macro', zero_division=0)

    # Confusion matrix
    cm = confusion_matrix(all_labels, all_preds)

    # EER
    eer = compute_eer(all_labels, all_probs)

    # Classification report
    report = classification_report(
        all_labels, all_preds,
        target_names=config.CLASS_NAMES,
        zero_division=0,
    )

    results = {
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1_score": f1,
        "eer": eer,
        "confusion_matrix": cm,
        "classification_report": report,
        "predictions": all_preds,
        "true_labels": all_labels,
        "probabilities": all_probs,
    }

    # Print results
    print(f"\n{'='*60}")
    print(f"TEST SET EVALUATION")
    print(f"{'='*60}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"EER:       {eer:.4f}")
    print(f"\nConfusion Matrix:")
    print(f"  {'':>12} Pred REAL  Pred CLONE")
    print(f"  True REAL   {cm[0][0]:>8}  {cm[0][1]:>10}")
    print(f"  True CLONE  {cm[1][0]:>8}  {cm[1][1]:>10}")
    print(f"\n{report}")
    print(f"{'='*60}\n")

    return results
