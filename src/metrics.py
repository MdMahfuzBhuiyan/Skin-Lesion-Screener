"""Metrics and threshold calibration for imbalanced binary classification."""

from __future__ import annotations

import numpy as np
from sklearn.metrics import balanced_accuracy_score, f1_score


def predict_with_threshold(probs_malignant: np.ndarray, threshold: float) -> np.ndarray:
    return (probs_malignant >= threshold).astype(int)


def find_best_threshold(
    labels: list[int] | np.ndarray,
    probs_malignant: list[float] | np.ndarray,
) -> tuple[float, float]:
    """
    Pick threshold that maximizes the worse of benign/malignant recall.
    Avoids always predicting the majority class (common with imbalanced lesions).
    """
    labels = np.asarray(labels)
    probs = np.asarray(probs_malignant)
    best_t, best_score = 0.5, -1.0
    for t in np.linspace(0.35, 0.75, 41):
        preds = predict_with_threshold(probs, t)
        benign_mask = labels == 0
        mal_mask = labels == 1
        rec_benign = (preds[benign_mask] == 0).mean() if benign_mask.any() else 0.0
        rec_mal = (preds[mal_mask] == 1).mean() if mal_mask.any() else 0.0
        score = min(rec_benign, rec_mal)
        if score > best_score:
            best_score, best_t = score, float(t)
    return best_t, best_score


def binary_metrics(
    labels: list[int],
    preds: list[int],
    probs_malignant: list[float] | None = None,
) -> dict:
    ba = balanced_accuracy_score(labels, preds)
    f1 = f1_score(labels, preds, average="binary")
    return {"balanced_accuracy": ba, "f1": f1}
