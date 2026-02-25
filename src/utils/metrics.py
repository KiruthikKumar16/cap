from typing import Dict, Optional, Tuple

import numpy as np
from sklearn import metrics


def compute_threshold(
    scores: np.ndarray,
    method: str = "quantile",
    quantile: float = 0.98,
    labels: Optional[np.ndarray] = None,
) -> float:
    """
    Compute an anomaly threshold.

    Args:
        scores: Anomaly scores (higher = more anomalous).
        method: "quantile", "roc", or "f1".
        quantile: Quantile value for quantile method.
        labels: Optional ground-truth labels (required for roc/f1).
    """
    if method == "quantile":
        return float(np.quantile(scores, quantile))
    if labels is None:
        raise ValueError(f"labels are required for threshold method: {method}")
    labels = labels.astype(int)
    if method == "roc":
        fpr, tpr, thr = metrics.roc_curve(labels, scores)
        j = tpr - fpr
        return float(thr[int(np.argmax(j))])
    if method == "f1":
        precision, recall, thr = metrics.precision_recall_curve(labels, scores)
        f1 = 2 * precision * recall / (precision + recall + 1e-9)
        # precision_recall_curve returns thresholds of length n-1
        if len(thr) == 0:
            return float(np.quantile(scores, quantile))
        return float(thr[int(np.argmax(f1[:-1]))])
    raise ValueError(f"Unsupported threshold method: {method}")


def evaluate_anomalies(
    scores: np.ndarray,
    labels: np.ndarray,
    threshold: float,
) -> Dict[str, float]:
    preds = (scores >= threshold).astype(int)
    precision = metrics.precision_score(labels, preds, zero_division=0)
    recall = metrics.recall_score(labels, preds, zero_division=0)
    f1 = metrics.f1_score(labels, preds, zero_division=0)
    roc_auc = metrics.roc_auc_score(labels, scores) if len(np.unique(labels)) > 1 else 0.0
    tn, fp, fn, tp = metrics.confusion_matrix(labels, preds, labels=[0, 1]).ravel()
    far = fp / (fp + tn) if (fp + tn) > 0 else 0.0
    return {"precision": precision, "recall": recall, "f1": f1, "roc_auc": roc_auc, "false_alarm_rate": far}


def detection_lead_time(preds: np.ndarray, labels: np.ndarray) -> Optional[float]:
    """
    Compute average lead time (timesteps) between first positive label
    and first predicted positive. Returns None if no positives.
    """
    label_idxs = np.where(labels == 1)[0]
    pred_idxs = np.where(preds == 1)[0]
    if len(label_idxs) == 0 or len(pred_idxs) == 0:
        return None
    return float(label_idxs[0] - pred_idxs[0])


def smooth_scores(scores: np.ndarray, window: int = 3) -> np.ndarray:
    if window <= 1:
        return scores
    kernel = np.ones(window) / window
    return np.convolve(scores, kernel, mode="same")

