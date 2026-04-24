from __future__ import annotations

import numpy as np
from sklearn.metrics import accuracy_score, log_loss, roc_auc_score


def _align_proba(y_proba: np.ndarray, n_classes: int) -> np.ndarray:
    """Right-pads a probability matrix with zero columns so it has exactly
    `n_classes` columns. Needed when a training fold happens to miss a class
    and the model therefore outputs a narrower probability matrix."""
    if y_proba.ndim == 1:
        y_proba = np.stack([1.0 - y_proba, y_proba], axis=1)
    if y_proba.shape[1] < n_classes:
        pad = np.zeros((y_proba.shape[0], n_classes - y_proba.shape[1]), dtype=y_proba.dtype)
        y_proba = np.concatenate([y_proba, pad], axis=1)
    return y_proba


def compute_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_proba: np.ndarray,
    classes: np.ndarray,
    *,
    auc_only: bool = False,
) -> dict[str, float]:
    """Computes ROC AUC (OvR macro for multiclass), accuracy, and log-loss
    on a single fold. `classes` must list all label indices the task has."""
    n_classes = len(classes)
    y_proba = _align_proba(np.asarray(y_proba), n_classes)

    if n_classes == 2:
        auc = roc_auc_score(y_true, y_proba[:, 1])
    else:
        auc = roc_auc_score(
            y_true, y_proba, multi_class="ovr", average="macro", labels=classes
        )

    metrics: dict[str, float] = {"roc_auc": float(auc)}
    if auc_only:
        return metrics

    metrics["accuracy"] = float(accuracy_score(y_true, y_pred))
    metrics["log_loss"] = float(log_loss(y_true, y_proba, labels=classes))
    return metrics
