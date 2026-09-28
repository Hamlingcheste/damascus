from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    cohen_kappa_score,
    confusion_matrix,
    f1_score,
    roc_auc_score,
)
from torch import nn


def to_probabilities(logits: torch.Tensor) -> np.ndarray:
    return torch.softmax(logits, dim=-1).detach().cpu().numpy()


def _auroc(probs: np.ndarray, targets: np.ndarray) -> float:
    num_classes = probs.shape[1]
    present = np.unique(targets)
    if num_classes == 2:
        if present.size < 2:
            return float("nan")
        return float(roc_auc_score(targets, probs[:, 1]))
    if present.size < 2:
        return float("nan")
    return float(
        roc_auc_score(targets, probs, multi_class="ovr", average="macro", labels=range(num_classes))
    )


def _sensitivity_specificity(preds: np.ndarray, targets: np.ndarray) -> tuple[float, float]:
    matrix = confusion_matrix(targets, preds, labels=[0, 1])
    tn, fp, fn, tp = matrix.ravel()
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return float(sensitivity), float(specificity)


def classification_metrics(logits: torch.Tensor, targets: torch.Tensor) -> dict[str, float]:
    probs = to_probabilities(logits)
    target_np = targets.detach().cpu().numpy().astype(int)
    preds = probs.argmax(axis=1)
    num_classes = probs.shape[1]
    metrics: dict[str, float] = {
        "accuracy": float(accuracy_score(target_np, preds)),
        "auroc": _auroc(probs, target_np),
        "macro_f1": float(f1_score(target_np, preds, average="macro", zero_division=0)),
    }
    if num_classes > 2:
        metrics["quadratic_kappa"] = float(
            cohen_kappa_score(target_np, preds, weights="quadratic", labels=range(num_classes))
        )
    if num_classes == 2:
        sensitivity, specificity = _sensitivity_specificity(preds, target_np)
        metrics["sensitivity"] = sensitivity
        metrics["specificity"] = specificity
        metrics["f1"] = float(f1_score(target_np, preds, zero_division=0))
    return metrics


def normalised_confusion(
    logits: torch.Tensor, targets: torch.Tensor, num_classes: int
) -> np.ndarray:
    preds = to_probabilities(logits).argmax(axis=1)
    target_np = targets.detach().cpu().numpy().astype(int)
    matrix = confusion_matrix(target_np, preds, labels=range(num_classes)).astype(float)
    row_sums = matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    return matrix / row_sums


def count_parameters(model: nn.Module) -> dict[str, float]:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {"total_millions": total / 1e6, "trainable_millions": trainable / 1e6}
