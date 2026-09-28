from __future__ import annotations

import numpy as np
import torch
from sklearn.metrics import accuracy_score, roc_auc_score

from damascus.proof.assay import bootstrap_ci, delong_auc_test, paired_permutation
from damascus.proof.edge_tests import classification_metrics, normalised_confusion


def test_metrics_match_sklearn() -> None:
    rng = np.random.default_rng(0)
    logits = torch.tensor(rng.normal(size=(40, 2)), dtype=torch.float32)
    targets = torch.tensor(rng.integers(0, 2, size=40))
    metrics = classification_metrics(logits, targets)
    probs = torch.softmax(logits, dim=-1).numpy()
    assert abs(metrics["accuracy"] - accuracy_score(targets.numpy(), probs.argmax(1))) < 1e-9
    assert abs(metrics["auroc"] - roc_auc_score(targets.numpy(), probs[:, 1])) < 1e-9


def test_confusion_rows_normalised() -> None:
    rng = np.random.default_rng(1)
    logits = torch.tensor(rng.normal(size=(50, 5)), dtype=torch.float32)
    targets = torch.tensor(rng.integers(0, 5, size=50))
    matrix = normalised_confusion(logits, targets, 5)
    rows = matrix.sum(axis=1)
    assert np.allclose(rows[rows > 0], 1.0)


def test_delong_matches_auc_point_estimate() -> None:
    rng = np.random.default_rng(2)
    targets = rng.integers(0, 2, size=200)
    score_a = targets * 0.6 + rng.normal(scale=0.5, size=200)
    score_b = targets * 0.3 + rng.normal(scale=0.5, size=200)
    result = delong_auc_test(targets.astype(float), score_a, score_b)
    assert abs(result["auc_a"] - roc_auc_score(targets, score_a)) < 1e-6
    assert abs(result["auc_b"] - roc_auc_score(targets, score_b)) < 1e-6
    assert 0.0 <= result["p_value"] <= 1.0


def test_bootstrap_and_permutation_bounds() -> None:
    rng = np.random.default_rng(3)
    targets = rng.integers(0, 2, size=120)
    scores = targets * 0.7 + rng.normal(scale=0.4, size=120)
    point, lower, upper = bootstrap_ci(targets, scores, n_resamples=200, seed=0)
    assert lower <= point <= upper
    perm = paired_permutation(scores, scores * 0.5, n_permutations=500, seed=0)
    assert 0.0 <= perm["p_value"] <= 1.0
