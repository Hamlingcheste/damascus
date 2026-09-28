from __future__ import annotations

import numpy as np
from scipy import stats
from sklearn.metrics import roc_auc_score


def bootstrap_ci(
    targets: np.ndarray,
    scores: np.ndarray,
    n_resamples: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> tuple[float, float, float]:
    rng = np.random.default_rng(seed)
    size = targets.shape[0]
    point = float(roc_auc_score(targets, scores))
    samples = []
    for _ in range(n_resamples):
        idx = rng.integers(0, size, size)
        if np.unique(targets[idx]).size < 2:
            continue
        samples.append(roc_auc_score(targets[idx], scores[idx]))
    if not samples:
        return point, float("nan"), float("nan")
    lower = float(np.quantile(samples, alpha / 2))
    upper = float(np.quantile(samples, 1 - alpha / 2))
    return point, lower, upper


def paired_permutation(
    metric_a: np.ndarray,
    metric_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 0,
) -> dict[str, float]:
    rng = np.random.default_rng(seed)
    diff = metric_a - metric_b
    observed = float(diff.mean())
    count = 0
    for _ in range(n_permutations):
        signs = rng.integers(0, 2, diff.shape[0]) * 2 - 1
        permuted = float((diff * signs).mean())
        if abs(permuted) >= abs(observed):
            count += 1
    p_value = (count + 1) / (n_permutations + 1)
    return {"observed": observed, "p_value": float(p_value)}


def _midrank(x: np.ndarray) -> np.ndarray:
    order = np.argsort(x)
    sorted_x = x[order]
    ranks = np.zeros(x.shape[0], dtype=float)
    i = 0
    n = x.shape[0]
    while i < n:
        j = i
        while j < n and sorted_x[j] == sorted_x[i]:
            j += 1
        ranks[i:j] = 0.5 * (i + j - 1) + 1
        i = j
    out = np.empty(n, dtype=float)
    out[order] = ranks
    return out


def _delong_components(scores: np.ndarray, positives: int) -> tuple[np.ndarray, np.ndarray, float]:
    pos = scores[:positives]
    neg = scores[positives:]
    m = pos.shape[0]
    n = neg.shape[0]
    tx = _midrank(pos)
    ty = _midrank(neg)
    tz = _midrank(scores)
    auc = (tz[:m].sum() - m * (m + 1) / 2) / (m * n)
    v01 = (tz[:m] - tx) / n
    v10 = 1.0 - (tz[m:] - ty) / m
    return v01, v10, float(auc)


def delong_auc_test(
    targets: np.ndarray, scores_a: np.ndarray, scores_b: np.ndarray
) -> dict[str, float]:
    order = np.argsort(-targets)
    targets = targets[order]
    scores_a = scores_a[order]
    scores_b = scores_b[order]
    positives = int(targets.sum())
    m = positives
    n = targets.shape[0] - positives
    v01_a, v10_a, auc_a = _delong_components(scores_a, positives)
    v01_b, v10_b, auc_b = _delong_components(scores_b, positives)
    sx = np.cov(np.stack([v01_a, v01_b]))
    sy = np.cov(np.stack([v10_a, v10_b]))
    var = sx / m + sy / n
    diff = auc_a - auc_b
    se = float(np.sqrt(var[0, 0] + var[1, 1] - 2 * var[0, 1]))
    if se == 0:
        return {"auc_a": auc_a, "auc_b": auc_b, "z": float("nan"), "p_value": float("nan")}
    z = diff / se
    p_value = float(2 * stats.norm.sf(abs(z)))
    return {"auc_a": auc_a, "auc_b": auc_b, "z": float(z), "p_value": p_value}
