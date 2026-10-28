"""Performance and calibration measures suited to rare-event models."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import average_precision_score, brier_score_loss, roc_auc_score


def pr_auc(y, s) -> float:
    return float(average_precision_score(y, s))


def roc_auc(y, s) -> float:
    return float(roc_auc_score(y, s))


def threshold_at_fpr(y, s, target_fpr: float) -> float:
    """Lowest threshold whose false-positive rate on (y, s) does not exceed target_fpr."""
    neg = np.sort(s[y == 0])
    k = int(np.floor(target_fpr * len(neg)))
    return float(neg[-k - 1]) if k < len(neg) else float(neg[0])


def at_threshold(y, s, thr: float) -> dict:
    pred = s > thr
    tp = int(np.sum(pred & (y == 1)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    return {
        "recall": tp / max(tp + fn, 1),
        "precision": tp / max(tp + fp, 1),
        "fpr": fp / max(fp + tn, 1),
        "alerts": tp + fp, "tp": tp, "fp": fp, "fn": fn,
    }


def bootstrap_ci(y, s, fn=pr_auc, n: int = 500, level: float = 0.95, seed: int = 42) -> tuple[float, float]:
    """Percentile bootstrap interval, resampling positives and negatives separately."""
    rng = np.random.default_rng(seed)
    pos, neg = np.flatnonzero(y == 1), np.flatnonzero(y == 0)
    stats = []
    for _ in range(n):
        idx = np.concatenate([rng.choice(pos, len(pos)), rng.choice(neg, len(neg))])
        stats.append(fn(y[idx], s[idx]))
    lo, hi = np.quantile(stats, [(1 - level) / 2, 1 - (1 - level) / 2])
    return float(lo), float(hi)


def brier(y, p) -> float:
    return float(brier_score_loss(y, p))


def expected_calibration_error(y, p, bins: int = 10) -> float:
    """ECE over equal-frequency bins of predicted probability."""
    order = np.argsort(p)
    chunks = np.array_split(order, bins)
    return float(sum(len(c) / len(p) * abs(p[c].mean() - y[c].mean()) for c in chunks if len(c)))


def calibration_in_the_large(y, p) -> float:
    """Mean predicted probability divided by the observed event rate (1.0 is perfect)."""
    return float(p.mean() / max(y.mean(), 1e-12))
