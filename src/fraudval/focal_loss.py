"""Focal Loss as a custom LightGBM objective.

Focal Loss (Lin et al., 2017) down-weights well-classified examples so that
training concentrates on the hard ones. For a binary label y and raw score x
(log-odds), with p = sigmoid(x):

    L(x, y=1) = -alpha       * (1 - p)^gamma * log(p)
    L(x, y=0) = -(1 - alpha) *       p^gamma * log(1 - p)

gamma = 0 and alpha = 0.5 give half the usual log loss.

The gradient with respect to the raw score is analytic. The Hessian is taken by
a central difference of the analytic gradient, then floored at a small positive
value because LightGBM needs positive second derivatives.
"""
from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar

_EPS = 1e-12


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def focal_loss(x: np.ndarray, y: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """Per-observation focal loss at raw score x."""
    p = np.clip(sigmoid(x), _EPS, 1 - _EPS)
    pt = np.where(y == 1, p, 1 - p)
    at = np.where(y == 1, alpha, 1 - alpha)
    return -at * (1 - pt) ** gamma * np.log(pt)


def focal_grad(x: np.ndarray, y: np.ndarray, alpha: float, gamma: float) -> np.ndarray:
    """dL/dx, derived analytically (see module docstring)."""
    p = np.clip(sigmoid(x), _EPS, 1 - _EPS)
    g_pos = alpha * (1 - p) ** gamma * (gamma * p * np.log(p) - (1 - p))
    g_neg = (1 - alpha) * p ** gamma * (p - gamma * (1 - p) * np.log(1 - p))
    return np.where(y == 1, g_pos, g_neg)


def focal_hess(x: np.ndarray, y: np.ndarray, alpha: float, gamma: float, h: float = 1e-4) -> np.ndarray:
    """d2L/dx2 by central difference of the analytic gradient."""
    return (focal_grad(x + h, y, alpha, gamma) - focal_grad(x - h, y, alpha, gamma)) / (2 * h)


def lgb_focal_objective(alpha: float, gamma: float, hess_floor: float = 1e-6):
    """Return a LightGBM custom objective (preds are raw scores)."""

    def objective(preds: np.ndarray, data) -> tuple[np.ndarray, np.ndarray]:
        y = data.get_label()
        grad = focal_grad(preds, y, alpha, gamma)
        hess = np.maximum(focal_hess(preds, y, alpha, gamma), hess_floor)
        return grad, hess

    return objective


def focal_init_score(y: np.ndarray, alpha: float, gamma: float) -> float:
    """Constant raw score that minimises mean focal loss (the boosting start point)."""
    res = minimize_scalar(lambda c: focal_loss(np.full(len(y), c), y, alpha, gamma).mean(),
                          bounds=(-15, 15), method="bounded")
    return float(res.x)
