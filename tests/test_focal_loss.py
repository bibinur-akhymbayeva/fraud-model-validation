import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from fraudval.focal_loss import focal_grad, focal_hess, focal_init_score, focal_loss, sigmoid  # noqa: E402
from fraudval.metrics import threshold_at_fpr  # noqa: E402
from fraudval.stability import psi  # noqa: E402

rng = np.random.default_rng(0)
X = rng.normal(0, 3, 2000)
Y = rng.integers(0, 2, 2000)


def test_gradient_matches_finite_difference():
    for alpha, gamma in [(0.25, 2.0), (0.5, 1.0), (0.75, 0.5)]:
        h = 1e-6
        num = (focal_loss(X + h, Y, alpha, gamma) - focal_loss(X - h, Y, alpha, gamma)) / (2 * h)
        assert np.allclose(focal_grad(X, Y, alpha, gamma), num, atol=1e-5)


def test_reduces_to_half_log_loss_gradient():
    # gamma = 0, alpha = 0.5: gradient is 0.5 * (p - y)
    assert np.allclose(focal_grad(X, Y, 0.5, 0.0), 0.5 * (sigmoid(X) - Y), atol=1e-10)


def test_hessian_matches_log_loss_when_gamma_zero():
    p = sigmoid(X)
    assert np.allclose(focal_hess(X, Y, 0.5, 0.0), 0.5 * p * (1 - p), atol=1e-6)


def test_init_score_is_base_rate_log_odds_for_log_loss():
    y = (rng.random(10_000) < 0.02).astype(int)
    assert abs(focal_init_score(y, 0.5, 0.0) - np.log(y.mean() / (1 - y.mean()))) < 1e-3


def test_threshold_respects_target_fpr():
    s = rng.random(100_000)
    y = (rng.random(100_000) < 0.01).astype(int)
    thr = threshold_at_fpr(y, s, 0.001)
    assert np.mean(s[y == 0] > thr) <= 0.001


def test_psi_is_zero_for_identical_samples_and_grows_with_shift():
    a = rng.normal(size=50_000)
    assert psi(a, a) < 1e-9
    assert psi(a, a + 0.5) > psi(a, a + 0.1) > 0
