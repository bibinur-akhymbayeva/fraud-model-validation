"""Four ways of handling class imbalance with the same LightGBM learner."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import product

import lightgbm as lgb
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score

from .data import Split
from .focal_loss import focal_init_score, lgb_focal_objective, sigmoid

SEED = 42
BASE_PARAMS = dict(
    objective="binary", learning_rate=0.05, num_leaves=31, min_child_samples=50,
    feature_fraction=0.8, bagging_fraction=0.8, bagging_freq=1, lambda_l2=1.0,
    metric="average_precision", verbose=-1, seed=SEED, num_threads=0,
)


@dataclass
class FittedModel:
    name: str
    booster: lgb.Booster
    init: float = 0.0
    notes: dict = field(default_factory=dict)
    platt: LogisticRegression | None = None

    def raw(self, X) -> np.ndarray:
        return self.booster.predict(X, raw_score=True) + self.init

    def proba(self, X) -> np.ndarray:
        """Model probability before any recalibration."""
        return sigmoid(self.raw(X))

    def fit_platt(self, X, y) -> None:
        self.platt = LogisticRegression(C=1e6, max_iter=1000).fit(self.raw(X).reshape(-1, 1), y)

    def proba_calibrated(self, X) -> np.ndarray:
        if self.platt is None:
            raise RuntimeError("Call fit_platt first")
        return self.platt.predict_proba(self.raw(X).reshape(-1, 1))[:, 1]


def _train(params: dict, tr: Split, va: Split, init: float = 0.0, rounds: int = 3000,
           X=None, y=None) -> lgb.Booster:
    X = tr.X if X is None else X
    y = tr.y if y is None else y
    dtr = lgb.Dataset(X, y, init_score=np.full(len(y), init))
    dva = lgb.Dataset(va.X, va.y, init_score=np.full(va.n, init), reference=dtr)
    return lgb.train(params, dtr, num_boost_round=rounds, valid_sets=[dva],
                     callbacks=[lgb.early_stopping(150, verbose=False)])


def baseline(tr: Split, va: Split) -> FittedModel:
    return FittedModel("Baseline (log loss)", _train(BASE_PARAMS, tr, va))


def class_weighted(tr: Split, va: Split) -> FittedModel:
    ratio = (len(tr.y) - tr.n_pos) / tr.n_pos
    params = {**BASE_PARAMS, "scale_pos_weight": ratio}
    return FittedModel("Class weights", _train(params, tr, va), notes={"scale_pos_weight": round(ratio, 1)})


def undersampled(tr: Split, va: Split, neg_per_pos: int = 10) -> FittedModel:
    rng = np.random.default_rng(SEED)
    pos = np.flatnonzero(tr.y == 1)
    neg = rng.choice(np.flatnonzero(tr.y == 0), size=neg_per_pos * len(pos), replace=False)
    idx = np.sort(np.concatenate([pos, neg]))
    booster = _train(BASE_PARAMS, tr, va, X=tr.X.iloc[idx], y=tr.y[idx])
    return FittedModel("Random undersampling", booster, notes={"negatives_per_positive": neg_per_pos})


def focal(tr: Split, va: Split, alphas=(0.25, 0.5, 0.75), gammas=(1.0, 2.0)) -> FittedModel:
    """Grid-search alpha and gamma on validation PR-AUC."""
    best = None
    grid = []
    for alpha, gamma in product(alphas, gammas):
        init = focal_init_score(tr.y, alpha, gamma)
        params = {**BASE_PARAMS, "objective": lgb_focal_objective(alpha, gamma)}
        booster = _train(params, tr, va, init=init)
        model = FittedModel("Focal Loss", booster, init=init, notes={"alpha": alpha, "gamma": gamma})
        ap = average_precision_score(va.y, model.raw(va.X))
        grid.append({"alpha": alpha, "gamma": gamma, "valid_pr_auc": ap, "trees": booster.best_iteration})
        if best is None or ap > best[0]:
            best = (ap, model)
    best[1].notes["grid"] = grid
    return best[1]


ALL = [baseline, class_weighted, undersampled, focal]
