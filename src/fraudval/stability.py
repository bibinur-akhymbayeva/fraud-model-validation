"""Population stability and drift diagnostics."""
from __future__ import annotations

import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import StratifiedKFold


def psi(expected: np.ndarray, actual: np.ndarray, bins: int = 10, eps: float = 1e-4) -> float:
    """Population Stability Index with quantile bins taken from the expected sample.

    Common reading: below 0.10 stable, 0.10 to 0.25 moderate shift, above 0.25 significant shift.
    """
    edges = np.unique(np.quantile(expected, np.linspace(0, 1, bins + 1)))
    edges[0], edges[-1] = -np.inf, np.inf
    e = np.histogram(expected, edges)[0] / len(expected)
    a = np.histogram(actual, edges)[0] / len(actual)
    e, a = np.clip(e, eps, None), np.clip(a, eps, None)
    return float(np.sum((a - e) * np.log(a / e)))


def feature_psi(X_ref: pd.DataFrame, X_new: pd.DataFrame) -> pd.Series:
    return pd.Series({c: psi(X_ref[c].to_numpy(), X_new[c].to_numpy()) for c in X_ref.columns}).sort_values(ascending=False)


def adversarial_validation(X_ref: pd.DataFrame, X_new: pd.DataFrame, folds: int = 5, seed: int = 42) -> dict:
    """Train a classifier to tell the two samples apart.

    An AUC near 0.5 means the samples are indistinguishable; values well above 0.5
    mean the feature distribution has shifted, and the importances show where.
    """
    X = pd.concat([X_ref, X_new], ignore_index=True)
    y = np.r_[np.zeros(len(X_ref)), np.ones(len(X_new))]
    oof = np.zeros(len(y))
    importance = pd.Series(0.0, index=X.columns)
    params = dict(objective="binary", learning_rate=0.1, num_leaves=31, verbose=-1, seed=seed)
    for tr, te in StratifiedKFold(folds, shuffle=True, random_state=seed).split(X, y):
        m = lgb.train(params, lgb.Dataset(X.iloc[tr], y[tr]), num_boost_round=200)
        oof[te] = m.predict(X.iloc[te])
        importance += pd.Series(m.feature_importance("gain"), index=X.columns)
    return {"auc": float(roc_auc_score(y, oof)), "importance": (importance / importance.sum()).sort_values(ascending=False)}
