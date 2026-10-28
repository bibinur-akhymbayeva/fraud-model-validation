"""Loading and out-of-time splitting of the credit card fraud dataset."""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

PCA_FEATURES = [f"V{i}" for i in range(1, 29)]
FEATURES = PCA_FEATURES + ["Amount", "hour"]
TARGET = "Class"
EXPECTED = set(PCA_FEATURES + ["Time", "Amount", TARGET])


@dataclass
class Split:
    name: str
    X: pd.DataFrame
    y: np.ndarray
    time: np.ndarray

    @property
    def n(self) -> int:
        return len(self.y)

    @property
    def n_pos(self) -> int:
        return int(self.y.sum())


def load(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    missing = EXPECTED - set(df.columns)
    if missing:
        raise ValueError(f"Dataset is missing columns: {sorted(missing)}")
    # 'Time' is seconds since the first transaction. Using it directly would let the
    # model learn the calendar position of the sample, which cannot generalise.
    # Hour of day keeps the behavioural signal without that leak.
    df["hour"] = (df["Time"] % 86_400) / 3_600
    return df.sort_values("Time", kind="stable").reset_index(drop=True)


def time_split(df: pd.DataFrame, train_frac: float = 0.6, valid_frac: float = 0.2) -> dict[str, Split]:
    """Chronological train / validation / out-of-time test split."""
    n = len(df)
    a, b = int(n * train_frac), int(n * (train_frac + valid_frac))
    parts = {"train": df.iloc[:a], "valid": df.iloc[a:b], "test": df.iloc[b:]}
    return {k: Split(k, v[FEATURES].copy(), v[TARGET].to_numpy(), v["Time"].to_numpy()) for k, v in parts.items()}
