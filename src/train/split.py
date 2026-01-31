from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Tuple

import pandas as pd


@dataclass(frozen=True)
class TimeFold:
    train_end: str
    valid_start: str
    valid_end: str


def make_time_folds(df: pd.DataFrame, folds: List[TimeFold], date_col: str = "race_date"):
    """
    時系列分割（未来を見ない）。
    - train: race_date <= train_end
    - valid: valid_start <= race_date <= valid_end
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])

    out = []
    for f in folds:
        tr = d[d[date_col] <= pd.Timestamp(f.train_end)].copy()
        va = d[(d[date_col] >= pd.Timestamp(f.valid_start)) & (d[date_col] <= pd.Timestamp(f.valid_end))].copy()
        out.append((tr, va, f))
    return out


def make_holdout(df: pd.DataFrame, train_end: str, test_start: str, date_col: str = "race_date"):
    """
    最終ホールドアウト（例：2025年）。
    - train: <= train_end
    - test : >= test_start
    """
    d = df.copy()
    d[date_col] = pd.to_datetime(d[date_col])
    train = d[d[date_col] <= pd.Timestamp(train_end)].copy()
    test = d[d[date_col] >= pd.Timestamp(test_start)].copy()
    return train, test
