from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    p = np.clip(p, eps, 1.0 - eps)
    return np.log(p / (1.0 - p))


def odds_to_prob(
    df: pd.DataFrame,
    odds_col: str,
    race_col: str = "race_id",
) -> np.ndarray:
    odds = pd.to_numeric(df[odds_col], errors="coerce")
    inv = 1.0 / odds
    inv = inv.replace([np.inf, -np.inf], np.nan).fillna(0.0)
    sum_inv = inv.groupby(df[race_col]).transform("sum")
    p = inv / sum_inv

    mask = sum_inv <= 0
    if mask.any():
        cnt = df.groupby(race_col)[race_col].transform("count")
        p = p.where(~mask, 1.0 / cnt)
    return p.values.astype(float)


def odds_base_margin(
    df: pd.DataFrame,
    odds_col: str,
    race_col: str = "race_id",
    eps: float = 1e-6,
) -> np.ndarray:
    p_odds = odds_to_prob(df, odds_col=odds_col, race_col=race_col)
    return logit(p_odds, eps=eps)
