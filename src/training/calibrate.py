from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, Tuple

import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression

from src.training.metrics import clip_prob, prob_metrics, race_equal_weights


@dataclass
class CalibModels:
    """
    - platt: ロジ回帰で確率を補正（Platt scaling）
    - iso  : 単調変換で補正（Isotonic）
    """
    platt: LogisticRegression
    iso: IsotonicRegression


def fit_calibrators(p_cal: np.ndarray, y_cal: np.ndarray, w_cal: np.ndarray) -> CalibModels:
    """
    sample_weight を入れて “レース均等” を保つ。
    IsotonicRegression も sample_weight 対応。:contentReference[oaicite:6]{index=6}
    """
    p_cal = clip_prob(p_cal)
    y_cal = np.asarray(y_cal).astype(int)

    platt = LogisticRegression(max_iter=200)
    platt.fit(p_cal.reshape(-1, 1), y_cal, sample_weight=w_cal)

    iso = IsotonicRegression(out_of_bounds="clip")
    iso.fit(p_cal, y_cal, sample_weight=w_cal)

    return CalibModels(platt=platt, iso=iso)


def apply_platt(platt: LogisticRegression, p: np.ndarray) -> np.ndarray:
    p = clip_prob(p)
    return platt.predict_proba(p.reshape(-1, 1))[:, 1]


def apply_iso(iso: IsotonicRegression, p: np.ndarray) -> np.ndarray:
    p = clip_prob(p)
    return iso.transform(p)


def choose_best_calibrator(y_eval: np.ndarray, p_raw: np.ndarray, calib: CalibModels) -> Tuple[str, Dict[str, float]]:
    """
    LogLoss最小のものを採用（raw / platt / iso の中から選ぶ）
    確率校正の基本的な目的は「確率として正しい」こと。:contentReference[oaicite:7]{index=7}
    """
    p_pl = apply_platt(calib.platt, p_raw)
    p_iso = apply_iso(calib.iso, p_raw)

    rep_raw = prob_metrics(y_eval, p_raw)
    rep_pl = prob_metrics(y_eval, p_pl)
    rep_iso = prob_metrics(y_eval, p_iso)

    candidates = {
        "raw": rep_raw,
        "platt": rep_pl,
        "iso": rep_iso,
    }
    best = min(candidates.items(), key=lambda kv: kv[1]["logloss"])
    return best[0], best[1]


def apply_best(kind: str, p_raw: np.ndarray, calib: CalibModels) -> np.ndarray:
    if kind == "raw":
        return p_raw
    if kind == "platt":
        return apply_platt(calib.platt, p_raw)
    if kind == "iso":
        return apply_iso(calib.iso, p_raw)
    raise ValueError(kind)
