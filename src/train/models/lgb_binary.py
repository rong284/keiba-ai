from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.train.features import make_X, align_categories
from src.train.metrics import race_equal_weights
from src.train.metrics_race import prep_group_index, top1_pos_rate_fast


def calc_scale_pos_weight(y: np.ndarray) -> float:
    """
    class imbalance 対策：
    - 陽性（1）が少ないので neg/pos を与える（LightGBMの定番）
    """
    y = np.asarray(y)
    pos = int((y == 1).sum())
    neg = int((y == 0).sum())
    return neg / max(pos, 1)


@dataclass
class BinaryTrainResult:
    model: lgb.Booster
    best_iter: int
    scale_pos_weight: float
    evals_result: Optional[Dict[str, Dict[str, List[float]]]] = None


def train_binary(
    tr_df: pd.DataFrame,
    va_df: Optional[pd.DataFrame],
    y_col: str,
    feature_cols: List[str],
    cat_cols: List[str],
    params: Dict,
    seed: int = 42,
) -> BinaryTrainResult:
    """
    LightGBMのtrain APIで2値分類を学習する。
    - va_df があれば early stopping を使う（推奨）:contentReference[oaicite:4]{index=4}
    """
    y_tr = tr_df[y_col].values.astype(int)
    X_tr = make_X(tr_df, feature_cols, cat_cols)
    w_tr = race_equal_weights(tr_df, "race_id")

    p = dict(params)
    p.update({
        "objective": "binary",
        # feval を主指標にするためデフォルトmetricを無効化
        "metric": "None",
        "seed": seed,
        "data_random_seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "verbosity": p.get("verbosity", -1),
        "scale_pos_weight": calc_scale_pos_weight(y_tr),
    })

    dtr = lgb.Dataset(X_tr, label=y_tr, weight=w_tr, categorical_feature=cat_cols, free_raw_data=False)

    valid_sets = []
    valid_names = []
    callbacks = []
    evals_result = None

    if va_df is not None:
        y_va = va_df[y_col].values.astype(int)
        X_va = make_X(va_df, feature_cols, cat_cols)
        X_va = align_categories(X_tr, X_va, cat_cols)
        w_va = race_equal_weights(va_df, "race_id")
        dva = lgb.Dataset(X_va, label=y_va, weight=w_va, categorical_feature=cat_cols, free_raw_data=False)

        valid_sets = [dva]
        valid_names = ["valid"]

        order, starts, counts, noise = prep_group_index(va_df["race_id"].values, seed=seed)

        def feval_top1(preds, data):
            y = data.get_label()
            score = top1_pos_rate_fast(preds, y, order, starts, counts, noise=noise)
            return ("top1_pos_rate", score, True)

        evals_result = {}
        callbacks = [
            lgb.record_evaluation(evals_result),
            lgb.early_stopping(200, first_metric_only=True),
            lgb.log_evaluation(0),
        ]

        model = lgb.train(
            p, dtr,
            num_boost_round=5000,
            valid_sets=valid_sets,
            valid_names=valid_names,
            feval=feval_top1,
            callbacks=callbacks
        )
        best_iter = int(model.best_iteration)
    else:
        # holdout無し学習（最終学習など）
        model = lgb.train(p, dtr, num_boost_round=2000)
        best_iter = int(model.current_iteration())

    return BinaryTrainResult(
        model=model,
        best_iter=best_iter,
        scale_pos_weight=float(p["scale_pos_weight"]),
        evals_result=evals_result,
    )


def predict_binary(
    model: lgb.Booster,
    train_ref_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
) -> np.ndarray:
    """
    学習側カテゴリに合わせて推論する（align_categories が重要）
    """
    X_tr = make_X(train_ref_df, feature_cols, cat_cols)
    X_te = make_X(target_df, feature_cols, cat_cols)
    X_te = align_categories(X_tr, X_te, cat_cols)
    return model.predict(X_te, num_iteration=model.best_iteration)
