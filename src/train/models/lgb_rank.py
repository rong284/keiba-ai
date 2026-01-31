from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
import lightgbm as lgb

from src.train.features import make_X, align_categories
from src.train.metrics import race_equal_weights


def sort_by_group(df: pd.DataFrame, group_col: str = "race_id") -> pd.DataFrame:
    """
    Rankingは group（クエリ）単位で Dataset を作るので
    - race_id で並べておく（groupサイズ計算と行順が一致するように）
    """
    return df.sort_values([group_col, "horse_id"]).reset_index(drop=True)


def make_relevance_from_rank(df: pd.DataFrame) -> np.ndarray:
    """
    LightGBMのrankingは「ラベルが大きいほど良い」前提。
    競馬のrank（1が勝ち）を relevance に変換する。

    例：rank=1 -> 最大、rankが大きいほど小さく
    """
    r = df["rank"].to_numpy()
    max_r = int(np.nanmax(r))
    rel = (max_r + 1) - r  # 1位が最大
    return rel.astype(float)


@dataclass
class RankTrainResult:
    model: lgb.Booster
    best_iter: int


def train_rank(
    tr_df: pd.DataFrame,
    va_df: Optional[pd.DataFrame],
    feature_cols: List[str],
    cat_cols: List[str],
    params: Dict,
    eval_at: List[int],
    seed: int = 42,
    num_boost_round: int = 5000,
    early_stopping_rounds: Optional[int] = 200,
) -> RankTrainResult:
    """
    LambdaRank（ランキング学習）で学習する。
    - objective='lambdarank', metric='ndcg' が基本 :contentReference[oaicite:5]{index=5}
    """
    tr = sort_by_group(tr_df, "race_id")
    X_tr = make_X(tr, feature_cols, cat_cols)
    y_tr = make_relevance_from_rank(tr)

    # rankingでも “レース均等” の思想は残す（ただしgroup学習なので重みは慎重に）
    # ここではサンプル重みを入れる（レース頭数の逆数）
    w_tr = race_equal_weights(tr, "race_id")

    group_tr = tr.groupby("race_id").size().to_list()

    p = dict(params)
    p.update({
        "objective": "lambdarank",
        "metric": ["ndcg"],
        # ndcg@k を見たいkのリスト（LightGBMの標準パラメータ）
        "eval_at": eval_at,
        # ndcg_eval_at は eval_at のエイリアス（明示しておく）
        "ndcg_eval_at": eval_at,
        "seed": seed,
        "data_random_seed": seed,
        "feature_fraction_seed": seed,
        "bagging_seed": seed,
        "verbosity": p.get("verbosity", -1),
    })

    dtr = lgb.Dataset(
        X_tr,
        label=y_tr,
        weight=w_tr,
        group=group_tr,
        categorical_feature=cat_cols,
        free_raw_data=False
    )

    valid_sets = []
    valid_names = []
    callbacks = []

    if va_df is not None:
        va = sort_by_group(va_df, "race_id")
        X_va = make_X(va, feature_cols, cat_cols)
        X_va = align_categories(X_tr, X_va, cat_cols)
        y_va = make_relevance_from_rank(va)
        w_va = race_equal_weights(va, "race_id")
        group_va = va.groupby("race_id").size().to_list()

        dva = lgb.Dataset(
            X_va, label=y_va, weight=w_va, group=group_va,
            categorical_feature=cat_cols, free_raw_data=False
        )

        valid_sets = [dva]
        valid_names = ["valid"]
        callbacks = [lgb.log_evaluation(0)]
        if early_stopping_rounds is not None:
            # ndcg@3 を先頭に置けば、first_metric_only=True で ndcg@3 を監視できる
            callbacks.insert(0, lgb.early_stopping(int(early_stopping_rounds), first_metric_only=True))

        model = lgb.train(
            p, dtr,
            num_boost_round=int(num_boost_round),
            valid_sets=valid_sets,
            valid_names=valid_names,
            callbacks=callbacks
        )
        best_iter = int(model.best_iteration)
    else:
        # holdout無し学習（最終学習など）
        model = lgb.train(
            p, dtr,
            num_boost_round=int(num_boost_round),
            callbacks=[lgb.log_evaluation(0)],
        )
        best_iter = int(num_boost_round)

    return RankTrainResult(model=model, best_iter=best_iter)


def predict_rank(
    model: lgb.Booster,
    train_ref_df: pd.DataFrame,
    target_df: pd.DataFrame,
    feature_cols: List[str],
    cat_cols: List[str],
) -> np.ndarray:
    """
    rankingも分類と同じくカテゴリ整合を取って推論する
    """

    X_tr = make_X(train_ref_df, feature_cols, cat_cols)
    X_te = make_X(target_df, feature_cols, cat_cols)
    X_te = align_categories(X_tr, X_te, cat_cols)

    return model.predict(X_te, num_iteration=model.best_iteration)
