from __future__ import annotations

from dataclasses import dataclass
from typing import List, Tuple

import pandas as pd


@dataclass(frozen=True)
class FeatureSpec:
    """
    どの列を特徴量として使うかを一元管理する。
    """
    feature_cols: List[str]
    cat_cols: List[str]
    dropped_cols: List[str]


MARKET_COLS = ["popularity", "tansho_odds"]

CAT_COLS_CAND = [
    "place", "race_type", "around", "dist_bin", "weather", "ground_state", "race_class",
    "wakuban", "umaban", "sex",
    "jockey_id", "trainer_id", "owner_id",
]


def build_feature_spec(df: pd.DataFrame, use_market: bool) -> FeatureSpec:
    """
    - リーク確定列やIDを落とす
    - market列を使う/使わないを切り替え
    - categorical候補が存在すれば cat_cols に入れる
    """
    always_drop = [
        "race_id", "horse_id", "race_date",
        "rank", "rank_raw",
        "y_top1", "y_top3",
    ]

    drop_cols = [c for c in always_drop if c in df.columns]
    if not use_market:
        drop_cols += [c for c in MARKET_COLS if c in df.columns]

    drop_cols = list(dict.fromkeys(drop_cols))  # 重複排除（順序維持）

    cat_cols = [c for c in CAT_COLS_CAND if c in df.columns]
    feature_cols = [c for c in df.columns if c not in drop_cols]

    # ガード：絶対に入れない列（安全柵）
    forbidden = set(["rank", "rank_raw", "y_top1", "y_top3"])
    bad = [c for c in feature_cols if c in forbidden]
    assert not bad, f"LEAK columns found in features: {bad}"

    return FeatureSpec(feature_cols=feature_cols, cat_cols=cat_cols, dropped_cols=drop_cols)


def make_X(df: pd.DataFrame, feature_cols: List[str], cat_cols: List[str]) -> pd.DataFrame:
    """
    LightGBMに渡すXを作る。
    - categorical列は pandas.category にしておく（LightGBMが扱いやすい）
    """
    X = df[feature_cols].copy()
    for c in cat_cols:
        if c in X.columns:
            X[c] = X[c].astype("category")
    return X


def align_categories(X_tr: pd.DataFrame, X_te: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    """
    重要：学習側のカテゴリ集合に合わせる（推論で未知カテゴリが出たときの事故防止）
    """
    X_te = X_te.copy()
    for c in cat_cols:
        if c in X_tr.columns and str(X_tr[c].dtype) == "category":
            X_te[c] = pd.Categorical(X_te[c], categories=X_tr[c].cat.categories)
    return X_te
