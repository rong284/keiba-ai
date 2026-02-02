from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import log_loss, brier_score_loss


def race_equal_weights(df: pd.DataFrame, race_col: str = "race_id") -> np.ndarray:
    """
    各レースを同じ重みで扱うためのサンプル重み。
    頭数が多いレースが過剰に効くのを防ぐ。
    """
    cnt = df.groupby(race_col)[race_col].transform("count")
    return (1.0 / cnt).values


def clip_prob(p: np.ndarray, eps: float = 1e-15) -> np.ndarray:
    p = np.asarray(p, dtype=float)
    return np.clip(p, eps, 1.0 - eps)


def race_rank_from_pred(df: pd.DataFrame, pred: np.ndarray, race_col: str = "race_id", seed: int = 42) -> pd.Series:
    """
    予測スコアを同一レース内で順位化。
    同点が出ると順位が不安定になるので微小ノイズでタイブレークする。
    """
    tmp = df[[race_col]].copy()
    tmp["pred"] = np.asarray(pred)

    rng = np.random.RandomState(seed)
    tmp["pred_tb"] = tmp["pred"] + 1e-12 * rng.normal(size=len(tmp))

    # 1が最高順位になるように降順で順位付け
    return tmp.groupby(race_col)["pred_tb"].rank(ascending=False, method="first")


def hit_at_k(df: pd.DataFrame, pred: np.ndarray, y: np.ndarray, k: int, race_col: str = "race_id", seed: int = 42) -> float:
    """
    レースごとに「上位kに当たりが含まれるか」を平均する。
    - winモデルなら「勝ち馬がTop-kに入ったか」
    - placeモデルなら「複勝圏の馬がTop-kに入ったか」みたいな解釈
    """
    pr = race_rank_from_pred(df, pred, race_col=race_col, seed=seed)
    tmp = df[[race_col]].copy()
    tmp["pr"] = pr
    tmp["y"] = np.asarray(y)
    return float(tmp.loc[tmp["pr"] <= k].groupby(race_col)["y"].max().mean())


def top1_pos_rate(df: pd.DataFrame, pred: np.ndarray, y: np.ndarray, race_col: str = "race_id", seed: int = 42) -> float:
    pr = race_rank_from_pred(df, pred, race_col=race_col, seed=seed)
    tmp = df[[race_col]].copy()
    tmp["pr"] = pr
    tmp["y"] = np.asarray(y)
    return float(tmp.loc[tmp["pr"] == 1, "y"].mean())


def mrr_for_winner(df: pd.DataFrame, pred: np.ndarray, rank_true: np.ndarray, race_col: str = "race_id", seed: int = 42) -> float:
    """
    MRR（Mean Reciprocal Rank）
    - winner（rank_true==1）が予測順位で何位かの逆数をレース平均
    """
    pr = race_rank_from_pred(df, pred, race_col=race_col, seed=seed)
    tmp = df[[race_col]].copy()
    tmp["pred_rank"] = pr
    tmp["true_rank"] = np.asarray(rank_true)

    # 各レースの勝ち馬の予測順位を取り、1/rank を平均
    win_rows = tmp[tmp["true_rank"] == 1]
    rr = 1.0 / win_rows.groupby(race_col)["pred_rank"].min()
    return float(rr.mean())


def ndcg_at_k(df: pd.DataFrame, pred: np.ndarray, relevance: np.ndarray, k: int, race_col: str = "race_id", seed: int = 42) -> float:
    """
    NDCG@k（Normalized Discounted Cumulative Gain）
    - ランキング学習の代表的な評価指標
    - relevance は「大きいほど良い」(勝ち馬が最大)
    """
    # 予測スコア順に並べてDCGを作る
    tmp = df[[race_col]].copy()
    tmp["pred"] = np.asarray(pred)
    tmp["rel"] = np.asarray(relevance)

    rng = np.random.RandomState(seed)
    tmp["pred_tb"] = tmp["pred"] + 1e-12 * rng.normal(size=len(tmp))

    def dcg(rel_list: np.ndarray) -> float:
        # rel_list は順位順（1位,2位,...）の relevance
        rel_list = rel_list[:k]
        denom = np.log2(np.arange(2, len(rel_list) + 2))
        return float(np.sum(rel_list / denom))

    ndcgs = []
    for rid, g in tmp.groupby(race_col):
        g_sorted = g.sort_values("pred_tb", ascending=False)
        rel_pred = g_sorted["rel"].to_numpy()

        # ideal（真のrelevanceを降順）
        rel_ideal = np.sort(g["rel"].to_numpy())[::-1]

        dcg_pred = dcg(rel_pred)
        dcg_ideal = dcg(rel_ideal)
        nd = 0.0 if dcg_ideal == 0 else dcg_pred / dcg_ideal
        ndcgs.append(nd)

    return float(np.mean(ndcgs))


def prob_metrics(y: np.ndarray, p: np.ndarray) -> Dict[str, float]:
    """
    確率としての品質：
    - LogLoss（小さいほど良い）
    - Brier（小さいほど良い）
    """
    y = np.asarray(y).astype(int)
    p = clip_prob(p)
    return {
        "logloss": float(log_loss(y, p)),
        "brier": float(brier_score_loss(y, p)),
        "mean_pred": float(np.mean(p)),
        "pos_rate": float(np.mean(y)),
    }
