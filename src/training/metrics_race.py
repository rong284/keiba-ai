# 旧パス: src/train/metrics_race.py
from __future__ import annotations
import numpy as np
import pandas as pd

def prep_group_index(race_ids, seed: int = 42):
    """
    race_ids から「同一レースの塊」を作るための前処理。
    学習中に毎回ソートしないで済むよう、index類を事前に作って返す。
    """
    race_ids = np.asarray(race_ids)
    # 安定のため factorize（sort=True）
    race_codes, _ = pd.factorize(race_ids, sort=True)
    race_codes = race_codes.astype(np.int32)

    # レースコード順に並べ替えた並び順と、グループ開始位置(starts), 件数(counts)
    order = np.argsort(race_codes, kind="mergesort")
    rc = race_codes[order]
    change = np.flatnonzero(rc[1:] != rc[:-1]) + 1
    starts = np.r_[0, change].astype(np.int32)
    counts = np.diff(np.r_[starts, len(rc)]).astype(np.int32)

    rng = np.random.RandomState(seed)
    noise = 1e-12 * rng.normal(size=len(rc))  # 同点崩し用のノイズ
    return order, starts, counts, noise

def top1_pos_rate_fast(preds: np.ndarray, y: np.ndarray, order, starts, counts, noise=None) -> float:
    """
    各レースでスコア最大の馬を1頭選び、その y の平均を返す。
    例）win: y=勝ち馬(1/0) → 予測1位が勝った率
        place: y=3着以内(1/0) → 予測1位が3着以内だった率
    """
    preds = np.asarray(preds, dtype=float)
    y = np.asarray(y, dtype=float)

    # レースコード順に並んだ予測
    s = preds[order]
    if noise is not None:
        s = s + noise

    # 各レースの最大値（reduceatはグループ開始位置に対する集約）
    max_s = np.maximum.reduceat(s, starts)
    max_rep = np.repeat(max_s, counts)

    # ノイズで一意になっている前提 → 各レースで最大の場所が1個
    pos = np.flatnonzero(s == max_rep)
    top_idx = order[pos]
    return float(np.mean(y[top_idx]))
