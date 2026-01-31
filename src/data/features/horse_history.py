import pandas as pd
from src.data.features.feature_engineering import _rolling_mean_by_horse


def add_last3_mean_rank(df_base: pd.DataFrame, df_horse: pd.DataFrame) -> pd.DataFrame:
    """
    df_base（主: df_result + df_race_info を結合して race_date を持っている想定）に、
    df_horse から「直近3走の平均着順」をリーク無しで付与する。

    リーク防止:
      - horseごと日付順
      - shift(1) で「当日成績」を過去から外す（=当日を見ない）

    Args:
        df_base: columns 必須: ["horse_id", "race_date", ...]
        df_horse: columns 必須: ["horse_id", "date", "rank", ...]

    Returns:
        df_base に hr_rank_mean_3 を追加した DataFrame
    """
    base = df_base.copy()
    h = df_horse[["horse_id", "date", "rank"]].copy()

    base["race_date"] = pd.to_datetime(base["race_date"], errors="coerce")
    h["date"] = pd.to_datetime(h["date"], errors="coerce")

    # horseごとに時系列ソート
    h = h.sort_values(["horse_id", "date"], kind="mergesort")

    # 当日を過去に含めない（リーク防止）
    h["rank_prev"] = h.groupby("horse_id")["rank"].shift(1)

    # 直近3走平均（min_periods=1: 過去1走でも値を出す。新馬等はNaNになりやすいので実用的）
    h["hr_rank_mean_3"] = _rolling_mean_by_horse(h, "rank_prev", window=3)

    # 付与用の最小テーブル（horse_id + date で一意な想定）
    feat = h[["horse_id", "date", "hr_rank_mean_3"]].rename(columns={"date": "race_date"})

    # 主テーブルへ結合
    if "hr_rank_mean_3" in base.columns:
        base = base.drop(columns=["hr_rank_mean_3"])

    out = base.merge(
        feat,
        on=["horse_id", "race_date"],
        how="left",
        validate="many_to_one"
    )
    return out
