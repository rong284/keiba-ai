from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
from src.utils.progress import log, tqdm

# 既存ローダ・パイプラインをそのまま使う
from src.data_collection.loaders.result_loader import load_results
from src.data_collection.loaders.horse_loader import load_horse_results
from src.data_collection.loaders.race_info_loader import load_race_info
from src.data_collection.pipelines.build_train_table import build_train_table


@dataclass(frozen=True)
class DataPaths:
    """
    学習で使う生データのglobパスをまとめるだけ。
    """
    result_glob: str
    horse_glob: str
    race_info_glob: str


def load_train_dataframe(paths: DataPaths) -> pd.DataFrame:
    """
    1) rawdfを読み込む
    2) build_train_table で (race_id, horse_id) 粒度に整形
    3) rank/race_date の最低限のクリーニング
    4) y_top1, y_top3 を作る（後段で win/place を切替）
    """
    steps = tqdm(total=6, desc="train data", leave=False)

    steps.set_description("load result csv")
    df_result = load_results(paths.result_glob)
    steps.update(1)

    steps.set_description("load horse csv")
    df_horse = load_horse_results(paths.horse_glob)
    steps.update(1)

    steps.set_description("load race info csv")
    df_race_info = load_race_info(paths.race_info_glob)
    steps.update(1)

    steps.set_description("build train table")
    df = build_train_table(df_result, df_race_info, df_horse).copy()
    steps.update(1)

    # ---- 基本クリーニング ----
    steps.set_description("basic cleaning")
    df["race_date"] = pd.to_datetime(df["race_date"], errors="coerce")

    # rank混在（例：中止/除外/取消など）への耐性を上げる
    df["rank_raw"] = df["rank"].astype(str)
    df["rank"] = df["rank_raw"].str.extract(r"(\d+)")[0]
    df["rank"] = pd.to_numeric(df["rank"], errors="coerce")

    before = len(df)
    df = df.dropna(subset=["race_id", "horse_id", "race_date", "rank"]).copy()
    after = len(df)
    df["rank"] = df["rank"].astype(int)

    log(f"[data] dropped rows: {before-after:,}")
    log(f"[data] n_races={df['race_id'].nunique():,} n_rows={len(df):,}")
    steps.update(1)

    # ---- 目的変数（2値） ----
    steps.set_description("make targets")
    df["y_top1"] = (df["rank"] <= 1).astype(int)
    df["y_top3"] = (df["rank"] <= 3).astype(int)
    steps.update(1)
    steps.close()

    return df
