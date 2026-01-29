# src/data/preprocess/horse.py
import pandas as pd
import re
from src.data.common.mappings import (
    WEATHER_MAP, GROUND_MAP, RACE_TYPE_MAP, race_class_mapper
)

def preprocess_horse_results_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # date
    df["date"] = pd.to_datetime(df["日付"], errors="coerce")

    # rank
    df["rank"] = pd.to_numeric(df["着順"], errors="coerce")
    df = df.dropna(subset=["rank"])
    df["rank"] = df["rank"].astype(int)

    # prize
    df["prize"] = pd.to_numeric(df["賞金"], errors="coerce").fillna(0.0)

    # rank_diff（着差）
    # 例: "0.0", "0.2", "-0.1", "ハナ", "クビ" 等が混ざる想定 → 数値化できないものはNaN
    rd = pd.to_numeric(df["着差"], errors="coerce")
    df["rank_diff"] = rd.mask(rd < 0, 0)  # 勝ち馬側にマイナス表記が来ても 0 扱い

    # weather
    df["weather"] = df["天気"].map(WEATHER_MAP)

    # race_type / course_len
    dist = df["距離"].astype(str)
    df["race_type"] = dist.str[0].map(RACE_TYPE_MAP)
    df["course_len"] = pd.to_numeric(dist.str.extract(r"(\d+)")[0], errors="coerce")

    # ground_state
    df["ground_state"] = df["馬場"].map(GROUND_MAP)

    # race_class
    df["race_class"] = df["レース名"].apply(race_class_mapper)

    # n_horses
    df["n_horses"] = pd.to_numeric(df["頭数"], errors="coerce")

    use_cols = [
        "horse_id",
        "date",
        "rank",
        "prize",
        "rank_diff",
        "weather",
        "race_type",
        "course_len",
        "ground_state",
        "race_class",
        "n_horses",
    ]
    return df[use_cols]
