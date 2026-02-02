# 旧パス: src/data/preprocess/horse.py
import pandas as pd
import re
from src.data_collection.common.mappings import (
    WEATHER_MAP, GROUND_MAP, RACE_TYPE_MAP, PLACE_MAP, race_class_mapper
)

def preprocess_horse_results_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # 日付
    df["date"] = pd.to_datetime(df["日付"], errors="coerce")

    # 着順
    df["rank"] = pd.to_numeric(df["着順"], errors="coerce")
    df = df.dropna(subset=["rank"])
    df["rank"] = df["rank"].astype(int)

    # 賞金
    df["prize"] = pd.to_numeric(df["賞金"], errors="coerce").fillna(0.0)

    # rank_diff（着差）
    # 例: "0.0", "0.2", "-0.1", "ハナ", "クビ" 等が混ざる想定 → 数値化できないものはNaN
    rd = pd.to_numeric(df["着差"], errors="coerce")
    df["rank_diff"] = rd.mask(rd < 0, 0)  # 勝ち馬側にマイナス表記が来ても 0 扱い

    # 天候
    df["weather"] = df["天気"].map(WEATHER_MAP)

    # レース種別 / 距離
    dist = df["距離"].astype(str)
    df["race_type"] = dist.str[0].map(RACE_TYPE_MAP)
    df["course_len"] = pd.to_numeric(dist.str.extract(r"(\d+)")[0], errors="coerce")

    # 馬場状態
    df["ground_state"] = df["馬場"].map(GROUND_MAP)

    # クラス
    df["race_class"] = df["レース名"].apply(race_class_mapper)

    # 頭数
    df["n_horses"] = pd.to_numeric(df["頭数"], errors="coerce")

    # 走破タイム (mm:ss.s) -> 秒
    def to_seconds(val):
        if pd.isna(val):
            return None
        s = str(val).strip()
        if not s or s == "**":
            return None
        m = re.match(r"^(\d+):(\d+\.?\d*)$", s)
        if not m:
            return None
        return float(m.group(1)) * 60.0 + float(m.group(2))
    df["time_sec"] = df["タイム"].apply(to_seconds)

    # 通過順 (例: "8-7-6-7") -> 先頭 / 平均 / 最後
    def parse_passing(val):
        if pd.isna(val):
            return (None, None, None)
        s = str(val).strip()
        if not s or s == "**":
            return (None, None, None)
        parts = []
        for p in s.replace(" ", "").split("-"):
            try:
                parts.append(int(p))
            except ValueError:
                return (None, None, None)
        if not parts:
            return (None, None, None)
        first = parts[0]
        avg = sum(parts) / len(parts)
        last = parts[-1]
        return (first, avg, last)
    pass_parsed = df["通過"].apply(parse_passing)
    df["passing_first"] = pass_parsed.apply(lambda x: x[0])
    df["passing_avg"] = pass_parsed.apply(lambda x: x[1])
    df["passing_last"] = pass_parsed.apply(lambda x: x[2])

    # 上がり3F
    df["up3f"] = pd.to_numeric(df["上り"], errors="coerce")

    # place（開催名→コード）
    def extract_place(text):
        if pd.isna(text):
            return None
        text = str(text)
        for k, v in PLACE_MAP.items():
            if k in text:
                return v
        return None
    df["place"] = df["開催"].apply(extract_place)

    use_cols = [
        "horse_id",
        "date",
        "rank",
        "prize",
        "rank_diff",
        "time_sec",
        "passing_first",
        "passing_avg",
        "passing_last",
        "up3f",
        "weather",
        "race_type",
        "course_len",
        "ground_state",
        "race_class",
        "n_horses",
        "place",
    ]
    return df[use_cols]
