# src/data/preprocess/horse.py
import pandas as pd
import re
from src.data.common.mappings import (
    WEATHER_MAP, GROUND_MAP, RACE_TYPE_MAP, PLACE_MAP, race_class_mapper
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

    # time (mm:ss.s) -> seconds
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

    # passing (e.g. "8-7-6-7") -> first / avg / last
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

    # last 3f
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
