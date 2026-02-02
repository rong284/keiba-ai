# 旧パス: src/data/preprocess/race_info.py
import pandas as pd
import re
import ast
from src.data_collection.common.mappings import (
    WEATHER_MAP, GROUND_MAP, RACE_TYPE_MAP, PLACE_MAP, race_class_mapper
)

def preprocess_race_info_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    # info1 / info2 はリスト文字列
    df["info1"] = df["info1"].apply(ast.literal_eval)
    df["info2"] = df["info2"].apply(ast.literal_eval)

    # 日付
    df["date"] = df["info2"].apply(
        lambda x: pd.to_datetime(x[0], format="%Y年%m月%d日", errors="coerce")
    )

    # 開催
    def extract_place(info2):
        if not isinstance(info2, list) or len(info2) < 2:
            return None
        text = str(info2[1])
        for k, v in PLACE_MAP.items():
            if k in text:
                return v
        return None
    df["place"] = df["info2"].apply(extract_place)

    # レース種別
    df["race_type"] = df["info1"].apply(lambda x: RACE_TYPE_MAP.get(str(x[0])[0]) if isinstance(x, list) and len(x) else None)

    # 回り
    def extract_around(text):
        if not isinstance(text, str):
            return None
        if "右" in text:
            return 0
        if "左" in text:
            return 1
        if "直" in text:
            return 2
        return None
    df["around"] = df["info1"].apply(lambda x: extract_around(str(x[0])) if isinstance(x, list) and len(x) else None)

    # 距離
    def extract_course_len(info1):
        # info1 は ['障芝外', '内2890m', ...] のようなリスト
        if isinstance(info1, list):
            for s in info1:
                if isinstance(s, str):
                    m = re.search(r"(\d+)m", s)
                    if m:
                        return int(m.group(1))
            return None
        if isinstance(info1, str):
            m = re.search(r"(\d+)m", info1)
            return int(m.group(1)) if m else None
        return None
    df["course_len"] = df["info1"].apply(extract_course_len)

    # 天候
    def extract_weather(info1):
        if not isinstance(info1, list):
            return None
        for s in info1:
            if isinstance(s, str) and s.startswith("天候:"):
                return WEATHER_MAP.get(s.split(":")[1])
        return None
    df["weather"] = df["info1"].apply(extract_weather)

    # 馬場状態
    def extract_ground_state(info1):
        if not isinstance(info1, list):
            return None
        for s in info1:
            if isinstance(s, str) and ":" in s:
                val = s.split(":")[1]
                if val in GROUND_MAP:
                    return GROUND_MAP[val]
        return None
    df["ground_state"] = df["info1"].apply(extract_ground_state)

    # クラス
    df["race_class"] = df["title"].apply(race_class_mapper)

    use_cols = [
        "race_id",
        "date",
        "race_type",
        "around",
        "course_len",
        "weather",
        "ground_state",
        "race_class",
        "place",
    ]
    return df[use_cols]
