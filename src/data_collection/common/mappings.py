# 旧パス: src/data/common/mappings.py
import re
import pandas as pd

WEATHER_MAP = {"晴": 0, "曇": 1, "雨": 2, "小雨": 2, "雪": 3, "小雪": 3}

GROUND_MAP = {
    "良": 0,
    "稍": 1,
    "稍重": 1,
    "重": 2,
    "不": 3,
    "不良": 3,
}

RACE_TYPE_MAP = {"芝": 0, "ダ": 1, "障": 2}

PLACE_MAP = {
    "札幌": 0, "函館": 1, "福島": 2, "新潟": 3,
    "東京": 4, "中山": 5, "中京": 6, "京都": 7,
    "阪神": 8, "小倉": 9,
    "門別": 10, "盛岡": 11, "水沢": 12,
    "浦和": 13, "船橋": 14, "大井": 15, "川崎": 16,
    "金沢": 17, "笠松": 18, "名古屋": 19,
    "園田": 20, "姫路": 21, "高知": 22, "佐賀": 23,
}

def race_class_mapper(name):
    if pd.isna(name):
        return -4
    name = str(name)

    if re.search(r"GIII|GⅢ|G3", name) or "JpnIII" in name:
        return 3
    if re.search(r"GII|GⅡ|G2", name) or "JpnII" in name:
        return 4
    if re.search(r"GI|G1", name) or "JpnI" in name:
        return 5

    if "(L)" in name or "リステッド" in name:
        return 2.5
    if "OP" in name or "オープン" in name or "特別" in name:
        return 2

    if "3勝" in name or "1600万" in name:
        return 1
    if "2勝" in name or "1000万" in name:
        return 0
    if "1勝" in name or "500万" in name:
        return -1

    if "新馬" in name:
        return -2
    if "未勝利" in name:
        return -3

    if name.startswith("A"):
        return 0.5
    if name.startswith("B"):
        return 0
    if name.startswith("C"):
        return -0.5

    return -4
