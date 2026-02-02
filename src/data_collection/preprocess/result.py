# src/data/preprocess/result.py
import pandas as pd
import re

def preprocess_race_results_df(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    
    df["rank"] = pd.to_numeric(df["着順"], errors="coerce")
    df = df.dropna(subset=["rank"])
    df["rank"] = df["rank"].astype(int)

    df["wakuban"] = pd.to_numeric(df["枠番"], errors="coerce").astype("Int64")
    df["umaban"] = pd.to_numeric(df["馬番"], errors="coerce").astype("Int64")

    sex_mapping = {"牡": 0, "牝": 1, "セ": 2}
    df["sex"] = df["性齢"].str[0].map(sex_mapping)
    df["age"] = pd.to_numeric(df["性齢"].str[1:], errors="coerce").astype("Int64")

    df["impost"] = pd.to_numeric(df["斤量"], errors="coerce")
    df["popularity"] = pd.to_numeric(df["人気"], errors="coerce").astype("Int64")
    df["tansho_odds"] = pd.to_numeric(df["単勝"], errors="coerce")

    df["weight"] = pd.to_numeric(df["馬体重"].astype(str).str.extract(r"(\d+)")[0], errors="coerce")
    df["weight_diff"] = pd.to_numeric(df["馬体重"].astype(str).str.extract(r"\(([-+0-9]+)\)")[0], errors="coerce")

    df = df.sort_values(["race_id", "umaban"])

    use_cols = [
        "race_id","horse_id","rank","wakuban","umaban","sex","age","impost",
        "jockey_id","popularity","tansho_odds","weight","weight_diff",
        "trainer_id","owner_id",
    ]
    return df[use_cols]
