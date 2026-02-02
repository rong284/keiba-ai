# 旧パス: src/data/loaders/race_info_loader.py
from pathlib import Path
import pandas as pd
from src.utils.progress import tqdm
from src.data_collection.preprocess.race_info import preprocess_race_info_df

def load_race_info(glob_pattern: str = "data/rawdf/race_info/race_info_*.csv") -> pd.DataFrame:
    paths = sorted(Path().glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")

    dfs = []
    for p in tqdm(paths, desc="race info csv", leave=False):
        raw = pd.read_csv(p, sep=",")
        raw.columns = [str(c).strip().lstrip("\ufeff") for c in raw.columns]
        dfs.append(preprocess_race_info_df(raw))

    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["race_id"], keep="last")
    out = out.sort_values(["race_id"]).reset_index(drop=True)
    return out
