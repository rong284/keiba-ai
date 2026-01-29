# src/data/loaders/horse_loader.py
from pathlib import Path
import pandas as pd
from src.data.preprocess.horse import preprocess_horse_results_df

def load_horse_results(glob_pattern: str = "data/rawdf/horse/horse_*.csv") -> pd.DataFrame:
    paths = sorted(Path().glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")

    dfs = []
    for p in paths:
        raw = pd.read_csv(p, sep=",")
        raw.columns = [str(c).strip().lstrip("\ufeff") for c in raw.columns]
        dfs.append(preprocess_horse_results_df(raw))

    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["horse_id", "date"], keep="last")
    out = out.sort_values(["horse_id", "date"]).reset_index(drop=True)
    return out
