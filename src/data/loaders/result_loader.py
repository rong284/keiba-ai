from pathlib import Path
import pandas as pd
from src.utils.progress import tqdm
from src.data.preprocess.result import preprocess_race_results_df

def load_results(glob_pattern: str = "data/rawdf/result/result_*.csv") -> pd.DataFrame:
    paths = sorted(Path().glob(glob_pattern))
    if not paths:
        raise FileNotFoundError(f"No files matched: {glob_pattern}")

    dfs = []
    for p in tqdm(paths, desc="result csv", leave=False):
        raw = pd.read_csv(p, sep=",")  # ★ここだけ
        raw.columns = [str(c).strip().lstrip("\ufeff") for c in raw.columns]  # 念のため

        df = preprocess_race_results_df(raw)
        dfs.append(df)

    out = pd.concat(dfs, ignore_index=True)
    out = out.drop_duplicates(subset=["race_id", "horse_id"], keep="last")
    out = out.sort_values(["race_id", "umaban"]).reset_index(drop=True)
    return out
