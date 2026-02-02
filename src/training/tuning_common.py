from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Tuple

import pandas as pd

from src.training.data import DataPaths, load_train_dataframe
from src.training.features import build_feature_spec
from src.training.split import TimeFold, make_time_folds


def load_config(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_dataframe() -> pd.DataFrame:
    return load_train_dataframe(DataPaths(
        result_glob="data/rawdf/result/result_*.csv",
        horse_glob="data/rawdf/horse/*.csv",
        race_info_glob="data/rawdf/race_info/*.csv",
    ))


def build_spec(df: pd.DataFrame, cfg: Dict):
    return build_feature_spec(df, use_market=bool(cfg["use_market"]))


def make_fold_pairs(df: pd.DataFrame, cfg: Dict):
    folds = [TimeFold(*x) for x in cfg["folds"]]
    return make_time_folds(df, folds, date_col="race_date")


def save_optuna_artifacts(out_dir: Path, best: Dict, trials: List[Dict], prefix: str):
    out_dir.mkdir(parents=True, exist_ok=True)
    best_path = out_dir / f"optuna_best_params_{prefix}.json"
    best_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")
    pd.DataFrame(trials).to_csv(out_dir / f"optuna_trials_{prefix}.csv", index=False, encoding="utf-8")
    return best_path


def load_best_params(path: Path, base_params: Dict) -> Dict:
    if not path.exists():
        return dict(base_params)
    best = json.loads(path.read_text(encoding="utf-8"))
    best = {k: v for k, v in best.items() if not str(k).startswith("_")}
    merged = dict(base_params)
    merged.update(best)
    return merged
