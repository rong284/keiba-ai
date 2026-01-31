from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict

import numpy as np
import optuna
import pandas as pd

from src.train.data import DataPaths, load_train_dataframe
from src.train.split import TimeFold, make_time_folds
from src.train.features import build_feature_spec
from src.train.models.lgb_rank import train_rank, predict_rank, make_relevance_from_rank, sort_by_group
from src.train.metrics import ndcg_at_k


def load_config(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def objective_factory(df, fold_pairs, spec, cfg):
    seed = int(cfg["random_seed"])
    base = dict(cfg["lgb_params"])

    def objective(trial: optuna.Trial) -> float:
        params = dict(base)

        # まずは効きが大きいところから（軽量）
        params.update({
            "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.08, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 31, 255),
            "min_data_in_leaf": trial.suggest_int("min_data_in_leaf", 20, 200),
            "feature_fraction": trial.suggest_float("feature_fraction", 0.6, 1.0),
            "bagging_fraction": trial.suggest_float("bagging_fraction", 0.6, 1.0),
            "bagging_freq": trial.suggest_int("bagging_freq", 0, 5),
            "lambda_l1": trial.suggest_float("lambda_l1", 0.0, 5.0),
            "lambda_l2": trial.suggest_float("lambda_l2", 0.0, 10.0),
        })

        scores = []
        for tr_df, va_df, _fold in fold_pairs:
            # group順で揃える（rankingは順序依存）
            tr_s = sort_by_group(tr_df, "race_id")
            va_s = sort_by_group(va_df, "race_id")

            res = train_rank(
                tr_df=tr_s,
                va_df=va_s,
                feature_cols=spec.feature_cols,
                cat_cols=spec.cat_cols,
                params=params,
                eval_at=list(cfg["eval_at"]),
                seed=seed,
                num_boost_round=5000,
                early_stopping_rounds=200,
            )

            # 予測 -> ndcg@3 を評価指標に採用（上位品質）
            pred = predict_rank(res.model, tr_s, va_s, spec.feature_cols, spec.cat_cols)
            rel = make_relevance_from_rank(va_s)
            s = ndcg_at_k(va_s, pred, rel, 3, seed=seed)
            scores.append(s)

        return float(np.mean(scores))

    return objective


def main(
    config_path: str = "src/configs/train_rank.json",
    out_dir: str = "outputs/03_train/rank/tables",
    n_trials: int = 40,
):
    cfg = load_config(config_path)
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ---- データ読み込み（学習用テーブル） ----
    df = load_train_dataframe(DataPaths(
        result_glob="data/rawdf/result/result_*.csv",
        horse_glob="data/rawdf/horse/*.csv",
        race_info_glob="data/rawdf/race_info/*.csv",
    ))

    # ---- 特徴量仕様（CVと同じ） ----
    spec = build_feature_spec(df, use_market=bool(cfg["use_market"]))

    # ---- CV folds（cfg["folds"] に従う：=2024まで） ----
    folds = [TimeFold(*x) for x in cfg["folds"]]
    fold_pairs = make_time_folds(df, folds, date_col="race_date")

    # ---- Optuna ----
    seed = int(cfg["random_seed"])
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    objective = objective_factory(df, fold_pairs, spec, cfg)
    study.optimize(objective, n_trials=n_trials)

    # ---- 保存 ----
    best = dict(study.best_params)
    best["_best_value_mean_ndcg_at_3"] = float(study.best_value)
    best["_n_trials"] = n_trials

    out_path = out_dir / "optuna_best_params_rank.json"
    out_path.write_text(json.dumps(best, ensure_ascii=False, indent=2), encoding="utf-8")

    # ついでに全試行のログもCSV保存（後で分析しやすい）
    trials = []
    for t in study.trials:
        row = dict(t.params)
        row["value"] = t.value
        row["state"] = str(t.state)
        trials.append(row)
    pd.DataFrame(trials).to_csv(out_dir / "optuna_trials_rank.csv", index=False, encoding="utf-8")

    print("[best_value mean(ndcg@3)]", study.best_value)
    print("[saved]", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/configs/train_rank.json")
    ap.add_argument("--out_dir", default="outputs/03_train/rank/tables")
    ap.add_argument("--n_trials", type=int, default=40)
    args = ap.parse_args()

    main(
        config_path=args.config,
        out_dir=args.out_dir,
        n_trials=args.n_trials,
    )
