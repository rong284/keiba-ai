from __future__ import annotations

from pathlib import Path
import argparse
from typing import Dict

import numpy as np
import optuna

from src.training.models.lgb_binary import train_binary, predict_binary
from src.training.metrics import hit_at_k
from src.training.tuning_common import (
    build_spec,
    load_config,
    load_dataframe,
    make_fold_pairs,
    save_optuna_artifacts,
)

TARGET_MAP = {"win": "y_top1", "place": "y_top3"}


def objective_factory(df, fold_pairs, spec, cfg, target_name: str):
    y_col = TARGET_MAP[target_name]
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
            res = train_binary(
                tr_df=tr_df,
                va_df=va_df,  # CVなのでvalidあり
                y_col=y_col,
                feature_cols=spec.feature_cols,
                cat_cols=spec.cat_cols,
                params=params,
                seed=seed,
                num_boost_round=5000,
                early_stopping_rounds=200,
            )
            pred = predict_binary(res.model, tr_df, va_df, spec.feature_cols, spec.cat_cols)
            y_va = va_df[y_col].values.astype(int)

            # 目的：レース内Top1命中率を最大化
            s = hit_at_k(va_df, pred, y_va, k=1, seed=seed)
            scores.append(s)

        return float(np.mean(scores))

    return objective


def main(
    config_path: str = "src/configs/train_binary.json",
    out_dir: str = "outputs/03_train/binary/tables",
    target: str = "win",
    n_trials: int = 40,
):
    cfg = load_config(config_path)
    out_dir = Path(out_dir)

    df = load_dataframe()
    spec = build_spec(df, cfg)
    fold_pairs = make_fold_pairs(df, cfg)

    # ---- Optuna ----
    seed = int(cfg["random_seed"])
    sampler = optuna.samplers.TPESampler(seed=seed)
    study = optuna.create_study(direction="maximize", sampler=sampler)

    objective = objective_factory(df, fold_pairs, spec, cfg, target_name=target)
    study.optimize(objective, n_trials=n_trials)

    # ---- 保存 ----
    best = dict(study.best_params)
    best["_best_value_mean_hit_at_1"] = float(study.best_value)
    best["_target"] = target
    best["_n_trials"] = n_trials

    # ついでに全試行のログもCSV保存（後で分析しやすい）
    trials = []
    for t in study.trials:
        row = dict(t.params)
        row["value"] = t.value
        row["state"] = str(t.state)
        trials.append(row)
    out_path = save_optuna_artifacts(out_dir, best, trials, prefix=target)

    print("[best_value mean(hit@1)]", study.best_value)
    print("[saved]", out_path)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/configs/train_binary.json")
    ap.add_argument("--out_dir", default="outputs/03_train/binary/tables")
    ap.add_argument("--target", default="win", choices=["win", "place"])
    ap.add_argument("--n_trials", type=int, default=40)
    args = ap.parse_args()

    main(
        config_path=args.config,
        out_dir=args.out_dir,
        target=args.target,
        n_trials=args.n_trials,
    )
