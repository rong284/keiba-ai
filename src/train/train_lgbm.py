from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Tuple

import numpy as np
import pandas as pd

from src.data.loaders.result_loader import load_results
from src.data.loaders.horse_loader import load_horse_results
from src.data.loaders.race_info_loader import load_race_info
from src.data.pipelines.build_train_table import build_train_table


def _default_categorical_cols(df: pd.DataFrame) -> List[str]:
    cols = [
        "place",
        "race_type",
        "around",
        "dist_bin",
        "weather",
        "ground_state",
        "race_class",
        "sex",
        "rest_bin",
        "jockey_id",
        "trainer_id",
        "owner_id",
    ]
    return [c for c in cols if c in df.columns]


def _default_drop_cols(use_market: bool) -> List[str]:
    drop = ["rank", "race_id", "horse_id", "race_date"]
    if not use_market:
        drop += ["popularity", "tansho_odds"]
    return drop


def _coerce_categories(df: pd.DataFrame, cat_cols: List[str]) -> pd.DataFrame:
    out = df.copy()
    for c in cat_cols:
        if c in out.columns:
            out[c] = out[c].astype("category")
    return out


def _make_Xy(
    df: pd.DataFrame,
    target_col: str,
    cat_cols: List[str],
    use_market: bool,
) -> Tuple[pd.DataFrame, pd.Series, List[str]]:
    drop_cols = set(_default_drop_cols(use_market)) | {"y_win", "y_top3"}
    drop_cols = [c for c in drop_cols if c in df.columns]

    X = df.drop(columns=drop_cols + [target_col], errors="ignore")
    y = df[target_col]

    cat_cols = [c for c in cat_cols if c in X.columns]
    X = _coerce_categories(X, cat_cols)
    return X, y, cat_cols


def _split_by_year(df: pd.DataFrame, valid_year: int | None) -> Tuple[pd.DataFrame, pd.DataFrame | None]:
    if valid_year is None:
        return df.copy(), None

    year = df["race_date"].dt.year
    train_df = df[year < valid_year].copy()
    valid_df = df[year == valid_year].copy()
    return train_df, valid_df


def _save_json(path: Path, payload: Dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)


def train_binary(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None,
    target: str,
    cat_cols: List[str],
    use_market: bool,
    params: Dict,
) -> Tuple["lgb.Booster", Dict]:
    import lightgbm as lgb
    from sklearn.metrics import roc_auc_score, average_precision_score

    params = dict(params)
    num_boost_round = int(params.pop("num_boost_round", 2000))

    X_tr, y_tr, cat_cols = _make_Xy(train_df, target, cat_cols, use_market)
    dtr = lgb.Dataset(X_tr, label=y_tr, categorical_feature=cat_cols, free_raw_data=False)

    valid_sets = [dtr]
    valid_names = ["train"]
    if valid_df is not None and len(valid_df) > 0:
        X_va, y_va, _ = _make_Xy(valid_df, target, cat_cols, use_market)
        dva = lgb.Dataset(X_va, label=y_va, categorical_feature=cat_cols, free_raw_data=False)
        valid_sets.append(dva)
        valid_names.append("valid")

    callbacks = [lgb.log_evaluation(200)]
    if valid_df is not None and len(valid_df) > 0:
        callbacks.insert(0, lgb.early_stopping(200))

    model = lgb.train(
        params,
        dtr,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    metrics = {"best_iteration": model.best_iteration}
    if valid_df is not None and len(valid_df) > 0:
        pred = model.predict(X_va, num_iteration=model.best_iteration)
        metrics.update(
            {
                "auc": float(roc_auc_score(y_va, pred)),
                "ap": float(average_precision_score(y_va, pred)),
                "pos_rate_valid": float(y_va.mean()),
            }
        )

    return model, metrics


def _prepare_rank_df(df: pd.DataFrame) -> pd.DataFrame:
    # LightGBM ranking uses group boundaries; keep race_id order stable.
    return df.sort_values(["race_id", "umaban"], kind="mergesort").reset_index(drop=True)


def _rank_groups(df: pd.DataFrame) -> List[int]:
    return df.groupby("race_id", sort=False).size().to_list()


def train_rank(
    train_df: pd.DataFrame,
    valid_df: pd.DataFrame | None,
    cat_cols: List[str],
    use_market: bool,
    params: Dict,
) -> Tuple["lgb.Booster", Dict]:
    import lightgbm as lgb

    params = dict(params)
    num_boost_round = int(params.pop("num_boost_round", 2000))

    tr = _prepare_rank_df(train_df)
    va = _prepare_rank_df(valid_df) if valid_df is not None else None

    X_tr, y_tr, cat_cols = _make_Xy(tr, "rank", cat_cols, use_market)
    # Higher relevance is better; rank=1 should be the highest.
    y_tr = -y_tr.astype(float)

    dtr = lgb.Dataset(
        X_tr,
        label=y_tr,
        group=_rank_groups(tr),
        categorical_feature=cat_cols,
        free_raw_data=False,
    )

    valid_sets = [dtr]
    valid_names = ["train"]
    if va is not None and len(va) > 0:
        X_va, y_va, _ = _make_Xy(va, "rank", cat_cols, use_market)
        y_va = -y_va.astype(float)
        dva = lgb.Dataset(
            X_va,
            label=y_va,
            group=_rank_groups(va),
            categorical_feature=cat_cols,
            free_raw_data=False,
        )
        valid_sets.append(dva)
        valid_names.append("valid")

    callbacks = [lgb.log_evaluation(200)]
    if va is not None and len(va) > 0:
        callbacks.insert(0, lgb.early_stopping(200))

    model = lgb.train(
        params,
        dtr,
        num_boost_round=num_boost_round,
        valid_sets=valid_sets,
        valid_names=valid_names,
        callbacks=callbacks,
    )

    metrics = {"best_iteration": model.best_iteration}
    return model, metrics


def _base_params_binary(seed: int) -> Dict:
    return {
        "objective": "binary",
        "metric": ["auc", "average_precision"],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": seed,
        "verbosity": -1,
    }


def _base_params_rank(seed: int) -> Dict:
    return {
        "objective": "lambdarank",
        "metric": ["ndcg"],
        "ndcg_at": [1, 3],
        "learning_rate": 0.05,
        "num_leaves": 63,
        "min_data_in_leaf": 50,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 1,
        "seed": seed,
        "verbosity": -1,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--valid-year", type=int, default=2025)
    parser.add_argument("--use-market", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--out-dir", type=str, default="outputs/03_train")
    parser.add_argument("--params-binary", type=str, default="")
    parser.add_argument("--params-rank", type=str, default="")
    parser.add_argument("--train-final", action="store_true")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    model_dir = out_dir / "models"
    table_dir = out_dir / "tables"
    model_dir.mkdir(parents=True, exist_ok=True)
    table_dir.mkdir(parents=True, exist_ok=True)

    df_result = load_results("data/rawdf/result/result_*.csv")
    df_horse = load_horse_results("data/rawdf/horse/*.csv")
    df_race_info = load_race_info("data/rawdf/race_info/*.csv")

    df = build_train_table(df_result, df_race_info, df_horse)
    df = df.dropna(subset=["rank", "race_date"]).reset_index(drop=True)

    df["y_win"] = (df["rank"] == 1).astype(int)
    df["y_top3"] = (df["rank"] <= 3).astype(int)

    cat_cols = _default_categorical_cols(df)
    train_df, valid_df = _split_by_year(df, args.valid_year)

    params_binary = _base_params_binary(args.seed)
    params_rank = _base_params_rank(args.seed)

    if args.params_binary:
        params_binary.update(json.loads(Path(args.params_binary).read_text()))
    if args.params_rank:
        params_rank.update(json.loads(Path(args.params_rank).read_text()))

    metrics = {}

    model_win, m_win = train_binary(
        train_df,
        valid_df,
        target="y_win",
        cat_cols=cat_cols,
        use_market=args.use_market,
        params=params_binary,
    )
    model_win.save_model(str(model_dir / f"lgb_y_win__split_{args.valid_year}.txt"))
    metrics["y_win"] = m_win

    model_top3, m_top3 = train_binary(
        train_df,
        valid_df,
        target="y_top3",
        cat_cols=cat_cols,
        use_market=args.use_market,
        params=params_binary,
    )
    model_top3.save_model(str(model_dir / f"lgb_y_top3__split_{args.valid_year}.txt"))
    metrics["y_top3"] = m_top3

    model_rank, m_rank = train_rank(
        train_df,
        valid_df,
        cat_cols=cat_cols,
        use_market=args.use_market,
        params=params_rank,
    )
    model_rank.save_model(str(model_dir / f"lgb_lambdarank__split_{args.valid_year}.txt"))
    metrics["lambdarank"] = m_rank

    _save_json(table_dir / f"metrics_split_{args.valid_year}.json", metrics)

    if args.train_final:
        # Train final models on all data using best_iteration from the tuned split.
        final_metrics = {}

        params_win_final = dict(params_binary)
        if m_win.get("best_iteration"):
            params_win_final["num_boost_round"] = m_win["best_iteration"]
        model_win_full, _ = train_binary(
            df,
            None,
            target="y_win",
            cat_cols=cat_cols,
            use_market=args.use_market,
            params=params_win_final,
        )
        model_win_full.save_model(str(model_dir / "lgb_y_win__full_2021_2025.txt"))
        final_metrics["y_win"] = {"num_boost_round": params_win_final.get("num_boost_round")}

        params_top3_final = dict(params_binary)
        if m_top3.get("best_iteration"):
            params_top3_final["num_boost_round"] = m_top3["best_iteration"]
        model_top3_full, _ = train_binary(
            df,
            None,
            target="y_top3",
            cat_cols=cat_cols,
            use_market=args.use_market,
            params=params_top3_final,
        )
        model_top3_full.save_model(str(model_dir / "lgb_y_top3__full_2021_2025.txt"))
        final_metrics["y_top3"] = {"num_boost_round": params_top3_final.get("num_boost_round")}

        params_rank_final = dict(params_rank)
        if m_rank.get("best_iteration"):
            params_rank_final["num_boost_round"] = m_rank["best_iteration"]
        model_rank_full, _ = train_rank(
            df,
            None,
            cat_cols=cat_cols,
            use_market=args.use_market,
            params=params_rank_final,
        )
        model_rank_full.save_model(str(model_dir / "lgb_lambdarank__full_2021_2025.txt"))
        final_metrics["lambdarank"] = {"num_boost_round": params_rank_final.get("num_boost_round")}

        _save_json(table_dir / "metrics_full_2021_2025.json", final_metrics)


if __name__ == "__main__":
    main()
