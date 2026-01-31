from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm.auto import tqdm

from src.train.models.lgb_rank import sort_by_group
from src.train.data import DataPaths, load_train_dataframe
from src.train.split import TimeFold, make_time_folds, make_holdout
from src.train.features import build_feature_spec
from src.train.metrics import mrr_for_winner, ndcg_at_k, hit_at_k
from src.train.models.lgb_rank import train_rank, predict_rank, make_relevance_from_rank
from src.train.artifacts import OutputDirs, save_json, save_table_csv, save_lgb_model


def load_config(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(
    config_path: str = "src/configs/train_rank.json",
    out_dir: str = "outputs/03_train/rank",
):
    cfg = load_config(config_path)
    out = OutputDirs.create(Path(out_dir))

    df = load_train_dataframe(DataPaths(
        result_glob="data/rawdf/result/result_*.csv",
        horse_glob="data/rawdf/horse/*.csv",
        race_info_glob="data/rawdf/race_info/*.csv",
    ))

    spec = build_feature_spec(df, use_market=bool(cfg["use_market"]))
    save_json(out.tab_dir / "feature_spec_rank.json", {
        "feature_cols": spec.feature_cols,
        "cat_cols": spec.cat_cols,
        "dropped_cols": spec.dropped_cols,
    })

    # ---- CV（2024まで） ----
    # 2025は一切見ない。best_iter は CV のみから決める。
    folds = [TimeFold(*x) for x in cfg["folds"]]
    fold_pairs = make_time_folds(df, folds, date_col="race_date")

    rows = []
    for tr_df, va_df, f in tqdm(fold_pairs, desc="cv folds (rank)"):
        # --- CV loop の中 ---
        tr_s = sort_by_group(tr_df, "race_id")
        va_s = sort_by_group(va_df, "race_id")

        # ---- LambdaRank CV ----
        # early stopping は ndcg@3 を主指標として使いたいので、
        # eval_at を [3,5] の順で渡し、first_metric_only=True で止める。
        res = train_rank(
            tr_df=tr_s,
            va_df=va_s,
            feature_cols=spec.feature_cols,
            cat_cols=spec.cat_cols,
            params=cfg["lgb_params"],
            eval_at=list(cfg["eval_at"]),
            seed=int(cfg["random_seed"]),
            num_boost_round=5000,
            early_stopping_rounds=200,
        )

        # ここで返る pred は va_s と同じ順番になっている必要がある
        pred = predict_rank(res.model, tr_s, va_s, spec.feature_cols, spec.cat_cols)

        # 評価は必ず va_s と pred でやる
        rel = make_relevance_from_rank(va_s)
        row = {
            "train_end": f.train_end,
            "valid_start": f.valid_start,
            "valid_end": f.valid_end,
            "best_iter": res.best_iter,
            "n_valid_races": int(va_s["race_id"].nunique()),
            # 評価は va_s と pred の組み合わせで統一（順序ズレ防止）
            "mrr_winner": mrr_for_winner(va_s, pred, va_s["rank"].values, seed=int(cfg["random_seed"])),
            "ndcg@3": ndcg_at_k(va_s, pred, rel, 3, seed=int(cfg["random_seed"])),
            "ndcg@5": ndcg_at_k(va_s, pred, rel, 5, seed=int(cfg["random_seed"])),
            "hit_at_1": hit_at_k(va_s, pred, (va_s["rank"].values == 1).astype(int), 1, seed=int(cfg["random_seed"])),
            "hit_at_3": hit_at_k(va_s, pred, (va_s["rank"].values == 1).astype(int), 3, seed=int(cfg["random_seed"])),
            "hit_at_5": hit_at_k(va_s, pred, (va_s["rank"].values == 1).astype(int), 5, seed=int(cfg["random_seed"])),
        }
        rows.append(row)

    cv_df = pd.DataFrame(rows)
    save_table_csv(out.tab_dir / "cv_rank.csv", cv_df)

    # ---- best_iter をCVから決めて保存（中央値が安定） ----
    best_iter = int(cv_df["best_iter"].median().round())
    save_json(out.tab_dir / "best_iter_from_cv.json", {"lambdarank": best_iter})

    # ついでに集計表も保存（見やすさ用）
    cv_sum = (cv_df[["best_iter", "mrr_winner", "ndcg@3", "ndcg@5", "hit_at_1", "hit_at_3", "hit_at_5"]]
              .agg(["mean", "std", "median"])
              .reset_index())
    save_table_csv(out.tab_dir / "cv_summary_rank.csv", cv_sum)

    # ---- 最終ホールドアウト（2025） ----
    train_A, test2025 = make_holdout(df, cfg["train_end"], cfg["test_start"], date_col="race_date")

    # ★CVで決めた best_iter を固定して最終学習（2025は使わない）
    best_iter = int(json.loads((out.tab_dir / "best_iter_from_cv.json").read_text(encoding="utf-8"))["lambdarank"])

    res = train_rank(
        tr_df=train_A,
        va_df=None,
        feature_cols=spec.feature_cols,
        cat_cols=spec.cat_cols,
        params=cfg["lgb_params"],
        eval_at=list(cfg["eval_at"]),
        seed=int(cfg["random_seed"]),
        num_boost_round=best_iter,
        early_stopping_rounds=None,
    )
    save_lgb_model(out.model_dir / "lgb_rank_lambdarank.txt", res.model)

    pred_2025 = predict_rank(res.model, train_A, test2025, spec.feature_cols, spec.cat_cols)
    rel_2025 = make_relevance_from_rank(test2025)

    hold = {
        "n_races": int(test2025["race_id"].nunique()),
        "mrr_winner": mrr_for_winner(test2025, pred_2025, test2025["rank"].values, seed=int(cfg["random_seed"])),
        "ndcg@3": ndcg_at_k(test2025, pred_2025, rel_2025, 3, seed=int(cfg["random_seed"])),
        "ndcg@5": ndcg_at_k(test2025, pred_2025, rel_2025, 5, seed=int(cfg["random_seed"])),
        "hit_at_1": hit_at_k(test2025, pred_2025, (test2025["rank"].values == 1).astype(int), 1, seed=int(cfg["random_seed"])),
        "hit_at_3": hit_at_k(test2025, pred_2025, (test2025["rank"].values == 1).astype(int), 3, seed=int(cfg["random_seed"])),
        "hit_at_5": hit_at_k(test2025, pred_2025, (test2025["rank"].values == 1).astype(int), 5, seed=int(cfg["random_seed"])),
        "final_num_boost_round": best_iter,
    }
    save_json(out.tab_dir / "holdout_rank_2025.json", hold)

    print("[done] outputs/03_train に保存しました")


if __name__ == "__main__":
    main()
