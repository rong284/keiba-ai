from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
from src.utils.progress import tqdm

from src.train.models.lgb_rank import sort_by_group
from src.train.data import DataPaths, load_train_dataframe
from src.train.split import TimeFold, make_time_folds, make_holdout, make_train_only, clip_test_end
from src.train.date_plan import resolve_date_plan
from src.train.reporting import (
    plot_cv_best_iter,
    plot_cv_metrics_rank,
    plot_holdout_rank,
    save_date_plan,
    write_report,
)
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
    plan = resolve_date_plan(cfg)
    save_date_plan(out.tab_dir, plan.as_dict())

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
    for tr_df, va_df, f in tqdm(fold_pairs, desc="cv folds (rank)", position=0):
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
    if plan.eval_enabled and plan.test_start:
        train_A, test2025 = make_holdout(df, plan.train_end, plan.test_start, date_col="race_date")
        test2025 = clip_test_end(test2025, plan.test_end, date_col="race_date")
    else:
        train_A = make_train_only(df, plan.train_end, date_col="race_date")
        test2025 = None

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

    hold = None
    if test2025 is not None and len(test2025) > 0:
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
        plot_holdout_rank(hold, out.fig_dir / "holdout_rank_2025.png")

        pred_out = test2025[["race_id", "horse_id", "umaban", "race_date"]].copy()
        pred_out["score"] = pred_2025
        save_table_csv(out.tab_dir / "preds_rank_2025_eval.csv", pred_out)

    plot_cv_best_iter(cv_df, out.fig_dir / "cv_best_iter_rank.png", "CV best_iter (rank)")
    plot_cv_metrics_rank(cv_df, out.fig_dir / "cv_metrics_rank.png")

    report_lines = [
        "# Rank training report",
        f"mode: {plan.mode}",
        f"train_end: {plan.train_end}",
        f"test_start: {plan.test_start}",
        f"test_end: {plan.test_end}",
        f"eval_enabled: {plan.eval_enabled}",
        "",
        f"CV summary: {out.tab_dir / 'cv_summary_rank.csv'}",
    ]
    if hold:
        report_lines += [
            f"Holdout summary: {out.tab_dir / 'holdout_rank_2025.json'}",
            "",
            json.dumps(hold, ensure_ascii=False, indent=2),
        ]
    write_report(out.out_dir / "report_rank.md", report_lines)

    print("[done] outputs/03_train に保存しました")


if __name__ == "__main__":
    main()
