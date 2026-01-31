from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List

import pandas as pd
from tqdm.auto import tqdm

from src.train.data import DataPaths, load_train_dataframe
from src.train.split import TimeFold, make_time_folds, make_holdout
from src.train.features import build_feature_spec
from src.train.metrics import hit_at_k, top1_pos_rate
from src.train.models.lgb_binary import train_binary, predict_binary
from src.train.calibrate import fit_calibrators, choose_best_calibrator, apply_best
from src.train.metrics import race_equal_weights
from src.train.artifacts import OutputDirs, save_json, save_table_csv, save_lgb_model


TARGET_MAP = {
    "win": "y_top1",
    "place": "y_top3",
}


def load_config(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(
    config_path: str = "src/configs/train_binary.json",
    out_dir: str = "outputs/03_train/binary",
):
    cfg = load_config(config_path)
    out = OutputDirs.create(Path(out_dir))

    # ---- データ読み込み ----
    df = load_train_dataframe(DataPaths(
        result_glob="data/rawdf/result/result_*.csv",
        horse_glob="data/rawdf/horse/*.csv",
        race_info_glob="data/rawdf/race_info/*.csv",
    ))

    # ---- 特徴量 ----
    spec = build_feature_spec(df, use_market=bool(cfg["use_market"]))
    save_json(out.tab_dir / "feature_spec.json", {
        "feature_cols": spec.feature_cols,
        "cat_cols": spec.cat_cols,
        "dropped_cols": spec.dropped_cols,
    })

    # ---- CV（2024まで） ----
    folds = [TimeFold(*x) for x in cfg["folds"]]
    fold_pairs = make_time_folds(df, folds, date_col="race_date")

    rows = []

    for target_name in tqdm(cfg["target_list"], desc="targets (cv)"):
        y_col = TARGET_MAP[target_name]
        for tr_df, va_df, f in tqdm(fold_pairs, desc=f"cv folds ({target_name})", leave=False):
            res = train_binary(
                tr_df=tr_df,
                va_df=va_df,
                y_col=y_col,
                feature_cols=spec.feature_cols,
                cat_cols=spec.cat_cols,
                params=cfg["lgb_params"],
                seed=int(cfg["random_seed"]),
            )
            pred = predict_binary(res.model, tr_df, va_df, spec.feature_cols, spec.cat_cols)
            y_va = va_df[y_col].values.astype(int)

            row = {
                "target": target_name,
                "train_end": f.train_end,
                "valid_start": f.valid_start,
                "valid_end": f.valid_end,
                "best_iter": res.best_iter,
                "scale_pos_weight": res.scale_pos_weight,
                "n_train_races": int(tr_df["race_id"].nunique()),
                "n_valid_races": int(va_df["race_id"].nunique()),
                "top1_pos_rate": top1_pos_rate(va_df, pred, y_va, seed=int(cfg["random_seed"])),
                "hit_at_1": hit_at_k(va_df, pred, y_va, 1, seed=int(cfg["random_seed"])),
                "hit_at_3": hit_at_k(va_df, pred, y_va, 3, seed=int(cfg["random_seed"])),
                "hit_at_5": hit_at_k(va_df, pred, y_va, 5, seed=int(cfg["random_seed"])),
            }
            rows.append(row)

    cv_df = pd.DataFrame(rows)
    save_table_csv(out.tab_dir / "cv_binary.csv", cv_df)

    # ★追加：targetごとに best_iter を決めて保存（中央値が安定）
    best_iter_map = (
        cv_df.groupby("target")["best_iter"]
        .median()
        .round()
        .astype(int)
        .to_dict()
    )
    save_json(out.tab_dir / "best_iter_from_cv.json", best_iter_map)

    # ついでに見やすい集計も保存
    cv_sum = (cv_df.groupby("target")[["best_iter","hit_at_1","hit_at_3","hit_at_5"]]
            .agg(["mean","std","median"])
            .reset_index())
    save_table_csv(out.tab_dir / "cv_summary_binary.csv", cv_sum)

    # ---- 最終ホールドアウト（2025）で 学習→校正→評価 ----
    train_A, test2025 = make_holdout(df, cfg["train_end"], cfg["test_start"], date_col="race_date")

    # ★Action4：2025を3分割（校正fit / 校正選択 / 最終評価）
    calib_fit_end = cfg.get("calib_fit_end", "2025-03-31")
    calib_sel_end = cfg.get("calib_sel_end", "2025-06-30")

    calib_fit = test2025[test2025["race_date"] <= pd.Timestamp(calib_fit_end)].copy()
    calib_sel = test2025[(test2025["race_date"] > pd.Timestamp(calib_fit_end)) &
                        (test2025["race_date"] <= pd.Timestamp(calib_sel_end))].copy()
    eval_2025  = test2025[test2025["race_date"] > pd.Timestamp(calib_sel_end)].copy()

    hold_rows = []

    # ★CVからbest_iter（木の本数）を取得
    best_iter_map = json.loads((out.tab_dir / "best_iter_from_cv.json").read_text(encoding="utf-8"))

    for target_name in tqdm(cfg["target_list"], desc="targets (holdout)"):
        y_col = TARGET_MAP[target_name]

        # ★最終学習：2024までのみ、木の本数はCV best_iterで固定（2025は絶対見ない）
        final_rounds = int(best_iter_map[target_name])

        res = train_binary(
            tr_df=train_A,
            va_df=None,  # 2025は使わない（評価の信頼性）
            y_col=y_col,
            feature_cols=spec.feature_cols,
            cat_cols=spec.cat_cols,
            params=cfg["lgb_params"],
            seed=int(cfg["random_seed"]),
            num_boost_round=final_rounds,   # ★ここが重要
            early_stopping_rounds=None,
        )
        save_lgb_model(out.model_dir / f"lgb_binary_{target_name}.txt", res.model)

        # --- 予測（校正fit / 校正選択 / 最終評価） ---
        p_fit = predict_binary(res.model, train_A, calib_fit, spec.feature_cols, spec.cat_cols)
        y_fit = calib_fit[y_col].values.astype(int)

        p_sel = predict_binary(res.model, train_A, calib_sel, spec.feature_cols, spec.cat_cols)
        y_sel = calib_sel[y_col].values.astype(int)

        p_ev = predict_binary(res.model, train_A, eval_2025, spec.feature_cols, spec.cat_cols)
        y_ev = eval_2025[y_col].values.astype(int)

        # --- 校正fit（レース均等重み） ---
        w_fit = race_equal_weights(calib_fit, "race_id")
        calib = fit_calibrators(p_fit, y_fit, w_fit)

        # ★校正法の選択は calib_sel でやる（eval_2025を見て選ばない＝リーク防止）
        best_kind, best_rep_sel = choose_best_calibrator(y_sel, p_sel, calib)

        # --- 最終評価（eval_2025） ---
        p_best = apply_best(best_kind, p_ev, calib)

        hold_row = {
            "target": target_name,
            "final_num_boost_round": final_rounds,
            "best_calib": best_kind,

            # 「どれを選んだか」は calib_sel 上の指標として保存
            **{f"sel_{k}": v for k, v in best_rep_sel.items()},

            # 最終評価は eval_2025 上で出す（ここは触らない）
            "top1_pos_rate": top1_pos_rate(eval_2025, p_best, y_ev, seed=int(cfg["random_seed"])),
            "hit_at_1": hit_at_k(eval_2025, p_best, y_ev, 1, seed=int(cfg["random_seed"])),
            "hit_at_3": hit_at_k(eval_2025, p_best, y_ev, 3, seed=int(cfg["random_seed"])),
            "hit_at_5": hit_at_k(eval_2025, p_best, y_ev, 5, seed=int(cfg["random_seed"])),
            "n_eval_races": int(eval_2025["race_id"].nunique()),
            "calib_fit_end": calib_fit_end,
            "calib_sel_end": calib_sel_end,
        }
        hold_rows.append(hold_row)

        # ついで：将来の回収率計算のために予測を保存（払戻が来たら即結合できる）
        pred_out = eval_2025[["race_id","horse_id","race_date"]].copy()
        pred_out["target"] = target_name
        pred_out["p_raw"] = p_ev
        pred_out["p_calib"] = p_best
        save_table_csv(out.tab_dir / f"preds_{target_name}_2025_eval.csv", pred_out)

    hold_df = pd.DataFrame(hold_rows)
    save_table_csv(out.tab_dir / "holdout_binary_2025.csv", hold_df)

    print("[done] outputs/03_train に保存しました")


if __name__ == "__main__":
    main()
