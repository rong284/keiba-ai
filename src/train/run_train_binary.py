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

    # ---- 最終ホールドアウト（2025）で 学習→校正→評価 ----
    train_A, test2025 = make_holdout(df, cfg["train_end"], cfg["test_start"], date_col="race_date")

    calib_end = cfg["calib_end"]
    calib_2025 = test2025[test2025["race_date"] <= pd.Timestamp(calib_end)].copy()
    eval_2025 = test2025[test2025["race_date"] > pd.Timestamp(calib_end)].copy()

    hold_rows = []

    for target_name in tqdm(cfg["target_list"], desc="targets (holdout)"):
        y_col = TARGET_MAP[target_name]

        # 2024までで最終学習
        res = train_binary(
            tr_df=train_A,
            va_df=None,
            y_col=y_col,
            feature_cols=spec.feature_cols,
            cat_cols=spec.cat_cols,
            params=cfg["lgb_params"],
            seed=int(cfg["random_seed"]),
        )
        save_lgb_model(out.model_dir / f"lgb_binary_{target_name}.txt", res.model)

        # 2025予測（校正用/評価用）
        p_cal = predict_binary(res.model, train_A, calib_2025, spec.feature_cols, spec.cat_cols)
        p_ev = predict_binary(res.model, train_A, eval_2025, spec.feature_cols, spec.cat_cols)
        y_cal = calib_2025[y_col].values.astype(int)
        y_ev = eval_2025[y_col].values.astype(int)

        # 校正（レース均等の重み）
        w_cal = race_equal_weights(calib_2025, "race_id")
        calib = fit_calibrators(p_cal, y_cal, w_cal)

        best_kind, best_rep = choose_best_calibrator(y_ev, p_ev, calib)
        p_best = apply_best(best_kind, p_ev, calib)

        # 当たりやすさ（順位が壊れてないか）
        hold_row = {
            "target": target_name,
            "best_calib": best_kind,
            **best_rep,
            "top1_pos_rate": top1_pos_rate(eval_2025, p_best, y_ev, seed=int(cfg["random_seed"])),
            "hit_at_1": hit_at_k(eval_2025, p_best, y_ev, 1, seed=int(cfg["random_seed"])),
            "hit_at_3": hit_at_k(eval_2025, p_best, y_ev, 3, seed=int(cfg["random_seed"])),
            "hit_at_5": hit_at_k(eval_2025, p_best, y_ev, 5, seed=int(cfg["random_seed"])),
            "n_eval_races": int(eval_2025["race_id"].nunique()),
        }
        hold_rows.append(hold_row)

    hold_df = pd.DataFrame(hold_rows)
    save_table_csv(out.tab_dir / "holdout_binary_2025.csv", hold_df)

    print("[done] outputs/03_train に保存しました")


if __name__ == "__main__":
    main()
