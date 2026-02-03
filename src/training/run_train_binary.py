from __future__ import annotations

import json
from pathlib import Path
from typing import Dict

import pandas as pd
from src.utils.progress import tqdm

from src.training.data import DataPaths, load_train_dataframe
from src.training.split import TimeFold, make_time_folds, make_holdout, make_train_only, clip_test_end
from src.training.date_plan import resolve_date_plan
from src.training.reporting import (
    plot_cv_best_iter,
    plot_holdout_binary,
    save_date_plan,
    write_report,
)
from src.training.tuning_common import load_best_params
from src.training.features import build_feature_spec
from src.training.metrics import hit_at_k, top1_pos_rate
from src.training.models.lgb_binary import train_binary, predict_binary
from src.training.artifacts import OutputDirs, save_json, save_table_csv, save_lgb_model
from src.training.odds import odds_base_margin
from src.training.feature_analysis import build_feature_summary, save_feature_importance


TARGET_MAP = {
    "win": "y_top1",
    "place": "y_top3",
}


def load_config(path: str) -> Dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def main(
    config_path: str = "src/configs/train_binary.json",
    out_dir: str = "outputs/train/binary",
):
    cfg = load_config(config_path)
    mode = cfg.get("mode", "standard")
    odds_col = cfg.get("odds_col", cfg.get("win_odds_col", "tansho_odds"))
    out = OutputDirs.create(Path(out_dir))
    plan = resolve_date_plan(cfg)
    save_date_plan(out.tab_dir, plan.as_dict())

    # ---- データ読み込み ----
    df = load_train_dataframe(DataPaths(
        result_glob="data/rawdf/result/result_*.csv",
        horse_glob="data/rawdf/horse/*.csv",
        race_info_glob="data/rawdf/race_info/*.csv",
    ))

    if mode == "residual" and odds_col not in df.columns:
        raise ValueError(f"odds_col not found for residual mode: {odds_col}")

    # ---- 特徴量 ----
    spec = build_feature_spec(df, use_market=bool(cfg["use_market"]))
    save_json(out.tab_dir / "feature_spec.json", {
        "feature_cols": spec.feature_cols,
        "cat_cols": spec.cat_cols,
        "dropped_cols": spec.dropped_cols,
    })
    feature_summary = build_feature_summary(df, spec.feature_cols, spec.cat_cols)
    save_table_csv(out.tab_dir / "feature_summary.csv", feature_summary)

    # ---- CV（2024まで） ----
    folds = [TimeFold(*x) for x in cfg["folds"]]
    fold_pairs = make_time_folds(df, folds, date_col="race_date")

    rows = []

    optuna_dir = Path(cfg.get("optuna_dir", out.tab_dir))
    manifest_targets: Dict[str, Dict] = {}

    for target_name in tqdm(cfg["target_list"], desc="targets (cv)", position=0):
        y_col = TARGET_MAP[target_name]
        best_path = optuna_dir / f"optuna_best_params_{target_name}.json"
        lgb_params = load_best_params(best_path, cfg["lgb_params"])
        if best_path.exists():
            save_json(out.tab_dir / f"lgb_params_{target_name}_resolved.json", lgb_params)
        for tr_df, va_df, f in tqdm(fold_pairs, desc=f"cv folds ({target_name})", position=1, leave=False):
            base_tr = odds_base_margin(tr_df, odds_col=odds_col) if mode == "residual" else None
            base_va = odds_base_margin(va_df, odds_col=odds_col) if mode == "residual" else None
            res = train_binary(
                tr_df=tr_df,
                va_df=va_df,
                y_col=y_col,
                feature_cols=spec.feature_cols,
                cat_cols=spec.cat_cols,
                params=lgb_params,
                seed=int(cfg["random_seed"]),
                base_margin_tr=base_tr,
                base_margin_va=base_va,
            )
            pred = predict_binary(res.model, tr_df, va_df, spec.feature_cols, spec.cat_cols, base_margin=base_va)
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

    # ★追加：ターゲットごとに best_iter を決めて保存（中央値が安定）
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
    if plan.eval_enabled and plan.test_start:
        train_A, test2025 = make_holdout(df, plan.train_end, plan.test_start, date_col="race_date")
        test2025 = clip_test_end(test2025, plan.test_end, date_col="race_date")
    else:
        train_A = make_train_only(df, plan.train_end, date_col="race_date")
        test2025 = None

    eval_2025 = test2025

    hold_rows = []

    # ★CVからbest_iter（木の本数）を取得
    best_iter_map = json.loads((out.tab_dir / "best_iter_from_cv.json").read_text(encoding="utf-8"))

    for target_name in tqdm(cfg["target_list"], desc="targets (holdout)", position=0):
        y_col = TARGET_MAP[target_name]

        # ★最終学習：2024までのみ、木の本数はCV best_iterで固定（2025は絶対見ない）
        final_rounds = int(best_iter_map[target_name])

        base_train = odds_base_margin(train_A, odds_col=odds_col) if mode == "residual" else None
        res = train_binary(
            tr_df=train_A,
            va_df=None,  # 2025は使わない（評価の信頼性）
            y_col=y_col,
            feature_cols=spec.feature_cols,
            cat_cols=spec.cat_cols,
            params=lgb_params,
            seed=int(cfg["random_seed"]),
            num_boost_round=final_rounds,   # ★ここが重要
            early_stopping_rounds=None,
            base_margin_tr=base_train,
            base_margin_va=None,
        )
        model_path = out.model_dir / f"lgb_binary_{target_name}.txt"
        save_lgb_model(model_path, res.model)
        save_feature_importance(out.tab_dir, target_name, res.model, feature_summary=feature_summary)

        if eval_2025 is not None:
            base_ev = odds_base_margin(eval_2025, odds_col=odds_col) if mode == "residual" else None
            p_ev = predict_binary(res.model, train_A, eval_2025, spec.feature_cols, spec.cat_cols, base_margin=base_ev)
            y_ev = eval_2025[y_col].values.astype(int)

            hold_row = {
                "target": target_name,
                "final_num_boost_round": final_rounds,
                "top1_pos_rate": top1_pos_rate(eval_2025, p_ev, y_ev, seed=int(cfg["random_seed"])),
                "hit_at_1": hit_at_k(eval_2025, p_ev, y_ev, 1, seed=int(cfg["random_seed"])),
                "hit_at_3": hit_at_k(eval_2025, p_ev, y_ev, 3, seed=int(cfg["random_seed"])),
                "hit_at_5": hit_at_k(eval_2025, p_ev, y_ev, 5, seed=int(cfg["random_seed"])),
                "n_eval_races": int(eval_2025["race_id"].nunique()),
            }
            hold_rows.append(hold_row)

            # 回収率計算のために予測を保存
            pred_out = eval_2025[["race_id", "horse_id", "umaban", "race_date"]].copy()
            pred_out["target"] = target_name
            pred_out["p_raw"] = p_ev
            save_table_csv(out.tab_dir / f"preds_{target_name}_2025_eval.csv", pred_out)

        manifest_targets[target_name] = {
            "model_path": str(model_path),
            "params_path": str(out.tab_dir / f"lgb_params_{target_name}_resolved.json"),
            "optuna_best_params_path": str(optuna_dir / f"optuna_best_params_{target_name}.json"),
        }

    hold_df = pd.DataFrame(hold_rows) if hold_rows else pd.DataFrame()
    if not hold_df.empty:
        save_table_csv(out.tab_dir / "holdout_binary_2025.csv", hold_df)
        plot_holdout_binary(hold_df, out.fig_dir / "holdout_binary_2025.png")

    manifest = {
        "mode": mode,
        "odds_col": odds_col,
        "use_market": bool(cfg.get("use_market", False)),
        "config_path": config_path,
        "out_dir": out_dir,
        "feature_spec_path": str(out.tab_dir / "feature_spec.json"),
        "best_iter_path": str(out.tab_dir / "best_iter_from_cv.json"),
        "targets": manifest_targets,
    }
    save_json(out.tab_dir / "model_manifest.json", manifest)

    plot_cv_best_iter(cv_df, out.fig_dir / "cv_best_iter_binary.png", "CV best_iter (binary)")

    report_lines = [
        "# Binary training report",
        f"mode: {plan.mode}",
        f"train_end: {plan.train_end}",
        f"test_start: {plan.test_start}",
        f"test_end: {plan.test_end}",
        f"eval_enabled: {plan.eval_enabled}",
        "",
        f"CV summary: {out.tab_dir / 'cv_summary_binary.csv'}",
    ]
    if not hold_df.empty:
        report_lines += [
            f"Holdout summary: {out.tab_dir / 'holdout_binary_2025.csv'}",
            "",
            hold_df.to_string(index=False),
        ]
    write_report(out.out_dir / "report_binary.md", report_lines)

    print("[done] outputs/train に保存しました")


if __name__ == "__main__":
    main()
