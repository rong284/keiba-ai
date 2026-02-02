from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from src.training.artifacts import save_table_csv
from src.training.calibrate import fit_calibrators, choose_best_calibrator, apply_best
from src.training.data import DataPaths, load_train_dataframe
from src.training.date_plan import resolve_date_plan
from src.training.features import build_feature_spec
from src.training.metrics import prob_metrics, race_equal_weights
from src.training.models.lgb_binary import train_binary, predict_binary
from src.training.reporting import write_report
from src.training.split import TimeFold, make_time_folds, make_holdout
from src.training.tuning_common import load_best_params

TARGET_MAP = {"win": "y_top1", "place": "y_top3"}


def _date_split(df: pd.DataFrame, ratio: float) -> Tuple[pd.DataFrame, pd.DataFrame]:
    d = df.sort_values("race_date").copy()
    if d.empty:
        return d, d
    idx = int(len(d) * (1.0 - ratio))
    idx = max(1, min(len(d) - 1, idx))
    cut = d.iloc[idx]["race_date"]
    left = d[d["race_date"] <= cut].copy()
    right = d[d["race_date"] > cut].copy()
    if right.empty:
        right = left.copy()
    return left, right


def _edge(p: np.ndarray, odds: np.ndarray) -> np.ndarray:
    return p * odds - 1.0


def _profit_from_odds(y: np.ndarray, odds: np.ndarray) -> np.ndarray:
    return np.where(y == 1, odds * 100.0 - 100.0, -100.0)


def _profit_from_payout(y: np.ndarray, payout: np.ndarray) -> np.ndarray:
    return np.where(y == 1, payout - 100.0, -100.0)


def _equity_curve(df: pd.DataFrame) -> pd.DataFrame:
    g = df.groupby("race_date")["profit"].sum().sort_index()
    equity = g.cumsum()
    peak = equity.cummax()
    drawdown = equity - peak
    return pd.DataFrame({
        "race_date": equity.index,
        "equity": equity.values,
        "drawdown": drawdown.values,
    })


def _band_roi_table(
    df: pd.DataFrame,
    value_col: str,
    bins: List[float],
    labels: List[str],
) -> pd.DataFrame:
    d = df.copy()
    d["band"] = pd.cut(d[value_col], bins=bins, labels=labels, include_lowest=True, right=False)
    rows = []
    for band, g in d.groupby("band", dropna=False):
        n = len(g)
        total_bet = n * 100.0
        total_profit = float(g["profit"].sum()) if n > 0 else 0.0
        total_return = total_profit + total_bet
        roi = total_return / total_bet if total_bet > 0 else 0.0
        hit_rate = float((g["y"] == 1).mean()) if n > 0 else 0.0
        rows.append({
            "band": str(band),
            "n_bets": int(n),
            "roi": roi,
            "profit": total_profit,
            "hit_rate": hit_rate,
        })
    return pd.DataFrame(rows)


def _race_softmax(df: pd.DataFrame, p_col: str) -> pd.Series:
    def norm(g: pd.Series) -> pd.Series:
        s = g.sum()
        if s <= 0:
            return pd.Series(np.full(len(g), 1.0 / len(g)), index=g.index)
        return g / s
    return df.groupby("race_id")[p_col].transform(norm)


def _race_gap(df: pd.DataFrame, p_col: str) -> pd.Series:
    def gap(g: pd.DataFrame) -> float:
        vals = np.sort(g[p_col].values)[::-1]
        if len(vals) < 2:
            return 0.0
        return float(vals[0] - vals[1])
    return df.groupby("race_id", group_keys=False).apply(gap).reindex(df["race_id"]).values


def _cap_per_race(df: pd.DataFrame, score_col: str, max_per_race: int) -> pd.DataFrame:
    if max_per_race is None or max_per_race <= 0:
        return df
    return (df.sort_values(["race_id", score_col], ascending=[True, False])
              .groupby("race_id", as_index=False)
              .head(max_per_race))


def _sweep_threshold_edge(
    df: pd.DataFrame,
    thresholds: List[float],
    max_per_race: int,
    min_p: Optional[float],
    min_odds_grid: List[Optional[float]],
    max_odds_grid: List[Optional[float]],
    gap_grid: List[Optional[float]],
    edge_max_grid: List[Optional[float]],
) -> pd.DataFrame:
    rows = []
    for t in thresholds:
        for min_odds in min_odds_grid:
            for max_odds in max_odds_grid:
                if min_odds is not None and max_odds is not None and min_odds > max_odds:
                    continue
                for gap in gap_grid:
                    for edge_max in edge_max_grid:
                        picks = df[df["edge"] >= t].copy()
                        if edge_max is not None:
                            picks = picks[picks["edge"] <= edge_max]
                        if min_p is not None:
                            picks = picks[picks["p_calib"] >= min_p]
                        if min_odds is not None:
                            picks = picks[picks["odds"] >= min_odds]
                        if max_odds is not None:
                            picks = picks[picks["odds"] <= max_odds]
                        if gap is not None:
                            picks = picks[picks["gap"] >= gap]
                        picks = _cap_per_race(picks, "edge", max_per_race)
                        n_bets = len(picks)
                        total_bet = n_bets * 100.0
                        total_profit = float(picks["profit"].sum())
                        total_return = total_profit + total_bet
                        roi = total_return / total_bet if total_bet > 0 else 0.0
                        hit_rate = float((picks["y"] == 1).mean()) if n_bets > 0 else 0.0
                        rows.append({
                            "edge_threshold": t,
                            "edge_max": edge_max,
                            "min_odds": min_odds,
                            "max_odds": max_odds,
                            "gap_threshold": gap,
                            "n_bets": int(n_bets),
                            "roi": roi,
                            "profit": total_profit,
                            "hit_rate": hit_rate,
                        })
    return pd.DataFrame(rows)


def _sweep_threshold_place(
    df: pd.DataFrame,
    p_grid: List[float],
    g_grid: List[float],
    max_per_race: int,
) -> pd.DataFrame:
    rows = []
    for t in p_grid:
        for g in g_grid:
            picks = df[(df["p_calib"] >= t) & (df["gap"] >= g)].copy()
            picks = _cap_per_race(picks, "p_calib", max_per_race)
            n_bets = len(picks)
            total_bet = n_bets * 100.0
            total_profit = float(picks["profit"].sum()) if "profit" in picks.columns else 0.0
            total_return = total_profit + total_bet
            roi = total_return / total_bet if total_bet > 0 else 0.0
            hit_rate = float((picks["y"] == 1).mean()) if n_bets > 0 else 0.0
            rows.append({
                "p_threshold": t,
                "gap_threshold": g,
                "n_bets": int(n_bets),
                "roi": roi,
                "profit": total_profit,
                "hit_rate": hit_rate,
            })
    return pd.DataFrame(rows)


def _select_best(df: pd.DataFrame, metric: str, min_bets: int) -> Optional[Dict]:
    if df.empty:
        return None
    d = df[df["n_bets"] >= min_bets].copy()
    if d.empty:
        return None
    if metric not in d.columns:
        metric = "roi"
    best = d.sort_values(metric, ascending=False).iloc[0].to_dict()
    return best


def _apply_win_rule(
    df: pd.DataFrame,
    rule: Dict,
    min_p: Optional[float],
    max_per_race: int,
) -> pd.DataFrame:
    picks = df[df["edge"] >= rule["edge_threshold"]].copy()
    if rule.get("edge_max") is not None:
        picks = picks[picks["edge"] <= float(rule["edge_max"])]
    if min_p is not None:
        picks = picks[picks["p_calib"] >= min_p]
    if rule.get("min_odds") is not None:
        picks = picks[picks["odds"] >= float(rule["min_odds"])]
    if rule.get("max_odds") is not None:
        picks = picks[picks["odds"] <= float(rule["max_odds"])]
    if rule.get("gap_threshold") is not None:
        picks = picks[picks["gap"] >= float(rule["gap_threshold"])]
    picks = _cap_per_race(picks, "edge", max_per_race)
    return picks


def _aggregate_rule_stats(per_fold: List[Dict]) -> Dict:
    rois = [x["roi"] for x in per_fold if x["n_bets"] > 0]
    profits = [x["profit"] for x in per_fold]
    n_bets = sum(x["n_bets"] for x in per_fold)
    total_profit = sum(x["profit"] for x in per_fold)
    total_bet = n_bets * 100.0
    roi_total = total_profit / total_bet + 1.0 if total_bet > 0 else 0.0
    roi_mean = float(np.mean(rois)) if rois else 0.0
    roi_median = float(np.median(rois)) if rois else 0.0
    roi_min = float(np.min(rois)) if rois else 0.0
    roi_std = float(np.std(rois)) if rois else 0.0
    roi_mean_minus_std = roi_mean - roi_std
    return {
        "n_bets": n_bets,
        "profit": total_profit,
        "roi_total": roi_total,
        "roi_mean": roi_mean,
        "roi_median": roi_median,
        "roi_min": roi_min,
        "roi_mean_minus_std": roi_mean_minus_std,
    }


def _to_markdown_table(df: pd.DataFrame) -> str:
    if df.empty:
        return "(empty)"
    cols = list(df.columns)
    header = "| " + " | ".join(cols) + " |"
    sep = "| " + " | ".join(["---"] * len(cols)) + " |"
    lines = [header, sep]
    for _, row in df.iterrows():
        vals = ["" if pd.isna(row[c]) else str(row[c]) for c in cols]
        lines.append("| " + " | ".join(vals) + " |")
    return "\n".join(lines)


def _plot_calibration(out_dir: Path, tag: str, y: np.ndarray, p_raw: np.ndarray, p_cal: np.ndarray):
    import matplotlib.pyplot as plt
    from sklearn.calibration import calibration_curve

    fig = plt.figure(figsize=(6.0, 5.0))
    ax = fig.add_subplot(111)
    frac_pos, mean_pred = calibration_curve(y, p_raw, n_bins=10, strategy="quantile")
    ax.plot(mean_pred, frac_pos, marker="o", label="raw")
    frac_pos_c, mean_pred_c = calibration_curve(y, p_cal, n_bins=10, strategy="quantile")
    ax.plot(mean_pred_c, frac_pos_c, marker="o", label="calib")
    ax.plot([0, 1], [0, 1], linestyle="--", color="#999999")
    ax.set_xlabel("mean predicted")
    ax.set_ylabel("fraction positive")
    ax.set_title(f"Calibration ({tag})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"calibration_{tag}.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(6.0, 4.2))
    ax = fig.add_subplot(111)
    ax.hist(p_raw[y == 1], bins=30, alpha=0.7, label="pos")
    ax.hist(p_raw[y == 0], bins=30, alpha=0.7, label="neg")
    ax.set_title(f"Pred dist raw ({tag})")
    ax.set_xlabel("p_raw")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"pred_dist_raw_{tag}.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(6.0, 4.2))
    ax = fig.add_subplot(111)
    ax.hist(p_cal[y == 1], bins=30, alpha=0.7, label="pos")
    ax.hist(p_cal[y == 0], bins=30, alpha=0.7, label="neg")
    ax.set_title(f"Pred dist calib ({tag})")
    ax.set_xlabel("p_calib")
    ax.set_ylabel("count")
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"pred_dist_calib_{tag}.png", dpi=150)
    plt.close(fig)


def _plot_threshold(out_dir: Path, tag: str, df: pd.DataFrame, x_col: str, title: str):
    if df.empty:
        return
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.2, 4.2))
    ax = fig.add_subplot(111)
    ax.plot(df[x_col], df["roi"], marker="o", label="roi")
    ax.plot(df[x_col], df["profit"] / 1000.0, marker="o", label="profit(k)")
    ax.plot(df[x_col], df["n_bets"], marker="o", label="bets")
    ax.set_title(title)
    ax.set_xlabel(x_col)
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_dir / f"threshold_sweep_{tag}.png", dpi=150)
    plt.close(fig)


def _plot_equity(out_dir: Path, tag: str, eq: pd.DataFrame):
    if eq.empty:
        return
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.6, 4.2))
    ax = fig.add_subplot(111)
    ax.plot(eq["race_date"], eq["equity"], label="equity")
    ax.set_title(f"Equity curve ({tag})")
    ax.set_xlabel("date")
    ax.set_ylabel("profit")
    fig.tight_layout()
    fig.savefig(out_dir / f"equity_{tag}.png", dpi=150)
    plt.close(fig)

    fig = plt.figure(figsize=(7.6, 3.6))
    ax = fig.add_subplot(111)
    ax.plot(eq["race_date"], eq["drawdown"], label="drawdown", color="#EF4444")
    ax.set_title(f"Drawdown ({tag})")
    ax.set_xlabel("date")
    ax.set_ylabel("drawdown")
    fig.tight_layout()
    fig.savefig(out_dir / f"drawdown_{tag}.png", dpi=150)
    plt.close(fig)


def _load_return_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["race_id", "bet_type", "combination", "payout"]).copy()
    df["payout"] = df["payout"].astype(str).str.replace(r"\D", "", regex=True)
    df["payout"] = pd.to_numeric(df["payout"], errors="coerce").fillna(0).astype(int)
    df["combination"] = df["combination"].astype(str).str.replace(" ", "")
    return df


def _build_payout_map(df: pd.DataFrame, bet_type: str) -> Dict[Tuple[str, str], int]:
    d = df[df["bet_type"] == bet_type].copy()
    d["race_id"] = d["race_id"].astype(str)
    d["combination"] = d["combination"].astype(str)
    payout = {}
    for _, r in d.iterrows():
        payout[(r["race_id"], r["combination"])] = int(r["payout"])
    return payout


def main(config_path: str, out_dir: str, return_path: str):
    cfg = json.loads(Path(config_path).read_text(encoding="utf-8"))
    plan = resolve_date_plan(cfg)
    out_dir = Path(out_dir)
    fig_dir = out_dir / "figures"
    tab_dir = out_dir / "tables"
    fig_dir.mkdir(parents=True, exist_ok=True)
    tab_dir.mkdir(parents=True, exist_ok=True)

    df = load_train_dataframe(DataPaths(
        result_glob="data/rawdf/result/result_*.csv",
        horse_glob="data/rawdf/horse/*.csv",
        race_info_glob="data/rawdf/race_info/*.csv",
    ))

    spec = build_feature_spec(df, use_market=bool(cfg.get("use_market", False)))
    folds = [TimeFold(*x) for x in cfg["folds"]]
    fold_pairs = make_time_folds(df, folds, date_col="race_date")

    win_odds_col = cfg.get("win_odds_col", "tansho_odds")
    edge_grid = cfg.get("edge_grid", [round(x, 3) for x in np.linspace(0.0, 0.2, 11)])
    edge_min_bets = int(cfg.get("edge_min_bets", 100))
    edge_min_bets_per_fold = int(cfg.get("edge_min_bets_per_fold", 0))
    edge_select_metric = cfg.get("edge_select_metric", "roi_mean")
    edge_select_metric = {
        "roi": "roi_mean",
        "profit": "profit",
    }.get(edge_select_metric, edge_select_metric)
    win_max_per_race = int(cfg.get("win_max_per_race", 1))
    win_min_p = cfg.get("win_min_p")
    win_min_p = float(win_min_p) if win_min_p is not None else None
    win_min_odds_grid = cfg.get("win_min_odds_grid", [None])
    win_max_odds_grid = cfg.get("win_max_odds_grid", [None])
    win_gap_grid = cfg.get("win_gap_grid", [None])
    win_edge_max_grid = cfg.get("win_edge_max_grid", [None])
    win_min_odds_grid = [None if x is None else float(x) for x in win_min_odds_grid]
    win_max_odds_grid = [None if x is None else float(x) for x in win_max_odds_grid]
    win_gap_grid = [None if x is None else float(x) for x in win_gap_grid]
    win_edge_max_grid = [None if x is None else float(x) for x in win_edge_max_grid]

    p_grid = cfg.get("place_prob_grid", [round(x, 3) for x in np.linspace(0.05, 0.5, 10)])
    g_grid = cfg.get("place_gap_grid", [round(x, 3) for x in np.linspace(0.01, 0.2, 10)])
    place_min_bets = int(cfg.get("place_min_bets", 100))
    place_select_metric = cfg.get("place_select_metric", "hit_rate")
    place_max_per_race = int(cfg.get("place_max_per_race", 1))

    calib_valid_ratio = float(cfg.get("calib_valid_ratio", 0.5))
    calib_train_ratio = float(cfg.get("calib_train_ratio", 0.2))

    optuna_dir = Path(cfg.get("optuna_dir", tab_dir))

    ret = _load_return_table(return_path)
    payout_win = _build_payout_map(ret, "単勝")
    payout_place = _build_payout_map(ret, "複勝")

    scoreboard_rows = []
    report_extra_lines: List[str] = []

    for target_name in cfg["target_list"]:
        y_col = TARGET_MAP[target_name]
        odds_col = win_odds_col if target_name == "win" else None
        has_odds = odds_col is not None and odds_col in df.columns

        best_path = optuna_dir / f"optuna_best_params_{target_name}.json"
        lgb_params = load_best_params(best_path, cfg["lgb_params"])

        # ---- CV（walk-forward）----
        cv_prob_rows = []
        cv_rows = []
        best_kind_counts: Dict[str, int] = {}
        best_iters: List[int] = []

        for tr_df, va_df, _f in fold_pairs:
            res = train_binary(
                tr_df=tr_df,
                va_df=va_df,
                y_col=y_col,
                feature_cols=spec.feature_cols,
                cat_cols=spec.cat_cols,
                params=lgb_params,
                seed=int(cfg["random_seed"]),
            )
            best_iters.append(int(res.best_iter))
            p_raw = predict_binary(res.model, tr_df, va_df, spec.feature_cols, spec.cat_cols)

            va_df = va_df.copy()
            va_df["p_raw"] = p_raw
            va_df["p_norm"] = _race_softmax(va_df, "p_raw")
            cal_fit, cal_eval = _date_split(va_df, calib_valid_ratio)

            calib = fit_calibrators(
                cal_fit["p_norm"].values,
                cal_fit[y_col].values.astype(int),
                race_equal_weights(cal_fit, "race_id"),
            )
            best_kind, rep = choose_best_calibrator(
                cal_eval[y_col].values.astype(int),
                cal_eval["p_norm"].values,
                calib,
            )
            best_kind_counts[best_kind] = best_kind_counts.get(best_kind, 0) + 1
            p_cal = apply_best(best_kind, cal_eval["p_norm"].values, calib)

            pm = prob_metrics(cal_eval[y_col].values.astype(int), p_cal)
            cv_prob_rows.append({"logloss": pm["logloss"], "brier": pm["brier"]})

            tmp = pd.DataFrame({
                "race_date": cal_eval["race_date"].values,
                "race_id": cal_eval["race_id"].values,
                "umaban": cal_eval["umaban"].values,
                "y": cal_eval[y_col].values.astype(int),
                "p_calib": p_cal,
            })

            if target_name == "win" and has_odds:
                odds = cal_eval[odds_col].astype(float).fillna(0.0).values
                tmp["odds"] = odds
                tmp["edge"] = _edge(p_cal, odds)
                tmp["gap"] = _race_gap(cal_eval.assign(p_calib=p_cal), "p_calib")
                tmp["profit"] = _profit_from_odds(cal_eval[y_col].values.astype(int), odds)
            else:
                tmp["gap"] = _race_gap(cal_eval, "p_raw")
            cv_rows.append(tmp)

        cv_prob = pd.DataFrame(cv_prob_rows)
        best_iter = int(np.median(best_iters)) if best_iters else 200
        best_kind = max(best_kind_counts.items(), key=lambda kv: kv[1])[0] if best_kind_counts else "platt"

        cv_score = {
            "model": target_name,
            "period": "cv",
            "logloss": float(cv_prob["logloss"].mean()) if not cv_prob.empty else None,
            "brier": float(cv_prob["brier"].mean()) if not cv_prob.empty else None,
        }

        cv_sweep = pd.DataFrame()
        best_rule = None
        if target_name == "win" and has_odds:
            fold_dfs = [x.copy() for x in cv_rows]
            rules = _sweep_threshold_edge(
                pd.concat(fold_dfs, ignore_index=True),
                edge_grid,
                win_max_per_race,
                win_min_p,
                win_min_odds_grid,
                win_max_odds_grid,
                win_gap_grid,
                win_edge_max_grid,
            )
            rule_rows = []
            for _, r in rules.iterrows():
                rule = r.to_dict()
                per_fold = []
                ok = True
                for fdf in fold_dfs:
                    picks = _apply_win_rule(fdf, rule, win_min_p, win_max_per_race)
                    n_bets = len(picks)
                    if edge_min_bets_per_fold and n_bets < edge_min_bets_per_fold:
                        ok = False
                    total_profit = float(picks["profit"].sum())
                    total_bet = n_bets * 100.0
                    roi = (total_profit + total_bet) / total_bet if total_bet > 0 else 0.0
                    hit_rate = float((picks["y"] == 1).mean()) if n_bets > 0 else 0.0
                    per_fold.append({"n_bets": n_bets, "profit": total_profit, "roi": roi, "hit_rate": hit_rate})
                if not ok:
                    continue
                agg = _aggregate_rule_stats(per_fold)
                rule_rows.append({
                    **rule,
                    "n_bets": agg["n_bets"],
                    "profit": agg["profit"],
                    "roi_mean": agg["roi_mean"],
                    "roi_median": agg["roi_median"],
                    "roi_min": agg["roi_min"],
                    "roi_mean_minus_std": agg["roi_mean_minus_std"],
                    "roi_total": agg["roi_total"],
                })
            cv_sweep = pd.DataFrame(rule_rows)
            best_rule = _select_best(cv_sweep, edge_select_metric, edge_min_bets)
            if best_rule:
                cv_score.update({
                    "edge_threshold": best_rule["edge_threshold"],
                    "edge_max": best_rule.get("edge_max"),
                    "min_odds": best_rule["min_odds"],
                    "max_odds": best_rule["max_odds"],
                    "gap_threshold": best_rule["gap_threshold"],
                    "bets": int(best_rule["n_bets"]),
                    "roi": float(best_rule.get(edge_select_metric, 0.0)),
                    "profit": float(best_rule["profit"]),
                })
            save_table_csv(tab_dir / "cv_threshold_sweep_win.csv", cv_sweep)
        elif target_name == "place":
            cv_all = pd.concat(cv_rows, ignore_index=True)
            cv_all["gap"] = _race_gap(cv_all, "p_calib")
            cv_sweep = _sweep_threshold_place(cv_all, p_grid, g_grid, place_max_per_race)
            best_rule = _select_best(cv_sweep, place_select_metric, place_min_bets)
            if best_rule:
                cv_score.update({
                    "p_threshold": best_rule["p_threshold"],
                    "gap_threshold": best_rule["gap_threshold"],
                    "bets": int(best_rule["n_bets"]),
                    "roi": float(best_rule["roi"]),
                    "profit": float(best_rule["profit"]),
                    "hit_rate": float(best_rule["hit_rate"]),
                })
            save_table_csv(tab_dir / "cv_threshold_sweep_place.csv", cv_sweep)
            _plot_threshold(fig_dir, "place_cv", cv_sweep, "p_threshold", "Prob sweep (cv, place)")

        scoreboard_rows.append(cv_score)

        # ---- 最終評価（2025 holdout）----
        train_A, test_2025 = make_holdout(df, plan.train_end, plan.test_start, date_col="race_date")
        if test_2025.empty:
            continue

        train_model, calib_fit = _date_split(train_A, calib_train_ratio)

        res = train_binary(
            tr_df=train_model,
            va_df=None,
            y_col=y_col,
            feature_cols=spec.feature_cols,
            cat_cols=spec.cat_cols,
            params=lgb_params,
            seed=int(cfg["random_seed"]),
            num_boost_round=best_iter,
            early_stopping_rounds=None,
        )
        p_fit = predict_binary(res.model, train_model, calib_fit, spec.feature_cols, spec.cat_cols)
        calib_fit = calib_fit.copy()
        calib_fit["p_norm"] = _race_softmax(calib_fit.assign(p_raw=p_fit), "p_raw")
        calib = fit_calibrators(
            calib_fit["p_norm"].values,
            calib_fit[y_col].values.astype(int),
            race_equal_weights(calib_fit, "race_id"),
        )

        p_raw_ev = predict_binary(res.model, train_model, test_2025, spec.feature_cols, spec.cat_cols)
        test_2025 = test_2025.copy()
        test_2025["p_raw"] = p_raw_ev
        test_2025["p_norm"] = _race_softmax(test_2025, "p_raw")
        p_cal_ev = apply_best(best_kind, test_2025["p_norm"].values, calib)

        pm_ev = prob_metrics(test_2025[y_col].values.astype(int), p_cal_ev)
        eval_score = {
            "model": target_name,
            "period": "2025",
            "logloss": pm_ev["logloss"],
            "brier": pm_ev["brier"],
        }

        payout_map = payout_win if target_name == "win" else payout_place
        payout = []
        for rid, um in zip(test_2025["race_id"].astype(str), test_2025["umaban"].astype(str)):
            payout.append(payout_map.get((rid, um), 0))
        payout = np.asarray(payout, dtype=float)

        df_ev = pd.DataFrame({
            "race_date": test_2025["race_date"].values,
            "race_id": test_2025["race_id"].values,
            "umaban": test_2025["umaban"].values,
            "y": test_2025[y_col].values.astype(int),
            "p_raw": p_raw_ev,
            "p_norm": test_2025["p_norm"].values,
            "p_calib": p_cal_ev,
            "payout": payout,
        })

        if target_name == "win" and has_odds:
            odds = test_2025[odds_col].astype(float).fillna(0.0).values
            df_ev["odds"] = odds
            df_ev["edge"] = _edge(p_cal_ev, odds)
            df_ev["gap"] = _race_gap(df_ev, "p_calib")
            df_ev["profit"] = _profit_from_payout(test_2025[y_col].values.astype(int), payout)
            if best_rule:
                picks = df_ev[df_ev["edge"] >= best_rule["edge_threshold"]].copy()
                if best_rule.get("edge_max") is not None:
                    picks = picks[picks["edge"] <= float(best_rule["edge_max"])]
                if win_min_p is not None:
                    picks = picks[picks["p_calib"] >= win_min_p]
                if best_rule.get("min_odds") is not None:
                    picks = picks[picks["odds"] >= float(best_rule["min_odds"])]
                if best_rule.get("max_odds") is not None:
                    picks = picks[picks["odds"] <= float(best_rule["max_odds"])]
                if best_rule.get("gap_threshold") is not None:
                    picks = picks[picks["gap"] >= float(best_rule["gap_threshold"])]
            else:
                picks = df_ev.copy()
            picks = _cap_per_race(picks, "edge", win_max_per_race)
            if win_min_p is not None:
                assert (picks["p_calib"] >= win_min_p - 1e-9).all()
            if best_rule and best_rule.get("max_odds") is not None:
                assert (picks["odds"] <= float(best_rule["max_odds"]) + 1e-9).all()
            if best_rule and best_rule.get("min_odds") is not None:
                assert (picks["odds"] >= float(best_rule["min_odds"]) - 1e-9).all()
        else:
            df_ev["gap"] = _race_gap(df_ev, "p_calib")
            df_ev["profit"] = _profit_from_payout(test_2025[y_col].values.astype(int), payout)
            if best_rule:
                picks = df_ev[(df_ev["p_calib"] >= best_rule["p_threshold"]) & (df_ev["gap"] >= best_rule["gap_threshold"])].copy()
            else:
                picks = df_ev.copy()
            picks = _cap_per_race(picks, "p_calib", place_max_per_race)

        n_bets = len(picks)
        total_bet = n_bets * 100.0
        total_profit = float(picks["profit"].sum())
        total_return = total_profit + total_bet
        roi = total_return / total_bet if total_bet > 0 else 0.0
        hit_rate = float((picks["y"] == 1).mean()) if n_bets > 0 else 0.0
        eq = _equity_curve(picks)
        dd = float(eq["drawdown"].min()) if not eq.empty else 0.0

        eval_score.update({
            "bets": int(n_bets),
            "roi": roi,
            "profit": total_profit,
            "hit_rate": hit_rate,
            "max_drawdown": dd,
        })
        if target_name == "win" and best_rule:
            eval_score["edge_threshold"] = best_rule["edge_threshold"]
            eval_score["edge_max"] = best_rule.get("edge_max")
            eval_score["min_odds"] = best_rule["min_odds"]
            eval_score["max_odds"] = best_rule["max_odds"]
            eval_score["gap_threshold"] = best_rule["gap_threshold"]
        if target_name == "place" and best_rule:
            eval_score["p_threshold"] = best_rule["p_threshold"]
            eval_score["gap_threshold"] = best_rule["gap_threshold"]

        scoreboard_rows.append(eval_score)

        save_table_csv(tab_dir / f"preds_{target_name}_2025.csv", df_ev)
        save_table_csv(tab_dir / f"drilldown_{target_name}_2025.csv", picks)

        _plot_equity(fig_dir, f"{target_name}_2025", eq)
        _plot_calibration(fig_dir, f"{target_name}_2025", test_2025[y_col].values.astype(int), test_2025["p_norm"].values, p_cal_ev)

        # band analysis（単勝のみ）
        if target_name == "win" and not picks.empty:
            odds_bins = cfg.get("odds_bins", [1, 2, 5, 10, 20, 30, 50, 100, 1000])
            odds_bins = [np.inf if str(x).lower() == "inf" else float(x) for x in odds_bins]
            odds_labels = [f"{odds_bins[i]}-{odds_bins[i+1]}" for i in range(len(odds_bins) - 1)]
            odds_band = _band_roi_table(picks, "odds", odds_bins, odds_labels)
            save_table_csv(tab_dir / "band_roi_odds_win.csv", odds_band)

            p_bins = cfg.get("p_bins", [0.0, 0.02, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0])
            p_bins = [np.inf if str(x).lower() == "inf" else float(x) for x in p_bins]
            p_labels = [f"{p_bins[i]}-{p_bins[i+1]}" for i in range(len(p_bins) - 1)]
            p_band = _band_roi_table(picks, "p_calib", p_bins, p_labels)
            save_table_csv(tab_dir / "band_roi_p_win.csv", p_band)

            edge_bins = cfg.get("edge_bins", [-1.0, 0.0, 0.05, 0.1, 0.15, 0.2, 0.3, 0.5, 1.0, "inf"])
            edge_bins = [np.inf if str(x).lower() == "inf" else float(x) for x in edge_bins]
            edge_labels = [f"{edge_bins[i]}-{edge_bins[i+1]}" for i in range(len(edge_bins) - 1)]
            edge_band = _band_roi_table(picks, "edge", edge_bins, edge_labels)
            save_table_csv(tab_dir / "band_roi_edge_win.csv", edge_band)

            report_extra_lines += [
                "",
                "## Win band analysis (2025)",
                "",
                "### Odds band ROI",
                _to_markdown_table(odds_band),
                "",
                "### p_calib band ROI",
                _to_markdown_table(p_band),
                "",
                "### edge band ROI",
                _to_markdown_table(edge_band),
            ]

    scoreboard = pd.DataFrame(scoreboard_rows)
    save_table_csv(tab_dir / "scoreboard.csv", scoreboard)

    report_lines = [
        "# Backtest report (rules-based)",
        "",
        "## Scoreboard (models x period)",
        "",
        _to_markdown_table(scoreboard),
        "",
        "## Notes",
        "- Win uses edge threshold with odds and optional min_p/max_odds filters.",
        "- Place uses prob+gap rule (no place odds).",
        "- 2025 is used only for final evaluation.",
    ] + report_extra_lines
    write_report(out_dir / "report.md", report_lines)

    print("[done] backtest saved to", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", default="src/configs/train_binary.json")
    ap.add_argument("--out_dir", default="outputs/05_backtest_rules")
    ap.add_argument("--return_path", default="data/rawdf/return/return_2025.csv")
    args = ap.parse_args()
    main(args.config, args.out_dir, args.return_path)
