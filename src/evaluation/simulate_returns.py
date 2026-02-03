from __future__ import annotations

import argparse
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from src.training.artifacts import save_table_csv
from src.training.reporting import write_report


BET_TYPE_WIN = "単勝"
BET_TYPE_PLACE = "複勝"


def _parse_payout(v: str) -> int:
    if v is None:
        return 0
    s = str(v)
    digits = "".join(ch for ch in s if ch.isdigit())
    return int(digits) if digits else 0


def load_return_table(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df = df.dropna(subset=["race_id", "bet_type", "combination", "payout"]).copy()
    df["payout"] = df["payout"].map(_parse_payout)
    df["combination"] = df["combination"].astype(str).str.replace(" ", "")
    return df


def build_payout_map(df: pd.DataFrame, bet_type: str) -> Dict[Tuple[str, str], int]:
    d = df[df["bet_type"] == bet_type].copy()
    d["race_id"] = d["race_id"].astype(str)
    d["combination"] = d["combination"].astype(str)
    payout = {}
    for _, r in d.iterrows():
        key = (r["race_id"], r["combination"])
        payout[key] = int(r["payout"])
    return payout


def _minmax_by_race(df: pd.DataFrame, score_col: str) -> pd.Series:
    def _scale(g: pd.Series) -> pd.Series:
        mn = g.min()
        mx = g.max()
        if mx == mn:
            return pd.Series(np.zeros(len(g)), index=g.index)
        return (g - mn) / (mx - mn)

    return df.groupby("race_id")[score_col].transform(_scale)


def simulate_strategy(
    preds: pd.DataFrame,
    payout_map: Dict[Tuple[str, str], int],
    bet_type: str,
    strategy: str,
    threshold: Optional[float] = None,
) -> Dict:
    if strategy == "top1":
        picks = preds.sort_values(["race_id", "score"], ascending=[True, False]).groupby("race_id").head(1)
    elif strategy == "threshold":
        if threshold is None:
            raise ValueError("threshold is required for threshold strategy")
        picks = preds[preds["score"] >= threshold].copy()
    else:
        raise ValueError(f"unknown strategy: {strategy}")

    if picks.empty:
        return {
            "strategy": strategy,
            "bet_type": bet_type,
            "threshold": threshold,
            "n_bets": 0,
            "total_bet": 0,
            "total_return": 0,
            "roi": 0.0,
        }

    picks["race_id"] = picks["race_id"].astype(str)
    picks["umaban"] = picks["umaban"].astype(str)

    total_return = 0
    for _, r in picks.iterrows():
        total_return += payout_map.get((r["race_id"], r["umaban"]), 0)

    n_bets = len(picks)
    total_bet = n_bets * 100
    roi = float(total_return / total_bet) if total_bet > 0 else 0.0

    return {
        "strategy": strategy,
        "bet_type": bet_type,
        "threshold": threshold,
        "n_bets": n_bets,
        "total_bet": total_bet,
        "total_return": total_return,
        "roi": roi,
    }


def build_ensemble(
    win: Optional[pd.DataFrame],
    place: Optional[pd.DataFrame],
    rank: Optional[pd.DataFrame],
    weights: Dict[str, float],
) -> pd.DataFrame:
    frames = []
    if win is not None:
        w = win[["race_id", "horse_id", "umaban", "score"]].copy()
        w["score"] = _minmax_by_race(w, "score")
        w = w.rename(columns={"score": "win_score"})
        frames.append(w)
    if place is not None:
        p = place[["race_id", "horse_id", "umaban", "score"]].copy()
        p["score"] = _minmax_by_race(p, "score")
        p = p.rename(columns={"score": "place_score"})
        frames.append(p)
    if rank is not None:
        r = rank[["race_id", "horse_id", "umaban", "score"]].copy()
        r["score"] = _minmax_by_race(r, "score")
        r = r.rename(columns={"score": "rank_score"})
        frames.append(r)

    if not frames:
        raise ValueError("no model predictions provided for ensemble")

    base = frames[0]
    for f in frames[1:]:
        base = base.merge(f, on=["race_id", "horse_id", "umaban"], how="inner")

    score = 0.0
    denom = 0.0
    for k, w in weights.items():
        col = f"{k}_score"
        if col in base.columns:
            score += w * base[col].values
            denom += w
    if denom == 0:
        raise ValueError("ensemble weights sum to zero")

    out = base[["race_id", "horse_id", "umaban"]].copy()
    out["score"] = score / denom
    return out


def _load_preds(path: Optional[str], score_col: str) -> Optional[pd.DataFrame]:
    if not path:
        return None
    df = pd.read_csv(path)
    df = df.dropna(subset=["race_id", "horse_id", "umaban"]).copy()
    df["score"] = df[score_col].astype(float)
    return df[["race_id", "horse_id", "umaban", "score"]]


def main(
    return_path: str,
    out_dir: str,
    preds_win: Optional[str] = None,
    preds_place: Optional[str] = None,
    preds_rank: Optional[str] = None,
    bet_type: str = BET_TYPE_WIN,
    thresholds: Optional[str] = None,
    ensemble_weights: str = "win:1,place:1,rank:1",
):
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    ret = load_return_table(return_path)
    payout_map = build_payout_map(ret, bet_type)

    win = _load_preds(preds_win, "p_raw")
    place = _load_preds(preds_place, "p_raw")
    rank = _load_preds(preds_rank, "score")

    weights = {}
    for part in ensemble_weights.split(","):
        if not part:
            continue
        k, v = part.split(":")
        weights[k.strip()] = float(v)

    rows = []

    # モデル別 top1
    for name, df in [("win", win), ("place", place), ("rank", rank)]:
        if df is None:
            continue
        r = simulate_strategy(df, payout_map, bet_type, strategy="top1")
        r["model"] = name
        rows.append(r)

    # アンサンブル top1
    if any(x is not None for x in [win, place, rank]):
        ens = build_ensemble(win, place, rank, weights)
        r = simulate_strategy(ens, payout_map, bet_type, strategy="top1")
        r["model"] = "ensemble"
        rows.append(r)

    # threshold探索
    thr_list = []
    if thresholds:
        thr_list = [float(x) for x in thresholds.split(",") if x.strip()]
    else:
        thr_list = [round(x, 3) for x in np.linspace(0.05, 0.5, 10)]

    for name, df in [("win", win), ("place", place), ("rank", rank)]:
        if df is None:
            continue
        for t in thr_list:
            r = simulate_strategy(df, payout_map, bet_type, strategy="threshold", threshold=t)
            r["model"] = name
            rows.append(r)

    if any(x is not None for x in [win, place, rank]):
        ens = build_ensemble(win, place, rank, weights)
        for t in thr_list:
            r = simulate_strategy(ens, payout_map, bet_type, strategy="threshold", threshold=t)
            r["model"] = "ensemble"
            rows.append(r)

    out_df = pd.DataFrame(rows)
    save_table_csv(out_dir / "return_simulation.csv", out_df)

    # 簡易プロット: ROI と threshold
    try:
        import matplotlib.pyplot as plt

        th = out_df[out_df["strategy"] == "threshold"].copy()
        if not th.empty:
            fig = plt.figure(figsize=(7.6, 4.2))
            ax = fig.add_subplot(111)
            for model in sorted(th["model"].unique()):
                d = th[th["model"] == model]
                ax.plot(d["threshold"], d["roi"], marker="o", label=model)
            ax.set_title(f"ROI vs threshold ({bet_type})")
            ax.set_xlabel("threshold")
            ax.set_ylabel("roi")
            ax.legend()
            fig.tight_layout()
            fig.savefig(out_dir / "return_roi_vs_threshold.png", dpi=150)
            plt.close(fig)
    except Exception:
        pass

    report_lines = [
        "# Return simulation report",
        f"return_path: {return_path}",
        f"bet_type: {bet_type}",
        f"preds_win: {preds_win}",
        f"preds_place: {preds_place}",
        f"preds_rank: {preds_rank}",
        f"ensemble_weights: {ensemble_weights}",
        "",
        f"output: {out_dir / 'return_simulation.csv'}",
    ]
    write_report(out_dir / "return_report.md", report_lines)

    print("[done] return simulation saved to", out_dir)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--return_path", default="data/rawdf/return/return_2025.csv")
    ap.add_argument("--out_dir", default="outputs/returns/2025")
    ap.add_argument("--preds_win", default="outputs/train/binary/tables/preds_win_2025_eval.csv")
    ap.add_argument("--preds_place", default="outputs/train/binary/tables/preds_place_2025_eval.csv")
    ap.add_argument("--preds_rank", default="outputs/train/rank/tables/preds_rank_2025_eval.csv")
    ap.add_argument("--bet_type", default=BET_TYPE_WIN)
    ap.add_argument("--thresholds", default="")
    ap.add_argument("--ensemble_weights", default="win:1,place:1,rank:1")
    args = ap.parse_args()

    main(
        return_path=args.return_path,
        out_dir=args.out_dir,
        preds_win=args.preds_win,
        preds_place=args.preds_place,
        preds_rank=args.preds_rank,
        bet_type=args.bet_type,
        thresholds=args.thresholds if args.thresholds else None,
        ensemble_weights=args.ensemble_weights,
    )
